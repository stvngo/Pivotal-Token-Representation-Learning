#!/usr/bin/env python
"""Fit the probes and directions that Phase 6 steers with, and export them.

Everything here is fit on **train only**; the test questions are never
touched, so the causal evaluation is not reporting on rows that chose its
hyperparameters.

Why one layer for both probes
-----------------------------
The reactive gate has to fire *before* the pivotal token is committed, in
the same forward pass that produces it. That is only possible if the gate
is read at a layer at or below the steering site. The simplest form of
that constraint -- and the one that removes the whole class of off-by-one
hook bugs -- is to put both at the same layer ``L`` and use a **post**-hook
on ``layers[L-1]``, whose output *is* ``hidden_states[L]``: the tensor both
probes were trained on. We read it, decide, perturb it, and the rest of
the stack computes the pivotal token's logits from the perturbed residual.

``L`` is selected on an inner question-split of train to maximise
``min(AUROC_unsigned, AUROC_signed)`` -- the cascade is only as good as its
weaker half, so the max-of-min is the honest objective, and it is one
number fixed before any intervention runs.

    python scripts/fit_steering_probes.py --acts data/acts_4b_tm1 \
        --tag qwen3-4b --out artifacts/steering/qwen3-4b.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_pipeline.activations_v2 import ActivationStoreV2  # noqa: E402
from probe_pipeline.probes import fit_probe, mean_diff_direction, split_groups  # noqa: E402


def qgroups(store: ActivationStoreV2) -> np.ndarray:
    """Question id per row, with the rollout-branch suffix stripped.

    Branches of one question share a prefix and are not independent, so the
    cluster unit is the question, not the branch.
    """
    ids = np.array(store.query_ids, dtype=object)[store.query_index()]
    return np.array([str(q).split("#b")[0] for q in ids])


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--c-grid", default="0.0001,0.001,0.01,0.1,1.0")
    ap.add_argument("--dead-zone", type=float, default=0.0)
    ap.add_argument("--inner-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=None,
                    help="skip the sweep and use this layer (with --C-unsigned "
                         "/ --C-signed). Use it to re-emit the export after a "
                         "sweep has already chosen the hyperparameters.")
    ap.add_argument("--C-unsigned", type=float, default=None)
    ap.add_argument("--C-signed", type=float, default=None)
    ap.add_argument("--n-check", type=int, default=128,
                    help="train rows stored alongside the weights so a run on "
                         "another machine can assert the gate reproduces the "
                         "probe's own logits before it steers anything")
    ap.add_argument("--min-rel-depth", type=float, default=0.3,
                    help="skip the earliest layers: a direction there is "
                         "mostly token identity, which we are trying to rule out")
    a = ap.parse_args()

    acts = Path(a.acts)
    tr = ActivationStoreV2.open(acts / f"{a.tag}_train.safetensors")
    g_tr = qgroups(tr)
    c_grid = [float(c) for c in a.c_grid.split(",")]

    y_uns = tr.xy(tr.layers[0])[1]
    m_piv = (tr.labels() > 0) & (np.abs(tr.prob_delta()) > a.dead_zone)
    g_sgn = g_tr[m_piv]
    y_sgn = tr.signed_xy(tr.layers[0], dead_zone=a.dead_zone)[1]

    fit_u, sel_u = split_groups(g_tr, frac=a.inner_frac, seed=a.seed)
    fit_s, sel_s = split_groups(g_sgn, frac=a.inner_frac, seed=a.seed)

    n_layers = max(tr.layers)
    cands = [L for L in tr.layers if L >= a.min_rel_depth * n_layers]

    if a.layer is not None and a.C_unsigned is not None and a.C_signed is not None:
        rows = []
        pick = {"layer": int(a.layer), "C_unsigned": a.C_unsigned,
                "C_signed": a.C_signed, "auroc_unsigned": float("nan"),
                "auroc_signed": float("nan"), "objective": float("nan")}
        print(f"using given layer {a.layer} (sweep skipped)")
        return _export(a, tr, pick, rows, y_uns, y_sgn)

    rows = []
    for L in cands:
        xu = tr.layer(L)
        xs = tr.signed_xy(L, dead_zone=a.dead_zone)[0]
        best_u = max(
            (auroc(y_uns[sel_u], fit_probe(xu[fit_u], y_uns[fit_u], C=C, seed=a.seed).decision(xu[sel_u])), C)
            for C in c_grid
        )
        best_s = max(
            (auroc(y_sgn[sel_s], fit_probe(xs[fit_s], y_sgn[fit_s], C=C, seed=a.seed).decision(xs[sel_s])), C)
            for C in c_grid
        )
        rows.append({
            "layer": int(L),
            "auroc_unsigned": best_u[0], "C_unsigned": best_u[1],
            "auroc_signed": best_s[0], "C_signed": best_s[1],
            "objective": min(best_u[0], best_s[0]),
        })
        print(f"  L{L:>3}  unsigned {best_u[0]:.3f} (C={best_u[1]:g})  "
              f"signed {best_s[0]:.3f} (C={best_s[1]:g})  min {rows[-1]['objective']:.3f}",
              flush=True)

    pick = max(rows, key=lambda r: r["objective"])
    print(f"\nselected layer {pick['layer']}: unsigned {pick['auroc_unsigned']:.3f}, "
          f"signed {pick['auroc_signed']:.3f}")
    _export(a, tr, pick, rows, y_uns, y_sgn)


def _export(a, tr, pick, rows, y_uns, y_sgn) -> None:
    L = pick["layer"]

    # Refit on ALL of train at the chosen layer -- the inner split existed
    # only to choose (L, C), and throwing away 30% of the fitting data
    # afterwards would weaken the direction for no statistical gain.
    xu, xs = tr.layer(L), tr.signed_xy(L, dead_zone=a.dead_zone)[0]
    p_uns = fit_probe(xu, y_uns, C=pick["C_unsigned"], layer=L, seed=a.seed)
    p_sgn = fit_probe(xs, y_sgn, C=pick["C_signed"], layer=L, seed=a.seed)

    # Directions. The signed mean-difference is the one Phase 6 steers along:
    # its polarity is meaningful by construction (helpful minus harmful),
    # unlike the unsigned direction, whose label is |prob_delta| > tau and so
    # has no good end. The unsigned direction is kept only as a control arm.
    v_signed = mean_diff_direction(xs, y_sgn)
    v_unsigned = mean_diff_direction(xu, y_uns)

    # A handful of real train activations, so the VM-side run can assert the
    # exported weights still reproduce the probe's logits. The previous gate
    # applied standardized coefficients to raw activations and nothing caught
    # it; this makes that failure loud instead of silent.
    idx = np.random.default_rng(a.seed).choice(
        len(xu), size=min(a.n_check, len(xu)), replace=False
    )
    check_x = xu[idx].astype(np.float32)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        layer=np.int64(L),
        gate_w=p_uns.w, gate_b=np.float32(p_uns.b),
        sign_w=p_sgn.w, sign_b=np.float32(p_sgn.b),
        v_signed=v_signed, v_unsigned=v_unsigned,
        v_signed_probe=p_sgn.direction, v_unsigned_probe=p_uns.direction,
        act_norm_mean=np.float32(np.linalg.norm(xu, axis=1).mean()),
        check_x=check_x, check_gate=p_uns.decision(check_x).astype(np.float32),
        check_sign=p_sgn.decision(check_x).astype(np.float32),
    )

    meta = {
        "tag": a.tag, "acts": str(a.acts), "layer": L,
        "C_unsigned": pick["C_unsigned"], "C_signed": pick["C_signed"],
        "inner_auroc_unsigned": pick["auroc_unsigned"],
        "inner_auroc_signed": pick["auroc_signed"],
        "n_train_unsigned": int(len(y_uns)), "n_train_signed": int(len(y_sgn)),
        "signed_positive_rate": float(np.mean(y_sgn)),
        "cos_signed_probe_to_caa": float(
            p_sgn.direction @ (v_signed / np.linalg.norm(v_signed))
        ),
        "cos_signed_to_unsigned_caa": float(
            (v_signed / np.linalg.norm(v_signed)) @ (v_unsigned / np.linalg.norm(v_unsigned))
        ),
        "act_norm_mean": float(np.linalg.norm(xu, axis=1).mean()),
        "sweep": rows,
    }
    # The fast path skips the sweep, so carry forward the record of how the
    # layer was chosen rather than silently dropping it.
    prior = out.with_suffix(".json")
    if not rows and prior.exists():
        old = json.loads(prior.read_text())
        meta["sweep"] = old.get("sweep", [])
        for k in ("inner_auroc_unsigned", "inner_auroc_signed"):
            if np.isnan(meta[k]):
                meta[k] = old.get(k)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    print(f"  cos(signed probe, signed CAA)   = {meta['cos_signed_probe_to_caa']:+.3f}")
    print(f"  cos(signed CAA, unsigned CAA)   = {meta['cos_signed_to_unsigned_caa']:+.3f}")
    print(f"  mean ||h|| at layer {L}          = {meta['act_norm_mean']:.2f}")


if __name__ == "__main__":
    main()
