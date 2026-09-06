#!/usr/bin/env python
"""Paired comparison of the probe against each cheaper explanation.

Comparing two marginal confidence intervals is badly conservative: two
detectors can have heavily overlapping intervals and still be reliably
ordered, or non-overlapping ones and not be. The right test resamples the
same questions for both and bootstraps the *difference*.

    python scripts/compare_baselines.py --acts data/acts_v2_full --tag qwen3-0.6b

Layer and C are selected on an inner split of train unless given.
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
from probe_pipeline.baselines import (  # noqa: E402
    random_direction_scores,
    token_identity_scores,
    token_onehot_probe_scores,
)
from probe_pipeline.probes import fit_probe, mean_diff_direction  # noqa: E402


def qgroups(store: ActivationStoreV2) -> np.ndarray:
    ids = np.array(store.query_ids, dtype=object)[store.query_index()]
    return np.array([str(q).split("#b")[0] for q in ids])


def paired_diff(y, a, b, grp, n_boot=4000, seed=0):
    """Bootstrap AUROC(a) - AUROC(b) over resampled questions."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(grp)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(grp == g) for g in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        out.append(roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx]))
    out = np.asarray(out)
    if out.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return out.mean(), *np.percentile(out, [2.5, 97.5]), float((out <= 0).mean())


def select(layer_x, y, groups, c_grid, seed=0, frac=0.3):
    from probe_pipeline.probes import split_groups

    fit_m, sel_m = split_groups(groups, frac=frac, seed=seed)
    best = (-1.0, None, None)
    for L, x in sorted(layer_x.items()):
        for C in c_grid:
            p = fit_probe(x[fit_m], y[fit_m], C=C, layer=L, seed=seed)
            s = p.decision(x[sel_m])
            sc = roc_auc_score(y[sel_m], s) if len(np.unique(y[sel_m])) > 1 else 0.5
            if sc > best[0]:
                best = (sc, L, C)
    return int(best[1]), float(best[2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--tag", default="qwen3-0.6b")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--c-grid", default="0.0001,0.001,0.01,0.1,1.0")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--signed", action="store_true",
                    help="the signed task: pivotal rows only, helpful vs harmful. "
                         "Entropy is sign-blind by construction, so a result here "
                         "cannot be explained by uncertainty.")
    ap.add_argument("--dead-zone", type=float, default=0.0,
                    help="drop pivotal rows with |prob_delta| <= this")
    ap.add_argument("--label", default="", help="name for this dataset in the output")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    acts = Path(a.acts)
    tr = ActivationStoreV2.open(acts / f"{a.tag}_train.safetensors")
    te = ActivationStoreV2.open(acts / f"{a.tag}_test.safetensors")
    g_tr, g_te = qgroups(tr), qgroups(te)

    if a.signed:
        # Restrict to pivotal rows and relabel by the sign of the shift.
        # Groups must be subset in the same row order signed_xy uses.
        m_tr = (tr.labels() > 0) & (np.abs(tr.prob_delta()) > a.dead_zone)
        m_te = (te.labels() > 0) & (np.abs(te.prob_delta()) > a.dead_zone)
        g_tr, g_te = g_tr[m_tr], g_te[m_te]
        ytr = tr.signed_xy(tr.layers[0], dead_zone=a.dead_zone)[1]
        yte = te.signed_xy(te.layers[0], dead_zone=a.dead_zone)[1]
        tok_tr, tok_te = tr.token_ids()[m_tr], te.token_ids()[m_te]
        unc_te = {k: v[m_te] for k, v in te.uncertainty().items()}
        get_tr = lambda L: tr.signed_xy(L, dead_zone=a.dead_zone)[0]  # noqa: E731
        get_te = lambda L: te.signed_xy(L, dead_zone=a.dead_zone)[0]  # noqa: E731
    else:
        ytr = tr.xy(tr.layers[0])[1]
        yte = te.xy(te.layers[0])[1]
        tok_tr, tok_te = tr.token_ids(), te.token_ids()
        unc_te = te.uncertainty()
        get_tr = lambda L: tr.xy(L)[0]  # noqa: E731
        get_te = lambda L: te.xy(L)[0]  # noqa: E731

    if a.layer is None or a.C is None:
        L, C = select(
            {L: get_tr(L) for L in tr.layers}, ytr, g_tr,
            [float(c) for c in a.c_grid.split(",")],
        )
    else:
        L, C = a.layer, a.C

    xtr = get_tr(L)
    probe = fit_probe(xtr, ytr, C=C, seed=0)
    xte = get_te(L)
    s_probe = probe.decision(xte)
    v_caa = mean_diff_direction(xtr, ytr)

    baselines = {
        "token_identity_freq": token_identity_scores(tok_tr, ytr, tok_te),
        "token_identity_lr": token_onehot_probe_scores(tok_tr, ytr, tok_te),
        "entropy": unc_te["entropy"],
        "neg_margin": -unc_te["margin"],
        "neg_top1_prob": -unc_te["top1_prob"],
        "caa_direction": xte @ v_caa,
        "random_direction": random_direction_scores(xte, seed=0),
    }

    head = (a.label or str(acts)) + (" [SIGNED: helpful vs harmful]" if a.signed else "")
    print(f"\n=== {head} ===")
    print(f"layer {L}, C={C} | {len(yte)} rows from {len(set(g_te))} questions"
          + (f" | positive rate {yte.mean():.3f}" if a.signed else ""))
    print(f"probe AUROC = {roc_auc_score(yte, s_probe):.4f}\n")
    print(f"{'baseline':<20} {'AUROC':>7} {'probe-base':>11} {'95% CI':>18} {'P(<=0)':>8}")

    report = {"dataset": head, "layer": L, "C": C,
              "n_rows": int(len(yte)), "n_questions": int(len(set(g_te))),
              "probe_auroc": float(roc_auc_score(yte, s_probe)), "baselines": {}}
    for name, sc in baselines.items():
        sc = np.asarray(sc, dtype=float)
        d, lo, hi, p = paired_diff(yte, s_probe, sc, g_te, n_boot=a.n_boot)
        auc = float(roc_auc_score(yte, sc))
        sep = "" if lo > 0 else "  not separated"
        print(f"{name:<20} {auc:7.4f} {d:+11.4f} [{lo:+.3f}, {hi:+.3f}] {p:8.3f}{sep}")
        report["baselines"][name] = {
            "auroc": auc, "diff": float(d), "ci_lo": float(lo),
            "ci_hi": float(hi), "p_le_0": float(p), "separated": bool(lo > 0),
        }

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
