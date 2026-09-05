#!/usr/bin/env python
"""Honest evaluation of pivotality probes on a v2 activation cache.

Answers, in order:

1. Is pivotality linearly decodable at t-1 above chance?
2. Does it beat token identity and next-token uncertainty?   <- the real test
3. Is any single layer actually special, or is "best layer" selection noise?
4. Is the SIGN of the impending shift decodable (helpful vs harmful)?

Selection discipline: every hyperparameter -- layer, regularization strength,
and the signed probe's layer -- is chosen on an inner split of TRAIN, carved
by question. The test split is touched once, to report. This matters: the
original pipeline picked the best of 29 layers on the same rows it reported,
and the gap that opens up is measured here as `winners_curse`.

    python scripts/evaluate_probes_v2.py --acts data/acts_v2 --tag qwen3-0.6b
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
from probe_pipeline.baselines import evaluate_baselines  # noqa: E402
from probe_pipeline.probes import (  # noqa: E402
    binary_metrics,
    cluster_bootstrap_ci,
    fit_probe,
    mean_diff_direction,
    null_max_distribution,
    split_groups,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--acts", default="data/acts_v2")
    p.add_argument("--tag", default="qwen3-0.6b")
    p.add_argument("--out", default="artifacts/probes_v2")
    p.add_argument("--c-grid", default="0.0001,0.001,0.01,0.1,1.0")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--dead-zone", type=float, default=0.0)
    p.add_argument("--inner-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def qgroups(store: ActivationStoreV2) -> np.ndarray:
    """Question id per row. Branches of one question must cluster together."""
    ids = np.array(store.query_ids, dtype=object)[store.query_index()]
    return np.array([str(q).split("#b")[0] for q in ids])


def select_layer_and_C(
    layer_x: dict[int, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    c_grid: list[float],
    *,
    inner_frac: float,
    seed: int,
) -> tuple[int, float, dict]:
    """Pick (layer, C) on an inner question-split of TRAIN only."""
    fit_mask, sel_mask = split_groups(groups, frac=inner_frac, seed=seed)
    grid: dict[str, float] = {}
    best = (-1.0, None, None)
    for layer, x in sorted(layer_x.items()):
        for C in c_grid:
            probe = fit_probe(x[fit_mask], y[fit_mask], C=C, layer=layer, seed=seed)
            s = probe.decision(x[sel_mask])
            score = auroc(y[sel_mask], s) if len(np.unique(y[sel_mask])) > 1 else 0.5
            grid[f"L{layer}_C{C}"] = score
            if score > best[0]:
                best = (score, layer, C)
    return int(best[1]), float(best[2]), grid


def main() -> None:
    args = parse_args()
    acts = Path(args.acts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_grid = [float(c) for c in args.c_grid.split(",")]

    train = ActivationStoreV2.open(acts / f"{args.tag}_train.safetensors")
    test = ActivationStoreV2.open(acts / f"{args.tag}_test.safetensors")
    g_train, g_test = qgroups(train), qgroups(test)
    layers = train.layers

    y_train = train.xy(layers[0])[1]
    y_test = test.xy(layers[0])[1]
    xtr_all = {L: train.xy(L)[0] for L in layers}
    xte_all = {L: test.xy(L)[0] for L in layers}

    report: dict = {
        "tag": args.tag,
        "model": train.manifest.model_name,
        "convention": {
            "hidden_state": train.manifest.hidden_state_convention,
            "position": train.manifest.extraction_position,
        },
        "n": {
            "train_rows": int(train.manifest.n_rows),
            "test_rows": int(test.manifest.n_rows),
            "train_questions": int(len(set(g_train))),
            "test_questions": int(len(set(g_test))),
            "test_pos_rate": float(y_test.mean()),
        },
    }
    print(json.dumps(report["n"], indent=1))

    # -- 1. selection on TRAIN only ---------------------------------------
    best_layer, best_C, grid = select_layer_and_C(
        xtr_all, y_train, g_train, c_grid, inner_frac=args.inner_frac, seed=args.seed
    )
    report["selection"] = {
        "best_layer": best_layer,
        "best_C": best_C,
        "selected_on": "inner question-split of train",
        "grid_size": len(grid),
    }
    print(f"\nselected on train: layer {best_layer}, C={best_C}")

    # -- 2. per-layer on TEST at the selected C ---------------------------
    per_layer: dict[int, dict] = {}
    for L in layers:
        probe = fit_probe(xtr_all[L], y_train, C=best_C, layer=L, seed=args.seed)
        s = probe.decision(xte_all[L])
        m = binary_metrics(y_test, s)
        pt, lo, hi = cluster_bootstrap_ci(
            y_test, s, g_test, auroc, n_boot=args.n_boot, seed=args.seed
        )
        m.update(auroc_ci_lo=lo, auroc_ci_hi=hi)
        per_layer[L] = m
    report["per_layer"] = per_layer

    accs = [per_layer[L]["accuracy"] for L in layers]
    report["null_max"] = null_max_distribution(
        accs, n_rows=int(test.manifest.n_rows), seed=args.seed
    )
    naive_best = max(per_layer, key=lambda L: per_layer[L]["auroc"])
    report["winners_curse"] = {
        "naive_best_layer": int(naive_best),
        "naive_best_auroc": per_layer[naive_best]["auroc"],
        "selected_layer_auroc": per_layer[best_layer]["auroc"],
        "gap": float(per_layer[naive_best]["auroc"] - per_layer[best_layer]["auroc"]),
    }
    print("null-max:", json.dumps(report["null_max"], indent=1))
    print("winners curse:", json.dumps(report["winners_curse"], indent=1))

    # -- 3. baselines at the selected layer -------------------------------
    xtr, xte = xtr_all[best_layer], xte_all[best_layer]
    probe = fit_probe(xtr, y_train, C=best_C, layer=best_layer, seed=args.seed)
    s_probe = probe.decision(xte)

    bl = evaluate_baselines(
        y_train=y_train,
        y_eval=y_test,
        token_ids_train=train.token_ids(),
        token_ids_eval=test.token_ids(),
        uncertainty_eval=test.uncertainty(),
        x_eval=xte,
        seed=args.seed,
    )
    v_caa = mean_diff_direction(xtr, y_train)
    bl["caa_direction"] = binary_metrics(y_test, xte @ v_caa)
    bl["probe"] = binary_metrics(y_test, s_probe)

    # Clustered CIs for every baseline, so "beats" means something.
    for name, m in bl.items():
        score = {
            "probe": s_probe,
            "caa_direction": xte @ v_caa,
        }.get(name)
        if score is None:
            continue
        _, lo, hi = cluster_bootstrap_ci(
            y_test, score, g_test, auroc, n_boot=args.n_boot, seed=args.seed
        )
        m["auroc_ci_lo"], m["auroc_ci_hi"] = lo, hi
    report["baselines"] = bl

    print(f"\n=== baselines at layer {best_layer} (C={best_C}) ===")
    for name, m in sorted(bl.items(), key=lambda kv: -kv[1].get("auroc", 0)):
        ci = (
            f" [{m['auroc_ci_lo']:.3f}, {m['auroc_ci_hi']:.3f}]"
            if "auroc_ci_lo" in m
            else ""
        )
        print(f"  {name:<22} auroc {m.get('auroc', float('nan')):.3f}{ci}  acc {m['accuracy']:.3f}")

    # -- 4. C sweep at the selected layer (diagnostic, not selection) ------
    csweep = {}
    for C in c_grid:
        pr = fit_probe(xtr, y_train, C=C, layer=best_layer, seed=args.seed)
        csweep[str(C)] = binary_metrics(y_test, pr.decision(xte)) | {
            "cos_to_caa": float(pr.direction @ (v_caa / np.linalg.norm(v_caa))),
            "w_norm": float(np.linalg.norm(pr.w)),
        }
    report["c_sweep"] = csweep
    print("\nC sweep at selected layer:")
    for C, m in csweep.items():
        print(f"  C={C:<8} auroc {m['auroc']:.3f}  cos_to_caa {m['cos_to_caa']:+.3f}")

    # -- 5. the signed probe: helpful vs harmful --------------------------
    sx_tr = {L: train.signed_xy(L, dead_zone=args.dead_zone) for L in layers}
    sx_te = {L: test.signed_xy(L, dead_zone=args.dead_zone) for L in layers}
    ys_tr = sx_tr[layers[0]][1]
    ys_te = sx_te[layers[0]][1]

    if len(np.unique(ys_tr)) > 1 and len(np.unique(ys_te)) > 1:
        # Groups for the pivotal subset, in the same row order signed_xy uses.
        piv_tr = (train.labels() > 0) & (np.abs(train.prob_delta()) > args.dead_zone)
        piv_te = (test.labels() > 0) & (np.abs(test.prob_delta()) > args.dead_zone)
        gs_tr, gs_te = g_train[piv_tr], g_test[piv_te]

        s_layer, s_C, _ = select_layer_and_C(
            {L: sx_tr[L][0] for L in layers},
            ys_tr,
            gs_tr,
            c_grid,
            inner_frac=args.inner_frac,
            seed=args.seed,
        )
        pr = fit_probe(sx_tr[s_layer][0], ys_tr, C=s_C, layer=s_layer, seed=args.seed)
        s_score = pr.decision(sx_te[s_layer][0])
        m = binary_metrics(ys_te, s_score)
        _, lo, hi = cluster_bootstrap_ci(
            ys_te, s_score, gs_te, auroc, n_boot=args.n_boot, seed=args.seed
        )
        v_signed = mean_diff_direction(sx_tr[s_layer][0], ys_tr)

        # Token identity cannot be ruled out here either -- check it.
        signed_bl = evaluate_baselines(
            y_train=ys_tr,
            y_eval=ys_te,
            token_ids_train=train.token_ids()[piv_tr],
            token_ids_eval=test.token_ids()[piv_te],
            uncertainty_eval={k: v[piv_te] for k, v in test.uncertainty().items()},
            x_eval=sx_te[s_layer][0],
            seed=args.seed,
        )
        signed_bl["signed_probe"] = m | {"auroc_ci_lo": lo, "auroc_ci_hi": hi}
        signed_bl["signed_caa"] = binary_metrics(ys_te, sx_te[s_layer][0] @ v_signed)

        report["signed"] = {
            "n_train": int(len(ys_tr)),
            "n_test": int(len(ys_te)),
            "train_positive_rate": float(ys_tr.mean()),
            "selected_layer": s_layer,
            "selected_C": s_C,
            "baselines": signed_bl,
        }
        print(f"\n=== signed probe (helpful vs harmful), L{s_layer} C={s_C} ===")
        print(f"  n_train={len(ys_tr)} n_test={len(ys_te)} pos_rate={ys_tr.mean():.3f}")
        for name, mm in sorted(signed_bl.items(), key=lambda kv: -kv[1].get("auroc", 0)):
            ci = (
                f" [{mm['auroc_ci_lo']:.3f}, {mm['auroc_ci_hi']:.3f}]"
                if "auroc_ci_lo" in mm
                else ""
            )
            print(f"  {name:<22} auroc {mm.get('auroc', float('nan')):.3f}{ci}")
        np.save(out_dir / f"{args.tag}_signed_w_L{s_layer}.npy", pr.w)
        np.save(out_dir / f"{args.tag}_signed_caa_L{s_layer}.npy", v_signed)

    path = out_dir / f"{args.tag}_report.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    np.save(out_dir / f"{args.tag}_probe_w_L{best_layer}.npy", probe.w)
    np.save(out_dir / f"{args.tag}_caa_L{best_layer}.npy", v_caa)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
