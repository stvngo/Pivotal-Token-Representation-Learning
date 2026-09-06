#!/usr/bin/env python
"""What the causal experiment's detectors score on held-out questions.

The steering layer is chosen by max-min of the unsigned and signed AUROCs,
which is not either task's own optimum. So the gate the intervention fires
on is *not* the paper's headline detector, and a reader needs the number it
actually achieves rather than the one from a different layer.

Kept separate from ``fit_steering_probes.py`` on purpose. That script
selects hyperparameters and must never see test; this one only reports, on
a layer and C already fixed. Merging them would put a selection loop and a
test-set evaluation in one file, which is how leakage starts.

    python scripts/gate_test_auroc.py --acts data/acts_4b_tm1 --tag qwen3-4b \
        --probes artifacts/steering/qwen3-4b.npz
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
from probe_pipeline.probes import cluster_bootstrap_ci, fit_probe  # noqa: E402


def qgroups(store: ActivationStoreV2) -> np.ndarray:
    ids = np.array(store.query_ids, dtype=object)[store.query_index()]
    return np.array([str(q).split("#b")[0] for q in ids])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--dead-zone", type=float, default=0.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.loads(Path(a.probes).with_suffix(".json").read_text())
    L = int(meta["layer"])
    acts = Path(a.acts)
    tr = ActivationStoreV2.open(acts / f"{a.tag}_train.safetensors")
    te = ActivationStoreV2.open(acts / f"{a.tag}_test.safetensors")

    out = {"tag": a.tag, "steering_layer": L,
           "C_unsigned": meta["C_unsigned"], "C_signed": meta["C_signed"]}

    # Unsigned gate, at the steering layer.
    xtr, ytr = tr.xy(L)
    xte, yte = te.xy(L)
    p = fit_probe(xtr, ytr, C=meta["C_unsigned"], seed=0)
    s = p.decision(xte)
    auc, lo, hi = cluster_bootstrap_ci(yte, s, qgroups(te), n_boot=a.n_boot)
    out["gate"] = {"auroc": auc, "ci_lo": lo, "ci_hi": hi,
                   "n_rows": int(len(yte)), "n_questions": int(len(set(qgroups(te))))}

    # Signed probe, at the same layer.
    m_tr = (tr.labels() > 0) & (np.abs(tr.prob_delta()) > a.dead_zone)
    m_te = (te.labels() > 0) & (np.abs(te.prob_delta()) > a.dead_zone)
    xstr, ystr = tr.signed_xy(L, dead_zone=a.dead_zone)
    xste, yste = te.signed_xy(L, dead_zone=a.dead_zone)
    ps = fit_probe(xstr, ystr, C=meta["C_signed"], seed=0)
    ss = ps.decision(xste)
    g_ste = qgroups(te)[m_te]
    aucs, los, his = cluster_bootstrap_ci(yste, ss, g_ste, n_boot=a.n_boot)
    out["sign"] = {"auroc": aucs, "ci_lo": los, "ci_hi": his,
                   "n_rows": int(len(yste)), "n_questions": int(len(set(g_ste)))}

    # The cascade is what actually gates, so score it directly: a position is
    # "should fire" when the next token is a pivot AND the shift is harmful.
    # Only defined on the unsigned row set, where both labels exist.
    delta = te.prob_delta()
    y_harmful = ((yte == 1) & (delta < 0)).astype(int)
    p_piv = 1.0 / (1.0 + np.exp(-s))
    p_help = 1.0 / (1.0 + np.exp(-ps.decision(xte)))
    cascade = p_piv * (1.0 - p_help)
    if len(np.unique(y_harmful)) > 1:
        ac, lc, hc = cluster_bootstrap_ci(y_harmful, cascade, qgroups(te), n_boot=a.n_boot)
        out["cascade_vs_harmful_pivot"] = {
            "auroc": ac, "ci_lo": lc, "ci_hi": hc,
            "positive_rate": float(y_harmful.mean()),
        }

    print(f"\n=== {a.tag} at the steering layer (L{L}) ===")
    print(f"  gate  (unsigned, C={meta['C_unsigned']:g})  AUROC {out['gate']['auroc']:.3f} "
          f"[{out['gate']['ci_lo']:.3f}, {out['gate']['ci_hi']:.3f}]  "
          f"{out['gate']['n_rows']} rows / {out['gate']['n_questions']} questions")
    print(f"  sign  (signed,   C={meta['C_signed']:g})  AUROC {out['sign']['auroc']:.3f} "
          f"[{out['sign']['ci_lo']:.3f}, {out['sign']['ci_hi']:.3f}]  "
          f"{out['sign']['n_rows']} rows / {out['sign']['n_questions']} questions")
    c = out.get("cascade_vs_harmful_pivot")
    if c:
        print(f"  cascade vs 'next token is a harmful pivot'  AUROC {c['auroc']:.3f} "
              f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]  base rate {c['positive_rate']:.3f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
