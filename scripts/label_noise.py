#!/usr/bin/env python
"""How much of the PTS positive class is sampling noise?

PTS accepts a token as pivotal when |p_after - p_before| exceeds tau, and
both probabilities are estimated from a finite number of rollouts. At the
common setting the acceptance threshold sits only about two standard
errors out, so a meaningful share of any positive class is expected to be
noise. The paper states this analytically; this measures it.

Because the search stores sufficient statistics (n_before, s_before,
n_after, s_after) rather than a verdict, every accepted event can be
re-tested after the fact with no GPU:

* a two-proportion z-test of H0: p_after == p_before, giving a per-event
  p-value and hence an empirical false-discovery estimate;
* the acceptance threshold tau re-applied at other values, since the
  underlying counts are intact.

The useful output is not the noise rate on its own but the pairing: if the
probe scores markedly better on the high-confidence subset, then part of
the gap between probe AUROC and 1.0 is the oracle's noise rather than the
probe's failure.

    python scripts/label_noise.py --events runs/qwen3-4b-full/events.jsonl \
        --acts data/acts_4b_tm1 --tag qwen3-4b --probes artifacts/steering/qwen3-4b.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_pipeline.activations_v2 import ActivationStoreV2  # noqa: E402
from probe_pipeline.artifacts_io import iter_jsonl  # noqa: E402
from probe_pipeline.probes import cluster_bootstrap_ci, fit_probe  # noqa: E402


def two_proportion_p(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided z-test for equality of two binomial proportions.

    Pooled-variance form, which is the standard choice under the null that
    the two proportions are equal -- the null we are testing.
    """
    if n1 == 0 or n2 == 0:
        return 1.0
    p_pool = (s1 + s2) / (n1 + n2)
    if p_pool in (0.0, 1.0):
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (s2 / n2 - s1 / n1) / se
    return math.erfc(abs(z) / math.sqrt(2))


def qgroups(store: ActivationStoreV2) -> np.ndarray:
    ids = np.array(store.query_ids, dtype=object)[store.query_index()]
    return np.array([str(q).split("#b")[0] for q in ids])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="per-event significance for the high-confidence subset")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ev = list(iter_jsonl(Path(a.events)))
    pv = np.array([two_proportion_p(e["s_before"], e["n_before"],
                                    e["s_after"], e["n_after"]) for e in ev])
    delta = np.array([e["prob_delta"] for e in ev])

    out = {
        "tag": a.tag, "n_events": len(ev),
        "p_value_quantiles": {q: float(np.quantile(pv, q))
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "frac_p_gt_0.05": float((pv > 0.05).mean()),
        "frac_p_gt_0.01": float((pv > 0.01).mean()),
        "frac_p_le_alpha": float((pv <= a.alpha).mean()),
        "median_abs_delta_all": float(np.abs(delta).mean()),
        "median_abs_delta_confident": float(np.abs(delta[pv <= a.alpha]).mean()),
    }

    print(f"\n=== {a.tag}: {len(ev)} accepted events ===")
    print(f"  a two-proportion test does not reject at 0.05 for "
          f"{out['frac_p_gt_0.05']*100:.1f}% of them")
    print(f"  survives alpha<={a.alpha}: {out['frac_p_le_alpha']*100:.1f}% "
          f"({int((pv <= a.alpha).sum())} events)")
    print(f"  mean |prob_delta|  all {out['median_abs_delta_all']:.3f}  "
          f"confident {out['median_abs_delta_confident']:.3f}")

    # Does the probe do better where the label is trustworthy? Rows are keyed
    # by (query, position); an event's t-1 row is at position-1.
    conf = {(str(e["query_uid"]), int(e["position"]) - 1)
            for e, p in zip(ev, pv) if p <= a.alpha}

    layer = int(json.loads(Path(a.probes).with_suffix(".json").read_text())["layer"])
    tr = ActivationStoreV2.open(Path(a.acts) / f"{a.tag}_train.safetensors")
    te = ActivationStoreV2.open(Path(a.acts) / f"{a.tag}_test.safetensors")
    xtr, ytr = tr.xy(layer)
    xte, yte = te.xy(layer)
    probe = fit_probe(xtr, ytr, C=1e-3, seed=0)
    s = probe.decision(xte)
    g = qgroups(te)

    uid = np.array([str(q).split("#b")[0] for q in
                    np.array(te.query_ids, dtype=object)[te.query_index()]])
    pos = te.token_positions()
    is_conf = np.array([(u, int(pp)) in conf for u, pp in zip(uid, pos)])

    # Keep every negative; drop only the positives whose label is shaky.
    keep = (yte == 0) | is_conf
    full = cluster_bootstrap_ci(yte, s, g, n_boot=a.n_boot)
    sub = cluster_bootstrap_ci(yte[keep], s[keep], g[keep], n_boot=a.n_boot)
    out["auroc_all_labels"] = {"auroc": full[0], "ci_lo": full[1], "ci_hi": full[2],
                               "n_pos": int((yte == 1).sum())}
    out["auroc_confident_only"] = {"auroc": sub[0], "ci_lo": sub[1], "ci_hi": sub[2],
                                   "n_pos": int((yte[keep] == 1).sum())}
    print(f"  probe AUROC  all labels {full[0]:.3f} [{full[1]:.3f}, {full[2]:.3f}] "
          f"({int((yte == 1).sum())} positives)")
    print(f"               confident  {sub[0]:.3f} [{sub[1]:.3f}, {sub[2]:.3f}] "
          f"({int((yte[keep] == 1).sum())} positives)")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
