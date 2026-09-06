#!/usr/bin/env python
"""The equalised causal table: three configurations, three scales, one n.

Every run in this study was evaluated on whatever in-band questions that
model happened to have, which made the scales incomparable -- 275, 149 and
336. This assembles the equalised results at a common n, pooling the two
halves where a model's band had to be drawn from both GSM8K splits.

Only the pre-registered primary arm of each configuration carries a
p-value. The controls are shown with net flips so their size and direction
are visible without inviting a reader to pick a winner out of a grid.

    python scripts/causal_summary.py --out paper/neurips2026/causal_table.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steering_report import pool  # noqa: E402

# Anchored to the repo, not the caller's directory: this is run from the
# repo root and from scripts/ and must find the same artifacts either way.
ART = Path(__file__).resolve().parent.parent / "artifacts" / "steering"

# (configuration label, filename stem). A stem may resolve to one file or to
# a train/test pair that has to be pooled.
CONFIGS = [
    ("cascade gate, signed direction, add", "eq_primary"),
    ("$P(\\text{pivotal})$ gate, probe weights, add", "eq_simple"),
    ("cascade gate, signed direction, ablate", "eq_ablate"),
]
MODELS = [("Qwen3-0.6B", "0.6B"), ("Qwen3-1.7B", "1.7B"), ("Qwen3-4B", "4B")]
PRIMARY = {"eq_primary": "reactive", "eq_simple": "reactive",
           "eq_ablate": "ablate_reactive"}


def load(stem: str, model: str) -> dict | None:
    single = ART / f"{stem}_{model}.json"
    if single.exists():
        return json.loads(single.read_text())
    parts = [ART / f"{stem}_{model}_{half}.json" for half in ("train", "test")]
    have = [json.loads(p.read_text()) for p in parts if p.exists()]
    if not have:
        return None
    if len(have) == 1:
        return have[0]
    return pool(have)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows, missing = [], []
    for label, stem in CONFIGS:
        cells = []
        for full, short in MODELS:
            d = load(stem, full)
            if d is None:
                missing.append(f"{stem}/{full}")
                cells.append(("--", "--", None))
                continue
            arm = d.get(PRIMARY[stem])
            if arm is None:
                cells.append(("--", "--", None))
                continue
            cells.append((f"{arm['delta_acc']:+.3f}",
                          f"{arm['vs_base']['net']:+d}",
                          arm["vs_base"]["p"]))
        rows.append((label, cells))

    print(f"{'configuration':<46} " + "".join(f"{s:>22}" for _, s in MODELS))
    for label, cells in rows:
        plain = label.replace("$P(\\text{pivotal})$", "P(pivotal)")
        cs = "".join(f"{d:>9} {n:>5} {('p=%.3f' % p) if p is not None else '':>8}"
                     for d, n, p in cells)
        print(f"{plain:<46} {cs}")

    ns = []
    for full, _ in MODELS:
        d = load("eq_primary", full)
        ns.append(d["base"]["n"] if d else 0)
    print(f"\nquestions: {dict(zip([s for _, s in MODELS], ns))}")
    if len(set(ns)) > 1:
        print(f"WARNING: sample sizes differ {ns}; the table is not equalised yet")
    if missing:
        print("missing runs: " + ", ".join(missing))

    if not a.out:
        return
    body = []
    for label, cells in rows:
        cs = " & ".join(f"${d}$ & {n} & {('%.3f' % p) if p is not None else '--'}"
                        for d, n, p in cells)
        body.append(f"{label} & {cs} \\\\")
    tex = "\n".join([
        "\\begin{tabular}{l" + "rrr" * len(MODELS) + "}",
        "\\toprule",
        " & " + " & ".join("\\multicolumn{3}{c}{%s}" % s for _, s in MODELS) + " \\\\",
        "configuration & " + " & ".join(["$\\Delta$ & net & $p$"] * len(MODELS)) + " \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
    ])
    Path(a.out).write_text(tex)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
