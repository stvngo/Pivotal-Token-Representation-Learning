#!/usr/bin/env python
"""Turn steer_eval outputs into the table and curve the paper reports.

Deliberately narrow about significance. Only the pre-registered primary arm
gets a p-value in the table; every other arm is shown with its net flips so
a reader can see the size and direction without being invited to read a
p-value off a grid of twelve. The project's earlier round tested 114 arms
and found 4 nominal hits where chance predicts 5.7, and the fix for that is
not a correction factor, it is not quoting the p-values.

    python scripts/steering_report.py artifacts/steering/steer_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Display order, and the question each arm answers.
ARMS = [
    ("base", "unsteered"),
    ("reactive", "reactive, signed direction (primary)"),
    ("always_on_matched_energy", "always-on, matched energy"),
    ("random_placement_matched", "random placement, matched duty"),
    ("sign_flipped", "coefficient negated"),
    ("unsigned_direction", "unsigned direction"),
    ("unsigned_gate", "gate on $P(\\text{pivotal})$ only"),
    ("random_direction", "random direction"),
]


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    d["_path"] = str(path)
    return d


def fmt(d: dict, key: str) -> str:
    a = d.get(key)
    if a is None:
        return "--"
    acc = a["accuracy"]
    if key == "base":
        return f"{acc:.3f} & -- & -- & -- & --"
    duty = a.get("duty_cycle_trimmed", 0.0)
    net = a["vs_base"]["net"]
    p = f"{a['vs_base']['p']:.3f}" if a.get("primary") else "--"
    return f"{acc:.3f} & {a['delta_acc']:+.3f} & {duty:.3f} & {net:+d} & {p}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None, help="write the LaTeX table here")
    a = ap.parse_args()

    runs = [load(Path(f)) for f in a.files]

    for d in runs:
        m = d.get("_meta", {})
        name = m.get("model", d["_path"])
        obs = d.get("observe", {})
        print(f"\n=== {name} ===")
        print(f"  band questions   {d['base']['n']}")
        print(f"  identity check   {d.get('identity_check')}")
        print(f"  decode positions {obs.get('n_decode_positions')}")
        for k in ("p_pivotal", "p_helpful", "p_steer"):
            v = obs.get(k)
            if v:
                print(f"  {k:<10} mean {v['mean']:.3f} sd {v['sd']:.3f} "
                      f"p95 {v['p95']:.3f} max {v['max']:.3f}")
        em = d.get("energy_match")
        if em:
            print(f"  energy match     duty {em['duty_estimate']:.3f} -> "
                  f"coefficient scaled by {em['energy_fraction']:.4f}")
        print(f"  {'arm':<38} {'acc':>6} {'delta':>7} {'duty':>6} "
              f"{'energy':>8} {'net':>5} {'p':>6}")
        for key, label in ARMS:
            arm = d.get(key)
            if arm is None:
                continue
            if key == "base":
                print(f"  {label:<38} {arm['accuracy']:>6.3f}")
                continue
            print(f"  {label:<38} {arm['accuracy']:>6.3f} "
                  f"{arm['delta_acc']:>+7.3f} "
                  f"{arm.get('duty_cycle_trimmed', 0):>6.3f} "
                  f"{arm.get('energy', 0):>8.0f} "
                  f"{arm['vs_base']['net']:>+5d} "
                  f"{arm['vs_base']['p']:>6.3f}")
        dose = sorted(k for k in d if k.startswith("dose_"))
        if dose:
            print("  dose response (secondary; no per-arm inference)")
            for k in dose:
                arm = d[k]
                print(f"    {k:<36} {arm['accuracy']:>6.3f} "
                      f"{arm['delta_acc']:>+7.3f} "
                      f"{arm.get('duty_cycle_trimmed', 0):>6.3f}")

    if not a.out:
        return

    rows = []
    for key, label in ARMS:
        cells = " & ".join(fmt(d, key) for d in runs)
        rows.append(f"{label} & {cells} \\\\")
    span = "r" * (5 * len(runs))
    header = " & ".join(
        "\\multicolumn{5}{c}{%s}" % d["_meta"]["model"].split("/")[-1] for d in runs
    )
    sub = " & ".join(["acc & $\\Delta$ & duty & net & $p$"] * len(runs))
    table = "\n".join([
        "\\begin{tabular}{l" + span + "}",
        "\\toprule",
        f" & {header} \\\\",
        f"arm & {sub} \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ])
    Path(a.out).write_text(table)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
