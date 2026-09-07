#!/usr/bin/env python
"""The trajectory PTS approximates, measured at every token.

The sparse figure has to be drawn as points because bisection only
evaluates a handful of prefixes. That leaves the obvious question open:
between those points, is the probability smooth, or is the search stepping
over structure it never sees? For one branch the question is answerable --
re-estimate every prefix from scratch and overlay the two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

C_DENSE, C_HELP, C_HARM, C_MISS = "#4a4a4a", "#0072B2", "#D55E00", "#9aa0a6"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="artifacts/dense_sweep_q355.json")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--out", default="paper/neurips2026/figures/dense_vs_bisection.pdf")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads((ROOT / a.sweep).read_text())
    curve = {int(k): v["p"] for k, v in d["curve"].items()}
    xs = sorted(curve)
    ys = [curve[x] for x in xs]
    plen = d["prompt_len"]

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.axhspan(0.2, 0.8, color="#9aa0a6", alpha=0.14, lw=0)
    ax.plot(xs, ys, color=C_DENSE, lw=1.1, zorder=2,
            label=f"every prefix re-estimated ({len(xs)} positions)")

    # What the search actually saw: two estimates per accepted event.
    sx, sy = [plen], [d["events"][0].get("prob_before", ys[0])]
    for e in d["events"]:
        sx += [e["position"], e["position"] + 1]
        sy += [e["prob_before"], e["prob_after"]]
    order = sorted(range(len(sx)), key=lambda i: sx[i])
    ax.scatter([sx[i] for i in order], [sy[i] for i in order], s=26,
               facecolor="white", edgecolor=C_DENSE, lw=1.0, zorder=4,
               label="estimates the search made")

    for e in d["events"]:
        p = e["position"]
        dense_step = curve.get(p + 1, 0) - curve.get(p, 0)
        replicated = abs(dense_step) >= a.tau
        c = (C_HELP if e["prob_delta"] > 0 else C_HARM) if replicated else C_MISS
        ax.annotate("", xy=(p + 1, e["prob_after"]), xytext=(p, e["prob_before"]),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.9,
                                    shrinkA=0, shrinkB=0,
                                    linestyle="-" if replicated else ":"), zorder=5)
        if not replicated:
            ax.annotate(f"accepted at $\\Delta p={e['prob_delta']:+.2f}$,\n"
                        f"re-measures {dense_step:+.2f}",
                        xy=(p, max(e["prob_before"], e["prob_after"])),
                        xytext=(-96, 16), textcoords="offset points",
                        fontsize=6.5, color=C_MISS, ha="left",
                        arrowprops=dict(arrowstyle="-", color=C_MISS, lw=0.5))

    ax.set_xlabel("token position in the sequence", fontsize=9)
    ax.set_ylabel("$P(\\mathrm{success})$", fontsize=9)
    ax.set_ylim(-0.06, 1.12)
    ax.set_xlim(plen - 6, max(xs) + 8)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=C_DENSE, lw=1.1, label=f"every prefix re-estimated ({len(xs)} positions)"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=C_DENSE,
               ms=5, label="estimates the search made"),
        Line2D([], [], color=C_HELP, lw=1.9, label="accepted pivot, replicated"),
        Line2D([], [], color=C_MISS, lw=1.9, ls=":", label="accepted pivot, not replicated"),
    ]
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="lower center",
               ncol=2, bbox_to_anchor=(0.55, -0.13))
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if a.png:
        fig.savefig(a.png, bbox_inches="tight", dpi=110)
    plt.close(fig)

    import numpy as np
    step = np.abs(np.diff(ys))
    big = [xs[i] for i in range(len(step)) if step[i] >= a.tau]
    acc = {e["position"] for e in d["events"]}
    print(f"wrote {out}")
    print(f"dense steps >= tau: {big}  (all accepted: {set(big) <= acc})")
    print(f"accepted but not replicated: {sorted(acc - set(big))}")
    print(f"mean |step| {step.mean():.3f}, p95 {np.percentile(step, 95):.3f}")


if __name__ == "__main__":
    main()
