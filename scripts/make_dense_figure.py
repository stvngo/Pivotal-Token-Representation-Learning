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
MUTED_TXT = "#6b6b6b"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps",
                    default="artifacts/dense_sweep_q1223.json",
                    help="one panel per sweep, stacked")
    ap.add_argument("--labels", default="Qwen3-4B, GSM8K")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--out", default="paper/neurips2026/figures/dense_vs_bisection.pdf")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np
    paths = [x for x in a.sweeps.split(",") if x]
    labels = a.labels.split(";")
    fig, axes = plt.subplots(len(paths), 1, figsize=(6.6, 3.0 * len(paths)))
    axes = np.atleast_1d(axes)
    summary = []

    for ax, path, lab in zip(axes, paths, labels):
        d = json.loads((ROOT / path).read_text())
        curve = {int(k): v["p"] for k, v in d["curve"].items()}
        xs = sorted(curve)
        ys = [curve[x] for x in xs]
        plen = d["prompt_len"]

        ax.axhspan(0.2, 0.8, color="#9aa0a6", alpha=0.14, lw=0)
        ax.plot(xs, ys, color=C_DENSE, lw=1.0, zorder=2)

        sx, sy = [], []
        for e in d["events"]:
            sx += [e["position"], e["position"] + 1]
            sy += [e["prob_before"], e["prob_after"]]
        ax.scatter(sx, sy, s=24, facecolor="white", edgecolor=C_DENSE, lw=1.0, zorder=4)

        n_rep = 0
        for e in d["events"]:
            pos = e["position"]
            dstep = curve.get(pos + 1, 0) - curve.get(pos, 0)
            rep = abs(dstep) >= a.tau
            n_rep += rep
            c = (C_HELP if e["prob_delta"] > 0 else C_HARM) if rep else C_MISS
            ax.annotate("", xy=(pos + 1, e["prob_after"]), xytext=(pos, e["prob_before"]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.9, shrinkA=0,
                                        shrinkB=0, linestyle="-" if rep else ":"), zorder=5)
            if not rep:
                ax.annotate(f"accepted at ${e['prob_delta']:+.2f}$, re-measures ${dstep:+.2f}$",
                            xy=(pos, max(e["prob_before"], e["prob_after"])),
                            xytext=(-8, 20), textcoords="offset points", fontsize=6.5,
                            color=C_MISS, ha="right",
                            arrowprops=dict(arrowstyle="-", color=C_MISS, lw=0.5))

        st = np.abs(np.diff(ys))
        big = {xs[i] for i in range(len(st)) if st[i] >= a.tau}
        acc = {e["position"] for e in d["events"]}
        summary.append((lab, len(d["events"]), n_rep, sorted(big - acc), st.mean()))

        ax.set_ylabel("$P(\\mathrm{success})$", fontsize=9)
        ax.set_ylim(-0.06, 1.16)
        ax.set_xlim(plen - 6, max(xs) + 8)
        ax.tick_params(labelsize=8)
        # Upper left: both curves run low-to-mid there, and the bottom
        # right of each panel carries the final collapse.
        if len(paths) > 1:
            ax.text(0.008, 0.94, lab, transform=ax.transAxes, fontsize=7.5,
                    color=MUTED_TXT, ha="left", va="top")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("token position in the sequence", fontsize=9)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=C_DENSE, lw=1.0, label="every prefix re-estimated"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=C_DENSE, ms=5,
               label="estimates the search made"),
        Line2D([], [], color=C_HELP, lw=1.9, label="accepted pivot, replicated"),
    ]
    # Only name the failure case if the figure actually contains one; a
    # legend entry with no instances invites a hunt for something absent.
    if any(n_rep < n_ev for _, n_ev, n_rep, _, _ in summary):
        handles.append(Line2D([], [], color=C_MISS, lw=1.9, ls=":",
                              label="accepted, not replicated"))
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.55, -0.055))
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if a.png:
        fig.savefig(a.png, bbox_inches="tight", dpi=110)
    plt.close(fig)

    print(f"wrote {out}")
    for lab, n_ev, n_rep, missed, mstep in summary:
        print(f"  {lab:<22} {n_rep}/{n_ev} accepted events replicate, "
              f"missed={missed}, mean |step| {mstep:.3f}")


if __name__ == "__main__":
    main()
