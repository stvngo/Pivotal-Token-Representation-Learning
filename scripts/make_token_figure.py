#!/usr/bin/env python
"""Which tokens are pivotal, and are they consistently helpful or harmful?

Two claims in the paper rest on facts about this distribution and neither
shows it. Token identity scores 0.69--0.79 as a detector, which is only
sensible if pivots concentrate on few tokens -- and they do: one token is
28% of them. And the signed probe asks whether the *direction* of the
shift is predictable, which is only a live question if tokens are not
already sign-determined -- and the most common one is not.

The bars are split by sign rather than plain counts because the split is
the part that carries information. A plain frequency chart would show the
concentration and hide the ambiguity.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Diverging pair: helpful vs harmful is a polarity job, not a categorical one.
# Validated (CVD dE 21.9 protan / 30.9 tritan against the light surface).
C_HELP, C_HARM = "#0072B2", "#D55E00"
INK, MUTED = "#1a1a1a", "#6b6b6b"


def label_for(tok, tid: int) -> str:
    t = tok.decode([tid])
    return (t.replace("\n", "\\n").replace("\t", "\\t")
            .replace(" ", "\u2423") or "\u2205")      # open-box for space


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="qwen3-0.6b-full,qwen3-1.7b-full,qwen3-4b-full")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="paper/neurips2026/figures/pivotal_token_distribution.pdf")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cnt, helpful, total = collections.Counter(), collections.Counter(), 0
    for run in a.runs.split(","):
        path = ROOT / "runs" / run / "events.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            e = json.loads(line)
            cnt[e["token_id"]] += 1
            total += 1
            if e["prob_delta"] > 0:
                helpful[e["token_id"]] += 1

    top = cnt.most_common(a.top)
    ys = list(range(len(top)))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 6.4))

    for y, (tid, c) in zip(ys, top):
        h = helpful[tid]
        # 2px surface gap between stacked segments, per mark specs.
        ax.barh(y, h, color=C_HELP, height=0.72, zorder=2)
        ax.barh(y, c - h, left=h + max(total * 0.0008, 1.2), color=C_HARM,
                height=0.72, zorder=2)
        # The bar length carries frequency, which is so skewed that the sign
        # split is invisible below the top few. A direct label carries the
        # split independently of length.
        if c > top[0][1] * 0.55:
            ax.text(c - total * 0.006, y, f"{c}", va="center", ha="right",
                    fontsize=6.5, color="white", zorder=3)
        else:
            ax.text(c + total * 0.006, y, f"{c}", va="center", fontsize=6.5, color=MUTED)
        ax.text(top[0][1] * 1.155, y, f"{h / c:.0%}", va="center", fontsize=6.5,
                color=C_HELP if h / c >= 0.5 else C_HARM, ha="right")

    ax.set_yticks(ys, [label_for(tok, t) for t, _ in top], fontsize=7,
                  fontfamily="monospace")
    ax.set_xlabel("pivotal events", fontsize=9)
    ax.text(top[0][1] * 1.155, len(top) - 0.15, "% helpful", fontsize=6.5,
            color=MUTED, ha="right")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(0, top[0][1] * 1.16)
    ax.grid(axis="x", lw=0.4, alpha=0.28, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=C_HELP, label="helpful ($\\Delta p > 0$)"),
                        Patch(facecolor=C_HARM, label="harmful ($\\Delta p < 0$)")],
               fontsize=7.5, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.55, -0.035))
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if a.png:
        fig.savefig(a.png, bbox_inches="tight", dpi=110)
    plt.close(fig)

    cover = sum(c for _, c in top) / total
    print(f"wrote {out}")
    print(f"{total} events, {len(cnt)} distinct tokens; top {a.top} cover {cover:.1%}")
    print(f"most common: {tok.decode([top[0][0]])!r} at {top[0][1]/total:.1%} of all pivots")


if __name__ == "__main__":
    main()
