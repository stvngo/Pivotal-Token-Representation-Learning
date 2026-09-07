#!/usr/bin/env python
"""One searched rollout, drawn: where PTS measured, and what moved.

The paper defines pivotality by an equation and reports AUROCs against it,
but never shows what a pivotal token looks like. This draws a single
branch: the estimated success probability at each position the bisection
actually evaluated, with the accepted pivots marked.

The honest detail the figure has to carry is that PTS does **not** estimate
at every token. It bisects, so it measures a handful of midpoints per
rollout -- 14 on average here -- and the accepted events are the subset of
those where the probability moved by more than tau. Drawing a smooth curve
over every position would imply a measurement that was never made.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

C_HELP, C_HARM, C_MEAS = "#0072B2", "#D55E00", "#4a4a4a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="runs/qwen3-4b-full/events.jsonl")
    ap.add_argument("--query", default="1223")
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--outdir", default="paper/neurips2026/figures")
    ap.add_argument("--tex", default=None,
                    help="default: paper/neurips2026/example_rollout_<query>.tex")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ev = [json.loads(l) for l in open(ROOT / a.events)]
    by = collections.defaultdict(list)
    for e in ev:
        by[(e["query_uid"], e["generation_index"])].append(e)
    es = sorted(by[(a.query, a.generation)], key=lambda e: e["position"])
    if not es:
        raise SystemExit(f"no events for query {a.query}")
    seq, plen = es[0]["sequence_token_ids"], es[0]["prompt_len"]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.axhspan(0.2, 0.8, color="#9aa0a6", alpha=0.16, lw=0)
    ax.text(plen + 4, 0.83, "searchable band", fontsize=7, color="#666")

    # Every estimate the search actually made on this branch, in position order.
    pts = [(plen, es[0]["baseline_prob"])]
    for e in es:
        pts += [(e["position"], e["prob_before"]), (e["position"] + 1, e["prob_after"])]
    pts.sort()
    ax.plot([p for p, _ in pts], [v for _, v in pts], color=C_MEAS, lw=1.0,
            alpha=0.45, zorder=1)
    ax.scatter([p for p, _ in pts], [v for _, v in pts], s=13, color=C_MEAS,
               zorder=2, label="estimated $P(\\mathrm{success})$")

    for e in es:
        c = C_HELP if e["prob_delta"] > 0 else C_HARM
        ax.annotate("", xy=(e["position"] + 1, e["prob_after"]),
                    xytext=(e["position"], e["prob_before"]),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8,
                                    shrinkA=0, shrinkB=0), zorder=3)
        label = tok.decode([e["token_id"]]).replace("\n", "\\n").strip() or "\\n"
        # Pivots often cluster within a few tokens, so fixed offsets overlap.
        # Step the label up for each neighbour closer than 4% of the sequence.
        near = sum(1 for o in es if 0 < e["position"] - o["position"] <=
                   max(6, 0.04 * (len(seq) - plen)))
        ax.annotate(f"{label!r}", xy=(e["position"], max(e["prob_before"], e["prob_after"])),
                    xytext=(0, 7 + 11 * near), textcoords="offset points", fontsize=6.5,
                    color=c, ha="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.4, alpha=0.5)
                    if near else None)

    ax.set_xlabel("token position in the sequence", fontsize=9)
    ax.set_ylabel("$P(\\mathrm{success})$", fontsize=9)
    ax.set_ylim(-0.05, 1.30)
    ax.set_xlim(plen - 8, len(seq) + 6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    from matplotlib.lines import Line2D
    # Below the axes: the arrows and the band already occupy the corners.
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="-", color=C_MEAS, ms=4, lw=1,
               label="estimates made by the search"),
        Line2D([], [], color=C_HELP, lw=1.8, label="accepted pivot, helpful"),
        Line2D([], [], color=C_HARM, lw=1.8, label="accepted pivot, harmful"),
    ], fontsize=7, frameon=False, loc="lower center", ncol=3,
       bbox_to_anchor=(0.55, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out = ROOT / a.outdir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"example_rollout_q{a.query}.pdf", bbox_inches="tight")
    if a.png:
        fig.savefig(a.png, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"wrote {out / f'example_rollout_q{a.query}.pdf'}")

    # LaTeX listing of the rollout with the pivots marked in place.
    piv = {e["position"]: e for e in es}
    def esc(t: str) -> str:
        for a_, b_ in [("\\", "\\textbackslash{}"), ("{", "\\{"), ("}", "\\}"),
                       ("$", "\\$"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"),
                       ("_", "\\_"), ("^", "\\^{}"), ("~", "\\~{}")]:
            t = t.replace(a_, b_)
        return t

    body = []
    for i in range(plen, min(len(seq), plen + 290)):
        t = tok.decode([seq[i]])
        if i in piv:
            c = "probehelp" if piv[i]["prob_delta"] > 0 else "probeharm"
            body.append("\\textbf{\\color{%s}[%s]}" % (c, esc(t.replace("\n", "\\n"))))
        else:
            body.append(esc(t).replace("\n", "\\\\\n"))
    listing = "".join(body)

    rows = "\n".join(
        "%d & \\texttt{%s} & %.2f & %.2f & %+.2f & %d/%d $\\to$ %d/%d \\\\" % (
            e["position"], esc(tok.decode([e["token_id"]]).replace("\n", "\\n")),
            e["prob_before"], e["prob_after"], e["prob_delta"],
            e["s_before"], e["n_before"], e["s_after"], e["n_after"])
        for e in es)

    import re as _re
    question = tok.decode(seq[:plen])
    question = _re.sub(r"<\|im_start\|>\w*\n?", "", question)
    question = _re.sub(r"<\|im_end\|>", "", question)
    question = _re.sub(r"<think>\s*</think>", "", question)
    question = _re.sub(r"\n\nPut your final answer in .*$", "", question, flags=_re.S)
    question = question.strip()

    tex = f"""% generated by scripts/make_example_figure.py -- do not edit by hand
\\definecolor{{probehelp}}{{HTML}}{{0072B2}}
\\definecolor{{probeharm}}{{HTML}}{{D55E00}}

\\textbf{{Question.}} \\emph{{{esc(question)}}}

\\smallskip
\\noindent\\textbf{{Rollout}} (first {min(290, len(seq) - plen)} generated tokens; accepted
pivots in brackets, {{\\color{{probehelp}}blue}} helpful and
{{\\color{{probeharm}}orange}} harmful):

\\smallskip
{{\\footnotesize\\setlength{{\\parindent}}{{0pt}}{listing}}}

\\smallskip
\\begin{{center}}
\\begin{{tabular}}{{rlcccr}}
\\toprule
position & token & $p_{{\\text{{before}}}}$ & $p_{{\\text{{after}}}}$ & $\\pdelta$ & rollouts \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
"""
    texpath = ROOT / (a.tex or f"paper/neurips2026/example_rollout_q{a.query}.tex")
    texpath.write_text(tex)
    print(f"wrote {texpath}")
    print(f"{len(es)} pivots, prompt {plen} tok, sequence {len(seq)} tok")


if __name__ == "__main__":
    main()
