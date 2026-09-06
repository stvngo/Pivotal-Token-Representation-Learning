#!/usr/bin/env python
"""Figures for the paper, from artifacts already on disk.

No titles: a figure in a paper is captioned, and a title duplicated above
the caption is noise. Each file is named for what it shows instead.

Every number here is read from an artifact rather than typed in, so a
re-run of the pipeline moves the figures with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART, CMP, OUT = ROOT / "artifacts" / "steering", ROOT / "artifacts" / "compare", \
    ROOT / "paper" / "neurips2026" / "figures"
sys.path.insert(0, str(ROOT / "scripts"))

SCALES = [("Qwen3-0.6B", "0.6B", 0.6), ("Qwen3-1.7B", "1.7B", 1.7), ("Qwen3-4B", "4B", 4.0)]
UNSIGNED = {"0.6B": "ours_v3", "1.7B": "17b_tm1", "4B": "4b_tm1"}
SIGNED = {"0.6B": "signed_06b", "1.7B": "signed_17b", "4B": "signed_4b"}

C_PROBE, C_ENT, C_TOK, C_RAND = "#0072B2", "#D55E00", "#009E73", "#9aa0a6"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)


def fig_scaling(plt):
    """Probe vs the cheap detectors, as model size grows -- the headline."""
    probe, ent, tok, rand = [], [], [], []
    for _, short, _ in SCALES:
        d = json.loads((CMP / f"{UNSIGNED[short]}.json").read_text())
        probe.append(d["probe_auroc"])
        ent.append(d["baselines"]["entropy"]["auroc"])
        tok.append(d["baselines"]["token_identity_freq"]["auroc"])
        rand.append(d["baselines"]["random_direction"]["auroc"])
    x = np.arange(len(SCALES))
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for ys, c, m, lab in ((probe, C_PROBE, "o", "linear probe"),
                          (tok, C_TOK, "s", "token identity"),
                          (ent, C_ENT, "^", "next-token entropy"),
                          (rand, C_RAND, "v", "random direction")):
        ax.plot(x, ys, marker=m, color=c, lw=1.8, ms=5, label=lab)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, [s for _, s, _ in SCALES])
    ax.set_xlabel("model scale", fontsize=9)
    ax.set_ylabel("AUROC", fontsize=9)
    ax.set_ylim(0.35, 1.0)
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    _style(ax)
    return fig, "auroc_vs_scale_unsigned"


def fig_signed(plt):
    """The signed task, where the result only resolves at the largest scale."""
    probe, tok, ent = [], [], []
    for _, short, _ in SCALES:
        d = json.loads((CMP / f"{SIGNED[short]}.json").read_text())
        probe.append(d["probe_auroc"])
        tok.append(d["baselines"]["token_identity_freq"]["auroc"])
        ent.append(d["baselines"]["entropy"]["auroc"])
    x = np.arange(len(SCALES))
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for ys, c, m, lab in ((probe, C_PROBE, "o", "signed probe"),
                          (tok, C_TOK, "s", "token identity"),
                          (ent, C_ENT, "^", "next-token entropy")):
        ax.plot(x, ys, marker=m, color=c, lw=1.8, ms=5, label=lab)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, [s for _, s, _ in SCALES])
    ax.set_xlabel("model scale", fontsize=9)
    ax.set_ylabel("AUROC (helpful vs harmful pivot)", fontsize=9)
    ax.set_ylim(0.35, 0.9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(ax)
    return fig, "auroc_vs_scale_signed"


def fig_causal(plt):
    """Every equalised causal arm against the random-direction null band."""
    from steering_report import pool

    def load(stem, model):
        one = ART / f"{stem}_{model}.json"
        if one.exists():
            return json.loads(one.read_text())
        parts = [ART / f"{stem}_{model}_{h}.json" for h in ("train", "test")]
        have = [json.loads(p.read_text()) for p in parts if p.exists()]
        return pool(have) if len(have) > 1 else (have[0] if have else None)

    configs = [("eq_primary", "reactive", "cascade gate,\nsigned direction"),
               ("eq_simple", "reactive", "$P$(pivotal) gate,\nprobe weights"),
               ("eq_ablate", "ablate_reactive", "cascade gate,\nablation")]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    rand_all = []
    for _, short, _ in SCALES:
        for stem, _, _ in configs:
            d = load(stem, f"Qwen3-{short}")
            if not d:
                continue
            rand_all += [d[k]["delta_acc"] for k in
                         ("random_direction", "random_direction_1", "random_direction_2")
                         if k in d]
    lo, hi = np.percentile(rand_all, [5, 95])
    ax.axhspan(lo, hi, color=C_RAND, alpha=0.28, lw=0,
               label="random directions (5--95%)")
    ax.axhline(0.0, color="k", lw=0.8)

    markers, offsets = ["o", "s", "^"], [-0.22, 0.0, 0.22]
    for j, (stem, arm, lab) in enumerate(configs):
        xs, ys = [], []
        for i, (_, short, _) in enumerate(SCALES):
            d = load(stem, f"Qwen3-{short}")
            if not d or arm not in d:
                continue
            xs.append(i + offsets[j])
            ys.append(d[arm]["delta_acc"])
        ax.scatter(xs, ys, marker=markers[j], s=46, color=C_PROBE,
                   edgecolor="white", lw=0.6, zorder=3, label=lab.replace("\n", " "))
    ax.set_xticks(range(len(SCALES)), [s for _, s, _ in SCALES])
    ax.set_xlabel("model scale", fontsize=9)
    ax.set_ylabel(r"$\Delta$ GSM8K accuracy", fontsize=9)
    ax.set_ylim(-0.062, 0.038)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.55, -0.10))
    _style(ax)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig, "causal_effects_vs_null_band"


def fig_layers(plt):
    """Per-layer AUROC: the probe works across the stack, not at one depth."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for (full, short, _), c in zip(SCALES, (C_RAND, C_TOK, C_PROBE)):
        w = json.loads((ART / f"{full.lower()}.json").read_text())
        sw = sorted(w["sweep"], key=lambda r: r["layer"])
        if not sw:
            continue
        depth = np.array([r["layer"] for r in sw]) / max(r["layer"] for r in sw)
        ax.plot(depth, [r["auroc_unsigned"] for r in sw], color=c, lw=1.6, label=short)
    ax.set_xlabel("relative depth (layer / total layers)", fontsize=9)
    ax.set_ylabel("AUROC (inner split)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, title="scale", title_fontsize=8)
    _style(ax)
    return fig, "auroc_by_relative_depth"


def fig_band(plt):
    """Why the causal evaluation is restricted: GSM8K saturates with scale."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    width = 0.26
    for j, (full, short, _) in enumerate(SCALES):
        ph = []
        for f in sorted(ART.glob(f"band_{full}*.json")):
            ph += json.loads(f.read_text())["p_hat"]
        ph = np.asarray(ph)
        bins = np.arange(0, 1.0001, 0.125)
        frac = np.histogram(ph, bins=bins)[0] / len(ph)
        ax.bar(np.arange(len(frac)) + (j - 1) * width, frac, width=width,
               label=f"{short} ($n{{=}}{len(ph)}$)",
               color=[C_RAND, C_TOK, C_PROBE][j], edgecolor="white", lw=0.4)
    ax.axvspan(1.5, 6.5, color=C_ENT, alpha=0.12, lw=0)
    ax.text(4.0, ax.get_ylim()[1] * 0.92, "searchable band", fontsize=7.5,
            ha="center", color=C_ENT)
    ax.set_xticks(range(8), [f"{b:.2f}" for b in np.arange(0.0625, 1.0, 0.125)])
    ax.set_xlabel(r"estimated $P(\mathrm{success})$, 8 rollouts", fontsize=9)
    ax.set_ylabel("fraction of screened questions", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)
    return fig, "band_saturation_by_scale"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_scaling, fig_signed, fig_causal, fig_layers, fig_band):
        try:
            fig, name = fn(plt)
        except Exception as exc:
            print(f"  SKIP {fn.__name__}: {type(exc).__name__}: {exc}")
            continue
        fig.tight_layout()
        path = OUT / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(f"/tmp/claude-501/-Users-svngo-Documents-GitHub-Pivotal-Token-Representation-Learning/74d1f515-bc85-4373-ba57-d5688e578faf/scratchpad/{name}.png",
                    bbox_inches="tight", dpi=110)
        plt.close(fig)
        print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
