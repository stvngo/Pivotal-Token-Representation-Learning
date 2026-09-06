#!/usr/bin/env python
"""2D projections of the three activation classes, per model.

Three classes, not two: a non-pivotal position, a position preceding a
*helpful* pivot, and one preceding a *harmful* pivot. Splitting the
pivotal class by the sign of ``prob_delta`` is the point -- the unsigned
label is what PTS supplies, and whether the sign is visible at all is the
question the signed probe answers numerically. These figures show what
that looks like geometrically.

Projections are fit on TEST rows only. Fitting PCA on train and projecting
test would be defensible, but the figure is descriptive rather than
inferential and mixing the two invites reading it as evidence about
generalisation, which it is not.

The layer is the steering layer, chosen (on train) to maximise
``min(AUROC_unsigned, AUROC_signed)`` -- the one place both distinctions
are jointly present, and so the only honest choice for a picture meant to
show both.

    python scripts/plot_embeddings.py --acts data/acts_4b_tm1 --tag qwen3-4b \
        --probes artifacts/steering/qwen3-4b.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_pipeline.activations_v2 import ActivationStoreV2  # noqa: E402

# Colour-blind-safe, and ordered so the two pivotal classes are the two
# saturated colours and the non-pivotal background is the muted one.
COLOURS = {
    "non-pivotal": "#9aa0a6",
    "helpful pivot": "#0072B2",
    "harmful pivot": "#D55E00",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--outdir", default="paper/neurips2026/figures")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", default=None, help="also write a PNG, for eyeballing")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    layer = int(json.loads(Path(a.probes).with_suffix(".json").read_text())["layer"])
    te = ActivationStoreV2.open(Path(a.acts) / f"{a.tag}_test.safetensors")
    x = te.layer(layer).astype(np.float64)
    y, delta = te.labels(), te.prob_delta()

    cls = np.where(y <= 0, 0, np.where(delta > 0, 1, 2))
    names = ["non-pivotal", "helpful pivot", "harmful pivot"]
    counts = {n: int((cls == i).sum()) for i, n in enumerate(names)}

    # Standardise before projecting. Qwen residual streams have a handful of
    # "massive activation" dimensions -- max |x| is in the hundreds -- and
    # without scaling both PCA and t-SNE render those few dimensions and
    # nothing else.
    xs = StandardScaler().fit_transform(x)

    pcs = PCA(n_components=2, random_state=a.seed).fit(xs)
    emb_pca = pcs.transform(xs)
    perp = min(a.perplexity, max(5.0, (len(xs) - 1) / 3.0))
    emb_tsne = TSNE(n_components=2, random_state=a.seed, perplexity=perp,
                    init="pca", max_iter=1000).fit_transform(xs)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, emb, labs in (
        (axes[0], emb_pca, (f"PC1 ({pcs.explained_variance_ratio_[0]*100:.1f}%)",
                            f"PC2 ({pcs.explained_variance_ratio_[1]*100:.1f}%)")),
        (axes[1], emb_tsne, ("t-SNE 1", "t-SNE 2")),
    ):
        # Non-pivotal first so the pivotal classes are not buried under it.
        for i in (0, 1, 2):
            m = cls == i
            ax.scatter(emb[m, 0], emb[m, 1], s=9, alpha=0.65,
                       c=COLOURS[names[i]], label=names[i],
                       linewidths=0, rasterized=True)
        ax.set_xlabel(labs[0], fontsize=9)
        ax.set_ylabel(labs[1], fontsize=9)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02), markerscale=1.8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"embed_{a.tag}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    if a.png:
        fig.savefig(Path(a.png), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"{a.tag}: layer {layer}, {len(y)} test rows {counts} -> {path}")


if __name__ == "__main__":
    main()
