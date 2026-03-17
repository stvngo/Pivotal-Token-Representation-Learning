"""Visualization helpers for probe metrics and embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _load_accuracy_json(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(k): float(v) for k, v in raw.items()}


def plot_accuracy_across_layers(config: dict[str, Any], logger: Any) -> Path:
    """Plot sklearn + pytorch validation accuracy by layer."""
    output_dir = Path(config["paths"]["outputs"]["plots_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pytorch_acc_path = Path(config["paths"]["outputs"]["pytorch_dir"]) / "probe_states" / "layer_wise_accuracies.json"
    sklearn_acc_path = Path(config["paths"]["outputs"]["sklearn_dir"]) / "probe_states" / "layer_wise_accuracies.json"

    pytorch_acc = _load_accuracy_json(pytorch_acc_path)
    sklearn_acc = _load_accuracy_json(sklearn_acc_path)
    if not pytorch_acc and not sklearn_acc:
        raise ValueError("No accuracy JSON files found for plotting.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    if pytorch_acc:
        layers = sorted(pytorch_acc)
        values = [pytorch_acc[l] for l in layers]
        sns.lineplot(x=layers, y=values, marker="o", ax=ax, label="PyTorch")
    if sklearn_acc:
        layers = sorted(sklearn_acc)
        values = [sklearn_acc[l] for l in layers]
        sns.lineplot(x=layers, y=values, marker="x", ax=ax, label="Sklearn")

    ax.set_title("Probe Validation Accuracy vs. Model Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.grid(True)
    ax.legend()

    output_path = output_dir / "accuracy_by_layer.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved accuracy plot to %s", output_path)
    return output_path


def _choose_layers_for_embeddings(
    analysis_root: Path,
    accuracy_json_path: Path,
    layers_override: list[int] | None,
    num_layers: int | None,
) -> list[int]:
    if layers_override:
        return layers_override
    accuracy_map = _load_accuracy_json(accuracy_json_path)
    if accuracy_map:
        ordered = sorted(accuracy_map.items(), key=lambda x: x[1], reverse=True)
        chosen = [layer for layer, _ in ordered]
    else:
        chosen = []
        for p in analysis_root.glob("layer_*"):
            if p.is_dir():
                try:
                    chosen.append(int(p.name.split("_", maxsplit=1)[1]))
                except ValueError:
                    continue
        chosen.sort()
    if num_layers is None:
        return chosen
    return chosen[: max(0, num_layers)]


def plot_pca_tsne(
    config: dict[str, Any],
    logger: Any,
    backend: str = "pytorch",
    layers_override: list[int] | None = None,
    num_layers_override: int | None = None,
) -> list[Path]:
    """Generate PCA and t-SNE plots for selected layers."""
    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    analysis_root = backend_root / "analysis_data"
    accuracy_json_path = backend_root / "probe_states" / "layer_wise_accuracies.json"

    viz_cfg = config.get("visualization", {})
    default_layers = viz_cfg.get("layers") or None
    selected = _choose_layers_for_embeddings(
        analysis_root=analysis_root,
        accuracy_json_path=accuracy_json_path,
        layers_override=layers_override if layers_override is not None else default_layers,
        num_layers=num_layers_override if num_layers_override is not None else viz_cfg.get("num_layers", 2),
    )
    if not selected:
        raise ValueError("No layers selected for PCA/t-SNE visualization.")

    output_dir = Path(config["paths"]["outputs"]["plots_dir"]) / backend
    output_dir.mkdir(parents=True, exist_ok=True)
    perplexity_cap = int(viz_cfg.get("tsne_perplexity", 30))
    saved_paths: list[Path] = []

    for layer in selected:
        layer_dir = analysis_root / f"layer_{layer}"
        x_path = layer_dir / f"activations_layer_{layer}.npy"
        y_path = layer_dir / "labels.npy"
        if not x_path.exists() or not y_path.exists():
            logger.warning("Skipping visualization for layer %s, missing files.", layer)
            continue

        x = np.load(x_path)
        y = np.load(y_path).reshape(-1)
        if x.shape[0] < 2:
            logger.warning("Skipping layer %s for PCA/t-SNE (not enough samples).", layer)
            continue

        pca = PCA(n_components=2, random_state=42)
        x_pca = pca.fit_transform(x)

        perplexity = min(perplexity_cap, x.shape[0] - 1)
        if perplexity <= 1:
            logger.warning("Skipping t-SNE for layer %s due to low sample count.", layer)
            continue
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=300, random_state=42)
        x_tsne = tsne.fit_transform(x)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=y, palette="viridis", s=35, alpha=0.8, ax=axes[0])
        axes[0].set_title(f"PCA Activations | Layer {layer} | {backend}")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")

        sns.scatterplot(x=x_tsne[:, 0], y=x_tsne[:, 1], hue=y, palette="viridis", s=35, alpha=0.8, ax=axes[1])
        axes[1].set_title(f"t-SNE Activations | Layer {layer} | {backend}")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")

        fig.tight_layout()
        save_path = output_dir / f"layer_{layer}_pca_tsne.png"
        fig.savefig(save_path)
        plt.close(fig)
        saved_paths.append(save_path)
        logger.info("Saved PCA/t-SNE plot to %s", save_path)

    return saved_paths


def plot_activation_heatmap(
    config: dict[str, Any],
    logger: Any,
    backend: str = "pytorch",
    layers_override: list[int] | None = None,
    max_dims: int | None = 256,
    max_samples: int | None = None,
) -> list[Path]:
    """Plot activation heatmaps for selected layers (samples x features)."""
    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    analysis_root = backend_root / "analysis_data"

    selected_layers = layers_override or []
    if not selected_layers:
        for p in sorted(analysis_root.iterdir()):
            if p.is_dir() and p.name.startswith("layer_"):
                try:
                    selected_layers.append(int(p.name.split("_", maxsplit=1)[1]))
                except ValueError:
                    continue
        selected_layers = sorted(selected_layers)

    if not selected_layers:
        raise ValueError(f"No layer data found under {analysis_root}")

    output_dir = Path(config["paths"]["outputs"]["plots_dir"]) / backend
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for layer in selected_layers:
        layer_dir = analysis_root / f"layer_{layer}"
        x_path = layer_dir / f"activations_layer_{layer}.npy"
        y_path = layer_dir / "labels.npy"
        if not x_path.exists() or not y_path.exists():
            logger.warning("Skipping heatmap for layer %s, missing files.", layer)
            continue

        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).reshape(-1)

        # Sort by label (pivotal=1 first, non-pivotal=0 second)
        order = np.argsort(-y)
        x_sorted = x[order]

        if max_samples is not None and x_sorted.shape[0] > max_samples:
            step = max(1, x_sorted.shape[0] // max_samples)
            idx = np.arange(0, x_sorted.shape[0], step)[:max_samples]
            x_sorted = x_sorted[idx]

        n_dims = x_sorted.shape[1]
        if max_dims is not None and n_dims > max_dims:
            step = max(1, n_dims // max_dims)
            dim_idx = np.arange(0, n_dims, step)[:max_dims]
            x_sorted = x_sorted[:, dim_idx]

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(x_sorted, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax.set_title(
            f"Activation Heatmap | Layer {layer} | {backend} | "
            f"{x_sorted.shape[0]} samples × {x_sorted.shape[1]} dims"
        )
        ax.set_xlabel("Hidden dimension")
        ax.set_ylabel("Sample (sorted: pivotal → non-pivotal)")
        plt.colorbar(im, ax=ax, label="Activation")
        fig.tight_layout()

        save_path = output_dir / f"layer_{layer}_heatmap.png"
        fig.savefig(save_path)
        plt.close(fig)
        saved_paths.append(save_path)
        logger.info("Saved activation heatmap to %s", save_path)

    return saved_paths


def plot_probe_weights(
    config: dict[str, Any],
    logger: Any,
    backend: str = "pytorch",
    layers_override: list[int] | None = None,
    top_k: int | None = 64,
) -> list[Path]:
    """Plot probe weight magnitudes (which dimensions drive the probe)."""
    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    analysis_root = backend_root / "analysis_data"

    selected_layers = layers_override or []
    if not selected_layers:
        for p in sorted(analysis_root.iterdir()):
            if p.is_dir() and p.name.startswith("layer_"):
                try:
                    selected_layers.append(int(p.name.split("_", maxsplit=1)[1]))
                except ValueError:
                    continue
        selected_layers = sorted(selected_layers)

    if not selected_layers:
        raise ValueError(f"No layer data found under {analysis_root}")

    output_dir = Path(config["paths"]["outputs"]["plots_dir"]) / backend
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for layer in selected_layers:
        layer_dir = analysis_root / f"layer_{layer}"
        w_path = layer_dir / "probe_weights.npy"
        if not w_path.exists():
            logger.warning("Skipping probe weights for layer %s, missing file.", layer)
            continue

        w = np.load(w_path).astype(np.float32).flatten()
        order = np.argsort(np.abs(w))[::-1]
        w_sorted = w[order]
        dims = np.arange(len(w_sorted))

        k = top_k if top_k is not None else len(w)
        k = min(k, len(w_sorted))
        w_plot = w_sorted[:k]
        dims_plot = dims[:k]

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = np.where(w_plot >= 0, "steelblue", "coral")
        ax.bar(dims_plot, w_plot, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Probe Weights (top {k} by magnitude) | Layer {layer} | {backend}")
        ax.set_xlabel("Dimension (sorted by |weight|)")
        ax.set_ylabel("Weight")
        fig.tight_layout()

        save_path = output_dir / f"layer_{layer}_probe_weights.png"
        fig.savefig(save_path)
        plt.close(fig)
        saved_paths.append(save_path)
        logger.info("Saved probe weights plot to %s", save_path)

    return saved_paths


def plot_sample_correlation(
    config: dict[str, Any],
    logger: Any,
    backend: str = "pytorch",
    layers_override: list[int] | None = None,
) -> tuple[list[Path], dict[int, dict[str, float]]]:
    """Plot sample × sample correlation heatmap (cosine similarity), sorted by label.
    Returns (saved_paths, separability_metrics per layer).
    """
    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    analysis_root = backend_root / "analysis_data"

    selected_layers = layers_override or []
    if not selected_layers:
        for p in sorted(analysis_root.iterdir()):
            if p.is_dir() and p.name.startswith("layer_"):
                try:
                    selected_layers.append(int(p.name.split("_", maxsplit=1)[1]))
                except ValueError:
                    continue
        selected_layers = sorted(selected_layers)

    if not selected_layers:
        raise ValueError(f"No layer data found under {analysis_root}")

    output_dir = Path(config["paths"]["outputs"]["plots_dir"]) / backend
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    separability_metrics: dict[int, dict[str, float]] = {}

    for layer in selected_layers:
        layer_dir = analysis_root / f"layer_{layer}"
        x_path = layer_dir / f"activations_layer_{layer}.npy"
        y_path = layer_dir / "labels.npy"
        if not x_path.exists() or not y_path.exists():
            logger.warning("Skipping sample correlation for layer %s, missing files.", layer)
            continue

        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).reshape(-1)

        # Cosine similarity: normalize rows, then X @ X.T
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x_norm = x / norms
        sim = x_norm @ x_norm.T

        # Sort by label (pivotal=1 first)
        order = np.argsort(-y)
        sim_sorted = sim[order][:, order]
        y_sorted = y[order]

        n_pos = int((y_sorted >= 0.5).sum())
        n_neg = len(y_sorted) - n_pos

        # Quadrant indices (after sorting: pivotal 0..n_pos-1, non-pivotal n_pos..n-1)
        # Intra pivotal: sim[0:n_pos, 0:n_pos], exclude diagonal
        # Intra non-pivotal: sim[n_pos:, n_pos:], exclude diagonal
        # Inter: sim[0:n_pos, n_pos:] and sim[n_pos:, 0:n_pos]

        mask_pos = np.arange(len(y_sorted)) < n_pos
        mask_neg = ~mask_pos

        intra_pos_vals = sim_sorted[:n_pos, :n_pos][np.triu_indices(n_pos, k=1)]
        intra_neg_vals = sim_sorted[n_pos:, n_pos:][np.triu_indices(n_neg, k=1)]
        inter_vals = sim_sorted[:n_pos, n_pos:].flatten()

        mean_intra_pos = float(np.mean(intra_pos_vals)) if len(intra_pos_vals) > 0 else 0.0
        mean_intra_neg = float(np.mean(intra_neg_vals)) if len(intra_neg_vals) > 0 else 0.0
        mean_inter = float(np.mean(inter_vals)) if len(inter_vals) > 0 else 0.0
        mean_intra = float(np.mean(np.concatenate([intra_pos_vals, intra_neg_vals]))) if (len(intra_pos_vals) + len(intra_neg_vals)) > 0 else 0.0
        separability_gap = mean_intra - mean_inter

        separability_metrics[layer] = {
            "mean_intra_pivotal": mean_intra_pos,
            "mean_intra_nonpivotal": mean_intra_neg,
            "mean_intra_overall": mean_intra,
            "mean_inter": mean_inter,
            "separability_gap": separability_gap,
        }

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim_sorted, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
        ax.axhline(n_pos - 0.5, color="white", linewidth=1)
        ax.axvline(n_pos - 0.5, color="white", linewidth=1)
        ax.set_title(
            f"Sample Correlation (cosine) | Layer {layer} | {backend}\n"
            f"Pivotal (top-left) vs Non-pivotal (bottom-right)"
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel("Sample")

        textstr = (
            f"Mean intra (pivotal): {mean_intra_pos:.3f}\n"
            f"Mean intra (non-pivotal): {mean_intra_neg:.3f}\n"
            f"Mean inter: {mean_inter:.3f}\n"
            f"Separability gap: {separability_gap:.3f}"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.9)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)

        plt.colorbar(im, ax=ax, label="Cosine similarity")
        fig.tight_layout()

        save_path = output_dir / f"layer_{layer}_sample_correlation.png"
        fig.savefig(save_path)
        plt.close(fig)
        saved_paths.append(save_path)
        logger.info(
            "Saved sample correlation plot to %s | intra=%.3f inter=%.3f gap=%.3f",
            save_path,
            mean_intra,
            mean_inter,
            separability_gap,
        )

    return saved_paths, separability_metrics


def plot_centroid_difference(
    config: dict[str, Any],
    logger: Any,
    backend: str = "pytorch",
    layers_override: list[int] | None = None,
    top_k: int | None = 64,
) -> list[Path]:
    """Plot centroid difference (mean_pivotal - mean_nonpivotal) per dimension."""
    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    analysis_root = backend_root / "analysis_data"

    selected_layers = layers_override or []
    if not selected_layers:
        for p in sorted(analysis_root.iterdir()):
            if p.is_dir() and p.name.startswith("layer_"):
                try:
                    selected_layers.append(int(p.name.split("_", maxsplit=1)[1]))
                except ValueError:
                    continue
        selected_layers = sorted(selected_layers)

    if not selected_layers:
        raise ValueError(f"No layer data found under {analysis_root}")

    output_dir = Path(config["paths"]["outputs"]["plots_dir"]) / backend
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for layer in selected_layers:
        layer_dir = analysis_root / f"layer_{layer}"
        x_path = layer_dir / f"activations_layer_{layer}.npy"
        y_path = layer_dir / "labels.npy"
        if not x_path.exists() or not y_path.exists():
            logger.warning("Skipping centroid difference for layer %s, missing files.", layer)
            continue

        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).reshape(-1)

        mask_pos = y >= 0.5
        mask_neg = y < 0.5
        mu_pos = x[mask_pos].mean(axis=0)
        mu_neg = x[mask_neg].mean(axis=0)
        diff = mu_pos - mu_neg

        order = np.argsort(np.abs(diff))[::-1]
        diff_sorted = diff[order]
        dims = np.arange(len(diff_sorted))

        k = top_k if top_k is not None else len(diff)
        k = min(k, len(diff_sorted))
        diff_plot = diff_sorted[:k]
        dims_plot = dims[:k]

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = np.where(diff_plot >= 0, "steelblue", "coral")
        ax.bar(dims_plot, diff_plot, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Centroid Difference (pivotal − non-pivotal) | Layer {layer} | {backend} | top {k} dims")
        ax.set_xlabel("Dimension (sorted by |diff|)")
        ax.set_ylabel("Difference")
        fig.tight_layout()

        save_path = output_dir / f"layer_{layer}_centroid_difference.png"
        fig.savefig(save_path)
        plt.close(fig)
        saved_paths.append(save_path)
        logger.info("Saved centroid difference plot to %s", save_path)

    return saved_paths


def plot_probe_analysis(
    config: dict[str, Any],
    logger: Any,
    backend: str = "pytorch",
    layers_override: list[int] | None = None,
    top_k: int | None = 64,
) -> list[Path]:
    """Plot probe weights, sample correlation, and centroid difference for selected layers."""
    paths: list[Path] = []
    paths.extend(
        plot_probe_weights(config, logger, backend=backend, layers_override=layers_override, top_k=top_k)
    )
    corr_paths, separability_metrics = plot_sample_correlation(
        config, logger, backend=backend, layers_override=layers_override
    )
    paths.extend(corr_paths)

    # Save separability metrics to JSON
    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    metrics_dir = backend_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if separability_metrics:
        with (metrics_dir / "separability_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in separability_metrics.items()}, f, indent=2)
        logger.info("Saved separability metrics to %s", metrics_dir / "separability_metrics.json")

    paths.extend(
        plot_centroid_difference(config, logger, backend=backend, layers_override=layers_override, top_k=top_k)
    )
    return paths


def plot_confusion_matrices(
    config: dict[str, Any],
    logger: Any,
    backend: str = "pytorch",
    layers_override: list[int] | None = None,
) -> list[Path]:
    """Plot confusion matrix heatmaps from evaluation metrics."""
    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    metrics_path = backend_root / "metrics" / "evaluation_metrics.json"
    if not metrics_path.exists():
        raise ValueError(f"Evaluation metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    layers_data = data.get("layers", {})

    selected_layers = sorted(int(x) for x in layers_data.keys())
    if layers_override:
        selected_layers = [layer for layer in layers_override if str(layer) in layers_data]

    output_dir = Path(config["paths"]["outputs"]["plots_dir"]) / backend
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for layer in selected_layers:
        metrics = layers_data[str(layer)]
        cm = np.array(metrics["confusion_matrix"], dtype=np.int64)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"Confusion Matrix | Layer {layer} | {backend}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(["Non-pivotal", "Pivotal"])
        ax.set_yticklabels(["Non-pivotal", "Pivotal"], rotation=0)
        fig.tight_layout()

        save_path = output_dir / f"layer_{layer}_confusion_matrix.png"
        fig.savefig(save_path)
        plt.close(fig)
        saved_paths.append(save_path)
        logger.info("Saved confusion matrix plot to %s", save_path)

    return saved_paths

