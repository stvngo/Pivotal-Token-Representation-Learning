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

