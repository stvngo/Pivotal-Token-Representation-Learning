"""Evaluate saved probes from persisted activation arrays."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import pick_device
from .metrics import compute_binary_metrics
from .train_pytorch import LinearProbe


def _discover_layer_dirs(analysis_root: Path) -> list[int]:
    layers: list[int] = []
    if not analysis_root.exists():
        return layers
    for path in analysis_root.iterdir():
        if not path.is_dir() or not path.name.startswith("layer_"):
            continue
        try:
            layers.append(int(path.name.split("_", maxsplit=1)[1]))
        except ValueError:
            continue
    return sorted(layers)


def evaluate_saved_probes(
    config: dict[str, Any],
    logger: Any,
    backend: str = "pytorch",
    threshold_override: float | None = None,
    layers_override: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate stored probe models against stored validation arrays."""
    backend = backend.lower().strip()
    if backend not in {"pytorch", "sklearn"}:
        raise ValueError(f"Unsupported backend: {backend}")

    outputs_cfg = config["paths"]["outputs"]
    backend_root = Path(outputs_cfg["pytorch_dir"] if backend == "pytorch" else outputs_cfg["sklearn_dir"])
    states_dir = backend_root / "probe_states"
    analysis_root = backend_root / "analysis_data"
    metrics_dir = backend_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(config.get("training", {}).get("threshold", 0.5))
    )

    candidate_layers = _discover_layer_dirs(analysis_root)
    if layers_override:
        candidate_layers = [layer for layer in layers_override if layer in candidate_layers]

    if not candidate_layers:
        raise ValueError(f"No layer data found under {analysis_root}")

    device = pick_device(config.get("device", "auto"))
    all_metrics: dict[int, dict[str, Any]] = {}

    for layer in candidate_layers:
        logger.info("--- Evaluating %s probe layer %s ---", backend, layer)
        layer_dir = analysis_root / f"layer_{layer}"
        x_path = layer_dir / f"activations_layer_{layer}.npy"
        y_path = layer_dir / "labels.npy"
        if not x_path.exists() or not y_path.exists():
            logger.warning("Skipping layer %s: missing validation arrays.", layer)
            continue

        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64).reshape(-1)

        if backend == "pytorch":
            state_path = states_dir / f"probe_layer_{layer}.pth"
            if not state_path.exists():
                logger.warning("Skipping layer %s: missing state dict.", layer)
                continue
            probe = LinearProbe(input_dim=x.shape[1]).to(device)
            probe.load_state_dict(torch.load(state_path, map_location=device))
            probe.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                y_prob = torch.sigmoid(probe(x_tensor)).cpu().numpy().reshape(-1)
        else:
            model_path = states_dir / f"probe_layer_{layer}.pkl"
            if not model_path.exists():
                logger.warning("Skipping layer %s: missing sklearn model.", layer)
                continue
            with model_path.open("rb") as handle:
                model = pickle.load(handle)
            y_prob = model.predict_proba(x)[:, 1]

        metrics = compute_binary_metrics(y_true=y, y_prob=y_prob, threshold=threshold)
        metrics["samples"] = int(len(y))
        metrics["backend"] = backend
        all_metrics[layer] = metrics
        logger.info(
            "Layer %s | Acc %.2f%% | F1 %.4f | AUROC %s | FP/FN %.4f",
            layer,
            metrics["accuracy"] * 100.0,
            metrics["f1_score"],
            f"{metrics['auroc_score']:.4f}" if metrics["auroc_score"] is not None else "N/A",
            metrics["fp_fn_ratio"],
        )

    results = {
        "backend": backend,
        "threshold": threshold,
        "layers": {str(k): v for k, v in all_metrics.items()},
    }
    out_path = metrics_dir / "evaluation_metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Saved evaluation metrics to %s", out_path)
    return results

