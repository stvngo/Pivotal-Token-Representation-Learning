"""Layer-wise scikit-learn probe training."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .activations import layer_arrays, list_available_layers, load_activation_store
from .common import parse_layer_selection
from .metrics import compute_binary_metrics


def _build_estimator(cfg: dict[str, Any]) -> Pipeline | LogisticRegression:
    model = LogisticRegression(
        solver=cfg.get("solver", "saga"),
        max_iter=int(cfg.get("max_iter", 3000)),
        C=float(cfg.get("C", 1.0)),
        random_state=int(cfg.get("random_state", 42)),
        class_weight=cfg.get("class_weight"),
    )
    if cfg.get("standardize", True):
        return Pipeline([("scaler", StandardScaler()), ("logreg", model)])
    return model


def run_sklearn_training(
    config: dict[str, Any],
    logger: Any,
    layers_override: list[int] | None = None,
    num_layers_override: int | None = None,
) -> dict[str, Any]:
    """Train sklearn logistic probes per layer."""
    paths = config["paths"]
    train_acts = load_activation_store(paths["activations"]["train"])
    test_acts = load_activation_store(paths["activations"]["test"])

    available_layers = list_available_layers(train_acts, test_acts)
    training_cfg = config["training"]
    selected_layers = parse_layer_selection(
        available_layers,
        explicit_layers=layers_override if layers_override is not None else training_cfg.get("layers"),
        num_layers=num_layers_override if num_layers_override is not None else training_cfg.get("num_layers"),
    )
    if not selected_layers:
        raise ValueError("No layers selected for sklearn training.")

    threshold = float(training_cfg.get("threshold", 0.5))
    sklearn_cfg = config.get("sklearn", {})

    output_root = Path(paths["outputs"]["sklearn_dir"])
    states_dir = output_root / "probe_states"
    analysis_dir = output_root / "analysis_data"
    metrics_dir = output_root / "metrics"
    for directory in [states_dir, analysis_dir, metrics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    layer_metrics: dict[int, dict[str, Any]] = {}
    layer_accuracies: dict[int, float] = {}

    for layer_num in selected_layers:
        logger.info("--- Training sklearn probe for layer %s ---", layer_num)
        x_train, y_train = layer_arrays(train_acts, layer_num)
        x_val, y_val = layer_arrays(test_acts, layer_num)

        if x_train.size == 0 or x_val.size == 0:
            logger.warning("Skipping layer %s due to empty train/val arrays.", layer_num)
            layer_accuracies[layer_num] = 0.0
            continue

        estimator = _build_estimator(sklearn_cfg)
        estimator.fit(x_train, y_train)
        val_prob = estimator.predict_proba(x_val)[:, 1]

        metrics = compute_binary_metrics(y_true=y_val, y_prob=val_prob, threshold=threshold)
        metrics["train_samples"] = int(len(y_train))
        metrics["val_samples"] = int(len(y_val))
        metrics["val_accuracy_pct"] = float(metrics["accuracy"] * 100.0)

        logger.info(
            "Layer %s | Acc %.2f%% | F1 %.4f | AUROC %s | CM %s",
            layer_num,
            metrics["val_accuracy_pct"],
            metrics["f1_score"],
            f"{metrics['auroc_score']:.4f}" if metrics["auroc_score"] is not None else "N/A",
            metrics["confusion_matrix"],
        )

        layer_metrics[layer_num] = metrics
        layer_accuracies[layer_num] = metrics["val_accuracy_pct"]

        with (states_dir / f"probe_layer_{layer_num}.pkl").open("wb") as handle:
            pickle.dump(estimator, handle)

        layer_dir = analysis_dir / f"layer_{layer_num}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / f"activations_layer_{layer_num}.npy", x_val)
        np.save(layer_dir / "labels.npy", y_val.astype(np.float32))

        if isinstance(estimator, Pipeline):
            model = estimator.named_steps["logreg"]
        else:
            model = estimator
        np.save(layer_dir / "probe_weights.npy", model.coef_)
        np.save(layer_dir / "probe_biases.npy", model.intercept_)

    best_layer = None
    best_accuracy = None
    if layer_accuracies:
        best_layer = max(layer_accuracies, key=layer_accuracies.get)
        best_accuracy = layer_accuracies[best_layer]

    with (metrics_dir / "layer_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in layer_metrics.items()}, handle, indent=2)
    with (states_dir / "layer_wise_accuracies.json").open("w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in layer_accuracies.items()}, handle, indent=2)
    with (states_dir / "best_probe_layer.txt").open("w", encoding="utf-8") as handle:
        if best_layer is None:
            handle.write("No valid layers were trained.\n")
        else:
            handle.write(f"Best performing probe layer: {best_layer}\n")
            handle.write(f"Validation Accuracy: {best_accuracy:.2f}%\n")

    summary = {
        "backend": "sklearn",
        "selected_layers": selected_layers,
        "best_layer": best_layer,
        "best_accuracy_pct": best_accuracy,
        "output_root": str(output_root),
    }
    with (metrics_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

