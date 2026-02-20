"""Layer-wise PyTorch linear probe training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .activations import layer_arrays, list_available_layers, load_activation_store
from .common import parse_layer_selection, pick_device
from .metrics import compute_binary_metrics


class LinearProbe(nn.Module):
    """Single linear layer probe (matches notebook setup)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _sample_cap(
    x: np.ndarray,
    y: np.ndarray,
    max_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optional cap for smoke tests and fast iterations."""
    if max_per_class is None:
        return x, y
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for cls in [0, 1]:
        cls_indices = np.where(y == cls)[0]
        if cls_indices.size <= max_per_class:
            selected_indices.extend(cls_indices.tolist())
            continue
        sampled = rng.choice(cls_indices, size=max_per_class, replace=False)
        selected_indices.extend(sampled.tolist())

    selected_indices = sorted(selected_indices)
    return x[selected_indices], y[selected_indices]


def run_pytorch_training(
    config: dict[str, Any],
    logger: Any,
    layers_override: list[int] | None = None,
    num_layers_override: int | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train layer-wise probes and save artifacts + metrics."""
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
        raise ValueError("No layers selected for training.")

    device = pick_device(config.get("device", "auto"))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    epochs = int(epochs_override if epochs_override is not None else training_cfg.get("epochs", 10))
    batch_size = int(training_cfg.get("batch_size", 32))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    threshold = float(training_cfg.get("threshold", 0.5))
    max_train = training_cfg.get("max_train_samples_per_class")
    max_val = training_cfg.get("max_val_samples_per_class")

    output_root = Path(paths["outputs"]["pytorch_dir"])
    states_dir = output_root / "probe_states"
    analysis_dir = output_root / "analysis_data"
    metrics_dir = output_root / "metrics"
    for directory in [states_dir, analysis_dir, metrics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    layer_metrics: dict[int, dict[str, Any]] = {}
    layer_accuracies: dict[int, float] = {}
    input_dim: int | None = None

    for layer_num in selected_layers:
        logger.info("--- Training probe for layer %s ---", layer_num)
        x_train, y_train = layer_arrays(train_acts, layer_num)
        x_val, y_val = layer_arrays(test_acts, layer_num)

        if x_train.size == 0 or x_val.size == 0:
            logger.warning("Skipping layer %s due to empty train/val arrays.", layer_num)
            layer_accuracies[layer_num] = 0.0
            continue

        x_train, y_train = _sample_cap(x_train, y_train, max_train, seed)
        x_val, y_val = _sample_cap(x_val, y_val, max_val, seed)
        input_dim = x_train.shape[1]

        logger.info(
            "Layer %s | train=%s samples (%s pos / %s neg), val=%s samples (%s pos / %s neg)",
            layer_num,
            len(y_train),
            int((y_train == 1).sum()),
            int((y_train == 0).sum()),
            len(y_val),
            int((y_val == 1).sum()),
            int((y_val == 0).sum()),
        )

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)

        train_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=bool(training_cfg.get("shuffle", True)),
        )

        probe = LinearProbe(input_dim=input_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(probe.parameters(), lr=learning_rate)

        epoch_losses: list[float] = []
        for epoch in range(epochs):
            probe.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = probe(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            mean_loss = total_loss / max(1, len(train_loader))
            epoch_losses.append(mean_loss)
            logger.info(
                "Layer %s | Epoch %s/%s | Loss %.6f",
                layer_num,
                epoch + 1,
                epochs,
                mean_loss,
            )

        probe.eval()
        with torch.no_grad():
            val_logits = probe(x_val_tensor)
            val_prob = torch.sigmoid(val_logits).cpu().numpy().reshape(-1)

        metrics = compute_binary_metrics(y_true=y_val, y_prob=val_prob, threshold=threshold)
        metrics["train_samples"] = int(len(y_train))
        metrics["val_samples"] = int(len(y_val))
        metrics["epoch_losses"] = epoch_losses
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

        torch.save(probe.state_dict(), states_dir / f"probe_layer_{layer_num}.pth")
        layer_dir = analysis_dir / f"layer_{layer_num}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / f"activations_layer_{layer_num}.npy", x_val)
        np.save(layer_dir / "labels.npy", y_val.astype(np.float32))
        np.save(layer_dir / "probe_weights.npy", probe.linear.weight.detach().cpu().numpy())
        np.save(layer_dir / "probe_biases.npy", probe.linear.bias.detach().cpu().numpy())

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
        "backend": "pytorch",
        "device": str(device),
        "selected_layers": selected_layers,
        "input_dim": input_dim,
        "best_layer": best_layer,
        "best_accuracy_pct": best_accuracy,
        "output_root": str(output_root),
    }
    with (metrics_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

