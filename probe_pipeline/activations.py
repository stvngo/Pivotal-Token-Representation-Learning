"""Activation extraction and serialization helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


ActivationStore = dict[int, dict[str, list[torch.Tensor]]]


def _to_cpu_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def normalize_activation_store(raw: dict[Any, Any]) -> ActivationStore:
    """Normalize loaded activation dictionary to a stable schema."""
    normalized: ActivationStore = {}
    for layer_key, buckets in raw.items():
        layer = int(layer_key)
        pivotal = [_to_cpu_float_tensor(x) for x in buckets.get("pivotal", [])]
        non_pivotal = [_to_cpu_float_tensor(x) for x in buckets.get("non_pivotal", [])]
        normalized[layer] = {"pivotal": pivotal, "non_pivotal": non_pivotal}
    return normalized


def load_activation_store(path: str | Path) -> ActivationStore:
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in activation file, got {type(raw)}.")
    return normalize_activation_store(raw)


def save_activation_store(data: ActivationStore, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def list_available_layers(train_acts: ActivationStore, test_acts: ActivationStore) -> list[int]:
    return sorted(set(train_acts.keys()).intersection(set(test_acts.keys())))


def layer_arrays(
    activations: ActivationStore,
    layer_num: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays for a single layer."""
    pivotal = activations[layer_num]["pivotal"]
    non_pivotal = activations[layer_num]["non_pivotal"]
    if not pivotal or not non_pivotal:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x_pos = torch.stack([x.detach().cpu().float() for x in pivotal], dim=0).numpy()
    x_neg = torch.stack([x.detach().cpu().float() for x in non_pivotal], dim=0).numpy()
    x = np.vstack([x_pos, x_neg]).astype(np.float32)
    y = np.concatenate(
        [
            np.ones(len(x_pos), dtype=np.int64),
            np.zeros(len(x_neg), dtype=np.int64),
        ]
    )
    return x, y


def extract_and_label_all_layers(
    dataset: list[dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    logger: Any | None = None,
) -> ActivationStore:
    """Extract hidden states for labels 1 and -1 across all layers."""
    model = model.to(device)
    model.eval()
    all_layers_activations: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"pivotal": [], "non_pivotal": []}
    )

    with torch.no_grad():
        for example in tqdm(dataset, desc="Extracting activations"):
            text = example["text"]
            labels = example["labels"]
            sample_id = example.get("original_dataset_item_id", "N/A")

            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            all_hidden_states = outputs.hidden_states

            for layer_num, hidden_states_layer in enumerate(all_hidden_states):
                hidden_states = hidden_states_layer.squeeze(0)
                min_len = min(hidden_states.shape[0], len(labels))
                if hidden_states.shape[0] != len(labels) and logger:
                    logger.warning(
                        "Layer %s mismatch for query %s: tokens=%s labels=%s -> truncating=%s",
                        layer_num,
                        sample_id,
                        hidden_states.shape[0],
                        len(labels),
                        min_len,
                    )

                for idx in range(min_len):
                    label = labels[idx]
                    activation = hidden_states[idx].detach().cpu().float()
                    if label == 1:
                        all_layers_activations[layer_num]["pivotal"].append(activation)
                    elif label == -1:
                        all_layers_activations[layer_num]["non_pivotal"].append(activation)

    return dict(all_layers_activations)

