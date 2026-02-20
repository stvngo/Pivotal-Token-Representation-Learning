"""Shared helpers for training and evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def pick_device(device_name: str = "auto") -> torch.device:
    """Resolve device from config value."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def parse_layer_selection(
    available_layers: Iterable[int],
    explicit_layers: list[int] | None = None,
    num_layers: int | None = None,
) -> list[int]:
    """Get layer IDs from availability and user selection knobs."""
    sorted_layers = sorted(set(int(x) for x in available_layers))
    if explicit_layers:
        explicit = [int(x) for x in explicit_layers]
        return [layer for layer in explicit if layer in sorted_layers]
    if num_layers is None:
        return sorted_layers
    return sorted_layers[: max(0, int(num_layers))]


def tensor_list_to_numpy(acts: list[torch.Tensor]) -> np.ndarray:
    """Stack activation tensors into a 2D float32 numpy array."""
    if not acts:
        return np.empty((0, 0), dtype=np.float32)
    stacked = torch.stack([a.detach().cpu().float() for a in acts], dim=0)
    return stacked.numpy()

