"""Activation steering for pivotal-token amplification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


def load_steering_vector(
    analysis_root: Path,
    layer: int,
    vector_type: str = "centroid_diff",
) -> torch.Tensor:
    """Load steering vector from analysis data. Returns (hidden_dim,) tensor."""
    layer_dir = analysis_root / f"layer_{layer}"
    if vector_type == "centroid_diff":
        x_path = layer_dir / f"activations_layer_{layer}.npy"
        y_path = layer_dir / "labels.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing activations/labels for layer {layer} in {analysis_root}")
        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).reshape(-1)
        mask_pos = y >= 0.5
        mask_neg = y < 0.5
        mu_pos = x[mask_pos].mean(axis=0)
        mu_neg = x[mask_neg].mean(axis=0)
        diff = mu_pos - mu_neg
        v = torch.tensor(diff, dtype=torch.float32)
    elif vector_type == "probe_weights":
        w_path = layer_dir / "probe_weights.npy"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing probe weights for layer {layer} in {analysis_root}")
        w = np.load(w_path).astype(np.float32)
        v = torch.tensor(w.flatten(), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown vector_type: {vector_type}")
    return v


def _get_decoder_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """Get the decoder layer at index layer_idx. Handles common HuggingFace layouts."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return model.decoder.layers[layer_idx]
    raise AttributeError(f"Cannot find decoder layers in model: {type(model)}")


def _make_steering_hook(
    vector: torch.Tensor,
    strength: float,
    device: torch.device,
) -> Callable[..., Any]:
    """Create a forward hook that adds strength * vector to the layer output."""

    def hook(module: nn.Module, args: tuple, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # hidden: (batch, seq, hidden_dim)
        v = vector.to(hidden.device).to(hidden.dtype)
        if v.dim() == 1:
            v = v.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        delta = strength * v
        if isinstance(output, tuple):
            return (hidden + delta,) + output[1:]
        return hidden + delta

    return hook


def register_steering_hooks(
    model: nn.Module,
    layer: int,
    vector: torch.Tensor,
    strength: float,
    device: torch.device,
) -> list[Any]:
    """Register forward hooks for steering. Returns list of handle objects for removal."""
    decoder_layer = _get_decoder_layer(model, layer)
    hook_fn = _make_steering_hook(vector, strength, device)
    handle = decoder_layer.register_forward_hook(hook_fn)
    return [handle]


def remove_hooks(handles: list[Any]) -> None:
    """Remove registered hooks."""
    for h in handles:
        h.remove()


def generate_with_steering(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    layer: int,
    vector: torch.Tensor,
    strength: float,
    device: torch.device,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.9,
    pad_token_id: int | None = None,
) -> str:
    """Generate text with steering applied at the specified layer."""
    model.eval()
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    handles = register_steering_hooks(model, layer, vector, strength, device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        remove_hooks(handles)


def create_steering_config(
    config: dict[str, Any],
    backend: str,
    layer: int,
    factors: list[float],
    vector_type: str = "centroid_diff",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Create and save steering config + vector. Returns config dict."""
    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    analysis_root = backend_root / "analysis_data"
    vector = load_steering_vector(analysis_root, layer, vector_type=vector_type)
    vector_np = vector.numpy()

    if output_dir is None:
        output_dir = backend_root / "steering_configs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_filename = f"steering_layer{layer}_vector.npy"
    vector_path = output_dir / vector_filename
    np.save(vector_path, vector_np)

    steering_config = {
        "backend": backend,
        "layer": layer,
        "vector_type": vector_type,
        "factors": factors,
        "hidden_dim": int(vector_np.shape[0]),
        "vector_norm": float(np.linalg.norm(vector_np)),
        "vector_path": vector_filename,
    }

    config_path = output_dir / f"steering_layer{layer}.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(steering_config, f, indent=2)

    return steering_config


def load_steering_config(config_path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    """Load steering config and vector. Returns (config_dict, vector_tensor)."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    vector_path = Path(cfg["vector_path"])
    if not vector_path.is_absolute():
        vector_path = (config_path.parent / vector_path).resolve()
    vector_np = np.load(str(vector_path))
    vector = torch.tensor(vector_np, dtype=torch.float32)
    return cfg, vector
