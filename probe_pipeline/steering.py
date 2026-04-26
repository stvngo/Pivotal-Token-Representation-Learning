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
    """Return the module whose forward-hook output IS ``outputs.hidden_states[layer_idx]``.

    Qwen3 / LLaMA-style: ``outputs.hidden_states[k]`` for ``k in 1..N`` is the
    output of ``model.model.layers[k-1]``. Index 0 is the embedding (post
    ``embed_tokens``, pre ``layers[0]``). Forward hooks see the layer output, so:

    - layer_idx == 0  -> hook ``embed_tokens`` (output == hidden_states[0])
    - layer_idx >= 1  -> hook ``layers[layer_idx - 1]``
                          (output == hidden_states[layer_idx])

    Closes Issue #2 from docs/issues.md (probe trained on hidden_states[L] but
    steering was applied to hidden_states[L+1]).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if layer_idx == 0:
            return model.model.embed_tokens
        return model.model.layers[layer_idx - 1]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        if layer_idx == 0:
            return model.transformer.wte
        return model.transformer.h[layer_idx - 1]
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        if layer_idx == 0:
            return model.decoder.embed_tokens
        return model.decoder.layers[layer_idx - 1]
    raise AttributeError(f"Cannot find decoder layers in model: {type(model)}")


HookMode = str  # one of: "additive_raw", "additive_normalized", "projection"
_VALID_MODES = {"additive_raw", "additive_normalized", "projection"}


def make_hook(
    vector: torch.Tensor,
    coef: float,
    mode: HookMode = "additive_normalized",
    position_mask: torch.Tensor | None = None,
) -> Callable[..., Any]:
    """Build a forward hook implementing one of three steering conventions.

    Modes:
        additive_raw         h <- h + coef * v
        additive_normalized  h <- h + coef * ||h_pos|| * v_hat   (per-position)
        projection           h <- h + (alpha - 1) * (h . v_hat) * v_hat
                              (alpha == coef; coef=0 ablates, coef=1 is identity,
                               coef=2 doubles the existing component along v_hat)

    Args:
        vector: (hidden_dim,) tensor. For projection / additive_normalized this
            is normalized to a unit ``v_hat`` internally.
        coef: scalar. Interpretation depends on ``mode`` (see above).
        mode: which convention to apply.
        position_mask: optional (seq_len,) bool tensor selecting positions where
            the perturbation is applied. ``None`` means all positions. Used by
            ``compute_token_nie`` to mask the delta to a single position.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_VALID_MODES}")

    def _expand_mask(mask: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 1:
            mask = mask.view(1, -1, 1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        return mask.to(device=hidden.device, dtype=hidden.dtype)

    def hook(module: nn.Module, args: tuple, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        v = vector.to(device=hidden.device, dtype=hidden.dtype)
        if v.dim() == 1:
            v_full = v.view(1, 1, -1)
        else:
            v_full = v

        if mode == "additive_raw":
            delta = coef * v_full
        elif mode == "additive_normalized":
            v_norm = v.norm() + 1e-8
            v_hat = (v / v_norm).view(1, 1, -1)
            h_norm = hidden.norm(dim=-1, keepdim=True)
            delta = coef * h_norm * v_hat
        else:  # projection
            v_norm = v.norm() + 1e-8
            v_hat = (v / v_norm).view(1, 1, -1)
            proj_scalar = (hidden * v_hat).sum(dim=-1, keepdim=True)
            delta = (coef - 1.0) * proj_scalar * v_hat

        if position_mask is not None:
            delta = delta * _expand_mask(position_mask, hidden)

        new_hidden = hidden + delta
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook


def _make_steering_hook(
    vector: torch.Tensor,
    strength: float,
    device: torch.device,
) -> Callable[..., Any]:
    """Backward-compat wrapper around ``make_hook(mode='additive_raw')``.

    ``strength`` here is the legacy ``factor - 1.0`` used everywhere by the
    pre-Issue-#7 callers. Equivalent to ``coef = strength``.
    """
    del device  # unused; kept for signature compat
    return make_hook(vector, strength, mode="additive_raw")


def register_steering_hooks(
    model: nn.Module,
    layer: int,
    vector: torch.Tensor,
    coef: float,
    device: torch.device,
    mode: HookMode = "additive_raw",
    position_mask: torch.Tensor | None = None,
) -> list[Any]:
    """Register forward hooks for steering. Returns handle list for removal.

    Defaults to ``mode='additive_raw'`` so legacy callers (``coef=factor-1.0``)
    keep their previous numerical behavior. New code should pass ``mode``
    explicitly.
    """
    del device  # accepted for back-compat; no longer needed.
    decoder_layer = _get_decoder_layer(model, layer)
    hook_fn = make_hook(vector, coef, mode=mode, position_mask=position_mask)
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
    mode: HookMode = "additive_raw",
) -> str:
    """Generate text with steering applied at the specified layer.

    ``strength`` is the coefficient passed through to ``make_hook``; its
    interpretation depends on ``mode`` (see ``make_hook``). Default
    ``additive_raw`` preserves legacy behavior.
    """
    model.eval()
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    handles = register_steering_hooks(model, layer, vector, strength, device, mode=mode)
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
