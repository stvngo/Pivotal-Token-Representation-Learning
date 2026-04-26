"""Locks in the off-by-one convention from docs/issues.md Issue #2.

Asserts that ``_get_decoder_layer(model, L).register_forward_hook`` captures a
tensor identical to ``outputs.hidden_states[L]`` for every layer ``L`` we care
about (the probe layer and a sample of others).

Skipped when the Qwen3-0.6B weights are not locally available so this can run
on developer machines without breaking on CI / Colab.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from probe_pipeline.steering import _get_decoder_layer


def _qwen3_local_or_skip():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"transformers not importable: {exc}")

    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path.home() / ".cache" / "huggingface" / "hub",
        )
    )
    qwen_dir = cache_root / "models--Qwen--Qwen3-0.6B"
    if not qwen_dir.exists():
        pytest.skip(f"Qwen3-0.6B not cached at {qwen_dir}; skipping layer-alignment test.")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


@pytest.mark.parametrize("layer_idx", [0, 1, 14, 27])
def test_hook_captures_hidden_states_layer(layer_idx: int) -> None:
    model, tok = _qwen3_local_or_skip()

    captured: dict[str, torch.Tensor] = {}

    def cap(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["pre"] = h.detach()
        return output

    handle = _get_decoder_layer(model, layer_idx).register_forward_hook(cap)
    try:
        with torch.no_grad():
            inputs = tok("Hello.", return_tensors="pt", add_special_tokens=False)
            outputs = model(**inputs, output_hidden_states=True)
    finally:
        handle.remove()

    assert "pre" in captured, "hook did not fire"
    expected = outputs.hidden_states[layer_idx]
    assert captured["pre"].shape == expected.shape, (
        f"shape mismatch at layer {layer_idx}: "
        f"hook={tuple(captured['pre'].shape)} vs hidden_states={tuple(expected.shape)}"
    )
    assert torch.allclose(captured["pre"], expected, atol=1e-5, rtol=1e-4), (
        f"hook output differs from outputs.hidden_states[{layer_idx}] - "
        f"max abs diff = {(captured['pre'] - expected).abs().max().item():.3e}"
    )


def test_layer28_is_post_rmsnorm_excluded() -> None:
    """``hidden_states[28]`` is post-final-RMSNorm (tie_last_hidden_states=True)
    per docs/issues.md §1. ``_get_decoder_layer(model, 28)`` would be
    ``layers[27]``, whose output is *pre*-RMSNorm and therefore NOT equal to
    ``hidden_states[28]``. We don't claim alignment at the final index; this
    test documents the boundary so future contributors don't get surprised.
    """
    model, _ = _qwen3_local_or_skip()
    n = len(model.model.layers)
    assert n == 28, f"expected 28 Qwen3 decoder layers, got {n}"
    last = _get_decoder_layer(model, n)
    assert last is model.model.layers[n - 1]
