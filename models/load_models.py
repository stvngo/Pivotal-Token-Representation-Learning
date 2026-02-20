"""Model loading helper wrappers."""

from __future__ import annotations

from typing import Any

from probe_pipeline.modeling import load_model_and_tokenizer
from utils.utils import set_seed


def load_models(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "auto",
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    """Load model/tokenizer/device with the shared pipeline utility."""
    set_seed(seed)
    return load_model_and_tokenizer(
        model_name=model_name,
        device_name=device,
        trust_remote_code=True,
        output_hidden_states=True,
    )