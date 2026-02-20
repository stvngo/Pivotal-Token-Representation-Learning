"""Model loading utilities."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import pick_device


def load_model_and_tokenizer(
    model_name: str,
    device_name: str = "auto",
    trust_remote_code: bool = True,
    output_hidden_states: bool = True,
) -> tuple[Any, Any, torch.device]:
    """Load tokenizer and model with conservative defaults."""
    device = pick_device(device_name)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "output_hidden_states": output_hidden_states,
    }

    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": str(device)}

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device

