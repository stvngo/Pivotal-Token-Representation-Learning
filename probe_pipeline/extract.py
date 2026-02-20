"""Activation extraction pipeline stage."""

from __future__ import annotations

from typing import Any

from .activations import extract_and_label_all_layers, save_activation_store
from .modeling import load_model_and_tokenizer


def run_activation_extraction(config: dict[str, Any], logger: Any) -> dict[str, str]:
    """Extract and cache train/test layer activations from probe datasets."""
    from datasets import load_from_disk

    dataset_cfg = config["paths"]["datasets"]
    activations_cfg = config["paths"]["activations"]
    model_cfg = config["model"]

    train_dataset = load_from_disk(dataset_cfg["train_probe"])
    test_dataset = load_from_disk(dataset_cfg["test_probe"])
    model, tokenizer, device = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device_name=config.get("device", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        output_hidden_states=True,
    )

    logger.info("Extracting training activations...")
    train_acts = extract_and_label_all_layers(train_dataset, model, tokenizer, device, logger=logger)
    logger.info("Extracting testing activations...")
    test_acts = extract_and_label_all_layers(test_dataset, model, tokenizer, device, logger=logger)

    save_activation_store(train_acts, activations_cfg["train"])
    save_activation_store(test_acts, activations_cfg["test"])
    logger.info("Saved train activations to %s", activations_cfg["train"])
    logger.info("Saved test activations to %s", activations_cfg["test"])
    return {"train": activations_cfg["train"], "test": activations_cfg["test"]}

