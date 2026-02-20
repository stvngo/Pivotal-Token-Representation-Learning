"""Preprocessing wrappers for the probe pipeline."""

from __future__ import annotations

from typing import Any

from probe_pipeline.preprocess import (
    create_doubled_negatives_dataset,
    create_labeled_probe_example,
    create_probe_dataset,
    save_probe_dataset,
)


def preprocess_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No-op hook kept for compatibility."""
    return data


def create_labeled_list_dataset(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: str,
    add_random_tokens: int = 5,
) -> dict[str, Any]:
    """Create one compact labeled row for a grouped query."""
    return create_labeled_probe_example(
        examples=examples,
        tokenizer=tokenizer,
        model=model,
        device=device,
        add_random_tokens=add_random_tokens,
    )


def create_dataset(
    train_raw: list[dict[str, Any]],
    test_raw: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build train and test compact probe datasets."""
    train_rows = create_probe_dataset(train_raw, tokenizer, model=model, device=device)
    test_rows = create_probe_dataset(test_raw, tokenizer, model=model, device=device)
    return train_rows, test_rows


def verify_data(data: list[dict[str, Any]]) -> bool:
    """Light sanity check for generated rows."""
    if not data:
        return False
    sample = data[0]
    if not {"text", "labels", "original_dataset_item_id"}.issubset(set(sample.keys())):
        return False
    return len(sample["labels"]) > 0


def save_data(data: list[dict[str, Any]], path: str) -> None:
    """Save generated rows to HuggingFace dataset directory."""
    save_probe_dataset(rows=data, output_dir=path)