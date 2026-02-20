"""Dataset loading wrappers for the probe pipeline."""

from __future__ import annotations

from typing import Any

from datasets import Dataset

from probe_pipeline.dataset import (
    dataset_summary,
    load_pts_dataset,
    save_split_datasets,
    split_pts_by_query as split_pts_by_query_core,
)


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load a dataset and return as list of dictionaries."""
    dataset = load_pts_dataset(dataset_path=path)
    return list(dataset)


def split_pts_by_query(
    dataset_path: str,
    test_size: float = 0.2,
    subset_size: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Public wrapper preserving the original function name."""
    return split_pts_by_query_core(
        dataset_path=dataset_path,
        test_size=test_size,
        subset_size=subset_size,
    )


def preview(train_dataset: Dataset, test_dataset: Dataset) -> None:
    """Print a quick split preview."""
    train_info = dataset_summary(train_dataset)
    test_info = dataset_summary(test_dataset)
    print(f"Train summary: {train_info}")
    print(f"Test summary: {test_info}")
    if len(train_dataset) > 0:
        print("Train sample:", train_dataset[0])
    if len(test_dataset) > 0:
        print("Test sample:", test_dataset[0])


def verify_dataset(data: list[dict[str, Any]]) -> bool:
    """Simple structural validation."""
    if not data:
        return False
    required = {"dataset_item_id", "query", "pivot_context", "pivot_token"}
    return required.issubset(set(data[0].keys()))


def save_dataset(train_dataset: Dataset, test_dataset: Dataset, path: str) -> dict[str, str]:
    """Save split datasets at `path/train` and `path/test`."""
    return save_split_datasets(train_dataset=train_dataset, test_dataset=test_dataset, output_root=path)