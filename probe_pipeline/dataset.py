"""Dataset loading and split helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


def load_pts_dataset(dataset_path: str, split: str = "train") -> Dataset:
    """Load a HuggingFace dataset split with fallback behavior."""
    try:
        dataset = load_dataset(dataset_path, split=split)
    except Exception:
        loaded = load_dataset(dataset_path)
        if isinstance(loaded, DatasetDict):
            if split in loaded:
                dataset = loaded[split]
            elif "train" in loaded:
                dataset = loaded["train"]
            else:
                first_key = list(loaded.keys())[0]
                dataset = loaded[first_key]
        else:
            dataset = loaded
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected HuggingFace Dataset, received {type(dataset)}")
    return dataset


def split_pts_by_query(
    dataset_path: str,
    test_size: float = 0.2,
    subset_size: int | None = None,
    random_state: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Load PTS rows, deduplicate them, and split by dataset_item_id.
    """
    dataset = load_pts_dataset(dataset_path=dataset_path, split="train")
    if subset_size is not None:
        dataset = dataset.select(range(min(subset_size, len(dataset))))

    df = dataset.to_pandas()
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    df = df.drop_duplicates()

    dedup_dataset = Dataset.from_pandas(df)
    unique_query_ids = sorted(set(dedup_dataset["dataset_item_id"]))
    train_query_ids, test_query_ids = train_test_split(
        unique_query_ids,
        test_size=test_size,
        random_state=random_state,
    )

    train_dataset = dedup_dataset.filter(lambda x: x["dataset_item_id"] in train_query_ids)
    test_dataset = dedup_dataset.filter(lambda x: x["dataset_item_id"] in test_query_ids)
    return train_dataset, test_dataset


def save_split_datasets(train_dataset: Dataset, test_dataset: Dataset, output_root: str | Path) -> dict[str, str]:
    """Save train/test datasets to disk."""
    root = Path(output_root)
    train_path = root / "train"
    test_path = root / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(train_path))
    test_dataset.save_to_disk(str(test_path))
    return {"train": str(train_path), "test": str(test_path)}


def dataset_summary(dataset: Dataset) -> dict[str, Any]:
    """Quick summary useful for CLI logging."""
    as_df = dataset.to_pandas()
    first_token_pivotal = int((as_df["query"] == as_df["pivot_context"]).sum()) if "query" in as_df.columns else 0
    return {
        "rows": int(len(as_df)),
        "unique_queries": int(as_df["dataset_item_id"].nunique()) if "dataset_item_id" in as_df.columns else None,
        "first_token_pivotal": first_token_pivotal,
    }

