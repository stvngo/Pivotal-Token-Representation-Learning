"""Dataset synthesis helpers for pivotal-token probing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm


def _group_by_query_id(examples: Iterable[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[example["dataset_item_id"]].append(example)
    return grouped


def create_labeled_probe_example(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    model: Any | None = None,
    device: str | torch.device | None = None,
    add_random_tokens: int = 5,
    negative_to_positive_ratio: float = 2.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Create a single probe row from all examples with the same dataset_item_id.

    Labels are:
      - 1 for token positions immediately preceding pivotal tokens
      - -1 for sampled non-pivotal positions
      - 0 for all other positions
    """
    if not examples:
        return {}

    rng = np.random.default_rng(seed)
    longest_example = max(examples, key=lambda ex: len(ex["pivot_context"]))
    longest_text = longest_example["pivot_context"] + longest_example["pivot_token"]
    longest_ids = tokenizer.encode(longest_text, add_special_tokens=False)
    query_id = examples[0].get("dataset_item_id", "N/A")

    pivotal_positions: set[int] = set()
    if len(longest_ids) > 1:
        pivotal_positions.add(len(longest_ids) - 2)

    for example in examples:
        if example is longest_example:
            continue
        prefix_text = example["pivot_context"] + example["pivot_token"]
        if longest_text.startswith(prefix_text):
            prefix_ids = tokenizer.encode(example["pivot_context"], add_special_tokens=False)
            if len(prefix_ids) > 0:
                pivotal_positions.add(len(prefix_ids) - 1)

    query_ids = tokenizer.encode(longest_example["query"], add_special_tokens=False)
    answer_start = max(0, len(query_ids) - 1)
    possible_negatives = [i for i in range(answer_start, len(longest_ids)) if i not in pivotal_positions]
    requested_negatives = max(1, int(round(len(pivotal_positions) * negative_to_positive_ratio)))
    sampled_negative_positions: set[int] = set()
    if possible_negatives:
        count = min(requested_negatives, len(possible_negatives))
        sampled = rng.choice(np.array(possible_negatives), size=count, replace=False)
        sampled_negative_positions = {int(x) for x in sampled.tolist()}

    final_ids = list(longest_ids)
    if add_random_tokens > 0:
        if model is not None and device is not None:
            dev = torch.device(device) if isinstance(device, str) else device
            inputs = tokenizer(longest_text, return_tensors="pt", add_special_tokens=False).to(dev)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=add_random_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                )
            final_ids = generated[0].detach().cpu().tolist()
        else:
            vocab_size = int(getattr(tokenizer, "vocab_size", 151936))
            random_ids = rng.integers(low=0, high=vocab_size, size=add_random_tokens).tolist()
            final_ids.extend(int(x) for x in random_ids)

    labels = [0] * len(final_ids)
    for pos in pivotal_positions:
        if 0 <= pos < len(labels):
            labels[pos] = 1
    for pos in sampled_negative_positions:
        if 0 <= pos < len(labels) and labels[pos] != 1:
            labels[pos] = -1

    return {
        "text": tokenizer.decode(final_ids, skip_special_tokens=True),
        "labels": labels,
        "original_dataset_item_id": query_id,
    }


def create_probe_dataset(
    raw_dataset: Iterable[dict[str, Any]],
    tokenizer: Any,
    model: Any | None = None,
    device: str | torch.device | None = None,
    add_random_tokens: int = 5,
    negative_to_positive_ratio: float = 2.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create a compact per-query probe dataset."""
    grouped = _group_by_query_id(raw_dataset)
    rows: list[dict[str, Any]] = []
    for _, examples in tqdm(grouped.items(), desc="Building probe dataset"):
        row = create_labeled_probe_example(
            examples=examples,
            tokenizer=tokenizer,
            model=model,
            device=device,
            add_random_tokens=add_random_tokens,
            negative_to_positive_ratio=negative_to_positive_ratio,
            seed=seed,
        )
        if row:
            rows.append(row)
    return rows


def create_doubled_negatives_dataset(
    dataset: Iterable[dict[str, Any]],
    tokenizer: Any,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Add additional negative labels to approximately double existing negatives.

    New negatives are sampled from currently-unlabeled positions beginning near
    the earliest existing labeled position and extending to the end of sequence.
    """
    rng = np.random.default_rng(seed)
    new_examples: list[dict[str, Any]] = []

    for example in tqdm(dataset, desc="Creating doubled-negatives dataset"):
        text = example["text"]
        labels = list(example["labels"])
        query_id = example.get("original_dataset_item_id", "N/A")

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_len = len(token_ids)

        if len(labels) < total_len:
            labels.extend([0] * (total_len - len(labels)))
        elif len(labels) > total_len:
            labels = labels[:total_len]

        existing_negatives = labels.count(-1)
        if existing_negatives <= 0:
            new_examples.append(
                {
                    "text": text,
                    "labels": labels,
                    "original_dataset_item_id": query_id,
                }
            )
            continue

        labeled_positions = [idx for idx, value in enumerate(labels) if value != 0]
        sampling_start = min(labeled_positions) if labeled_positions else 0
        candidate_indices = [idx for idx in range(sampling_start, total_len) if labels[idx] == 0]

        additional_needed = existing_negatives
        additional_count = min(additional_needed, len(candidate_indices))
        if additional_count > 0:
            sampled = rng.choice(np.array(candidate_indices), size=additional_count, replace=False)
            for idx in sampled.tolist():
                labels[int(idx)] = -1

        new_examples.append(
            {
                "text": text,
                "labels": labels,
                "original_dataset_item_id": query_id,
            }
        )

    return new_examples


def save_probe_dataset(rows: list[dict[str, Any]], output_dir: str) -> None:
    """Persist generated dataset to disk in HuggingFace format."""
    from datasets import Dataset as HFDataset

    dataset = HFDataset.from_list(rows)
    dataset.save_to_disk(output_dir)

