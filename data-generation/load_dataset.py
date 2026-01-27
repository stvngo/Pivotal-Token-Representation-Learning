'''
Load the dataset from the given path
'''

from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def load_dataset(path: str) -> list[dict]:
    return data # TODO: Implement dataset loading

def split_pts_by_query(dataset_path: str, test_size: float = 0.2, subset_size: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load PTS dataset, remove duplicates, and split by query ID to avoid data leakage.

    :param dataset_path: Path/name of your PTS dataset on HuggingFace
    :param test_size: Fraction for test split
    :param subset_size: If provided, creates a subset of the dataset for debugging.
    :return: train_dataset, test_dataset split by query
    """
    # Load the PTS dataset with explicit configuration
    print(f"Loading dataset: {dataset_path}")

    try:
        # Try loading without any wildcards or special patterns
        dataset = load_dataset(dataset_path, split='train')
        print(f"Loaded {len(dataset)} examples")

    except Exception as e:
        print(f"Error with split='train', trying default loading: {e}")
        try:
            # Try loading all splits then select one
            dataset_dict = load_dataset(dataset_path)
            print(f"Available splits: {list(dataset_dict.keys())}")

            # Get the main split
            if 'train' in dataset_dict:
                dataset = dataset_dict['train']
            else:
                split_name = list(dataset_dict.keys())[0]
                dataset = dataset_dict[split_name]
                print(f"Using split: {split_name}")

        except Exception as e2:
            print(f"Final error: {e2}")
            print("Try loading the dataset manually first to debug")
            raise e2

    # Create a subset if requested
    if subset_size:
        dataset = dataset.select(range(min(subset_size, len(dataset))))
        print(f"Using a subset of {len(dataset)} examples for debugging.")

    # Remove duplicates
    df = dataset.to_pandas()

    # Drop the timestamp column if it exists
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
        print("Dropped the 'timestamp' column.")

    num_rows_before = len(df)
    df_deduplicated = df.drop_duplicates()
    num_rows_after = len(df_deduplicated)
    num_duplicates_removed = num_rows_before - num_rows_after

    print(f"Removed {num_duplicates_removed} duplicate rows.")
    print(f"Number of rows left: {num_rows_after}")

    # Count number of examples where first token is pivotal
    count = 0
    for _, row in df_deduplicated.iterrows():
        if row["query"] == row['pivot_context']:
            count += 1

    total_examples = len(df_deduplicated)
    percentage = (count / total_examples) * 100

    print(f"Sanity Check Results:")
    print(f"Number of examples where the first token after the query is pivotal: {count}")
    print(f"Total number of examples: {total_examples}")
    print(f"Percentage: {percentage:.2f}%")

    dataset = Dataset.from_pandas(df_deduplicated)


    # Get unique query IDs
    unique_query_ids = list(set(dataset['dataset_item_id']))
    print(f"Total unique queries: {len(unique_query_ids)}")

    # Split query IDs (not individual examples)
    train_query_ids, test_query_ids = train_test_split( # train: 1,3,4,... | test: 2,5,...
        unique_query_ids,
        test_size=test_size,
        random_state=42 # for reproducibility
    )

    # Filter dataset by query splits
    train_dataset = dataset.filter(lambda x: x['dataset_item_id'] in train_query_ids)
    test_dataset = dataset.filter(lambda x: x['dataset_item_id'] in test_query_ids)

    print(f"Train queries: {len(train_query_ids)}, Train examples: {len(train_dataset)}")
    print(f"Test queries: {len(test_query_ids)}, Test examples: {len(test_dataset)}")

    return train_dataset, test_dataset

def verify_dataset(data: list[dict]) -> bool:
    return True # TODO: Implement dataset verification

def save_dataset(data: list[dict], path: str) -> None:
    return None # TODO: Implement dataset saving