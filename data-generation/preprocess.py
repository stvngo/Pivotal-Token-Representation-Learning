'''
Extract pivotal token positions from the dataset for linear probe usage

- Append 5 random tokens to the end of each sample
- Label position immediately preceding the pivotal token as 1
- Label all other positions as 0
- Verify the processed dataset
- Save the processed dataset to a file
'''

import os
from datasets import Dataset as HFDataset
from collections import defaultdict
from tqdm.auto import tqdm
import torch
import random
from utils.utils import set_seed
import time
import logging

def preprocess_data(data: list[dict]) -> list[dict]:

    """
    Extract pivotal token positions from the dataset for linear probe usage
    """
    return data # TODO: Implement data preprocessing logic

def create_labeled_list_dataset(
    examples: list,
    tokenizer,
    model,
    device,
    add_random_tokens: int = 5
) -> dict:
    """
    Processes a list of examples for a single query ID to generate a single
    data row with a balanced set of positive and negative labels.
    """
    if not examples:
        return {}

    # 1. Find the example with the longest pivot_context
    longest_example = max(examples, key=lambda ex: len(ex['pivot_context']))
    longest_context_text = longest_example['pivot_context'] + longest_example['pivot_token']
    longest_context_ids = tokenizer.encode(longest_context_text, add_special_tokens=False)
    query_id = examples[0].get('dataset_item_id', 'N/A')

    print(f"\n--- Processing Query ID: {query_id} ---")
    print(f"Longest context has {len(longest_context_ids)} tokens.")

    # 2. Safely identify pivotal token positions
    pivotal_positions = set()
    if len(longest_context_ids) > 1:
        pos = len(longest_context_ids) - 2
        if pos >= 0:
            pivotal_positions.add(pos)
            print(f"  - Added pivotal position {pos} from the longest context.")

    for example in examples:
        if example != longest_example:
            shorter_context = example['pivot_context']
            if longest_context_text.startswith(shorter_context + example["pivot_token"]):
                shorter_context_ids = tokenizer.encode(shorter_context, add_special_tokens=False)
                if len(shorter_context_ids) > 1:
                    pos = len(shorter_context_ids) - 1
                    pivotal_positions.add(pos)
                    print(f"  - Added pivotal position {pos} from a prefix context.")

    num_positives = len(pivotal_positions)
    print(f"  - Found {num_positives} pivotal positions.")

    # 3. Identify the "answer" portion and sample negatives
    query_text = longest_example['query']
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)
    answer_start_pos = len(query_ids) - 1

    # ********** MODIFIED SAMPLING LOGIC **********
    # Sample 2 * num_positives from the answer portion and appended tokens

    # Define the sampling range: from the end of the original query (inclusive) to the end of the text
    # The end of the original query is approximately at index answer_start_pos
    # The end of the text is the total length of the tokenized text
    text_token_ids = tokenizer.encode(longest_context_text, add_special_tokens=False)
    total_text_length = len(text_token_ids)

    # Possible negative indices are from answer_start_pos up to total_text_length - 1
    # Exclude any indices that are pivotal positions
    possible_negative_indices = [
        i for i in range(answer_start_pos, total_text_length)
        if i not in pivotal_positions
    ]

    # Determine how many negatives to sample (2 * num_positives)
    num_negatives_to_sample = 2 * num_positives

    sampled_negative_positions = set()

    if len(possible_negative_indices) >= num_negatives_to_sample:
        # If we have enough possible negatives, sample directly
        sampled_negative_positions = set(random.sample(possible_negative_indices, num_negatives_to_sample))
        print(f"  - Sampled {len(sampled_negative_positions)} negative positions from the answer portion/appended tokens.")
    else:
        # If not enough, take all available and warn
        sampled_negative_positions.update(possible_negative_indices)
        print(f"  - WARNING: Not enough possible negatives ({len(possible_negative_indices)}) to sample {num_negatives_to_sample}. Sampled all available.")


    # 4. Generate and append random tokens
    final_text_token_ids = longest_context_ids
    if add_random_tokens > 0 and model and device:
        inputs = tokenizer(longest_context_text, return_tensors='pt', add_special_tokens=False).to(device)
        with torch.no_grad():
            generated_outputs = model.generate(
                **inputs,
                max_new_tokens=add_random_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        final_text_token_ids = generated_outputs[0].tolist()
        print(f"  - Appended {len(final_text_token_ids) - len(longest_context_ids)} random tokens.")

    final_text = tokenizer.decode(final_text_token_ids, skip_special_tokens=True)

    # Re-calculate total_text_length based on the final text with appended tokens
    final_text_token_ids = tokenizer.encode(final_text, add_special_tokens=False)
    total_text_length_with_appended = len(final_text_token_ids)


    # 5. Create the labels list
    labels = [0] * total_text_length_with_appended # Initialize with zeros

    # Add positive labels
    for i in pivotal_positions:
        if i < total_text_length_with_appended:
            labels[i] = 1
        else:
            print(f"  - WARNING: Pivotal position {i} is out of bounds for final text length {total_text_length_with_appended}. Skipping.")


    # Add sampled negative labels
    for i in sampled_negative_positions:
         if i < total_text_length_with_appended:
            # Ensure we don't overwrite a positive label (shouldn't happen with sampling logic)
            if labels[i] != 1:
                labels[i] = -1
            # else:
                # print(f"  - WARNING: Sampled negative position {i} is already labeled 1. Skipping.")
         else:
             print(f"  - WARNING: Sampled negative position {i} is out of bounds for final text length {total_text_length_with_appended}. Skipping.")


    # 6. Return the single data row
    return {
        'text': final_text,
        'labels': labels,
        'original_dataset_item_id': query_id
    }

# --- Main Execution ---
# This part will be executed when the cell runs.
# It will regenerate probe_train_data_focused and probe_test_data_focused
# using the modified function.

train_query_groups = defaultdict(list)
for example in train_raw:
    train_query_groups[example['dataset_item_id']].append(example)

test_query_groups = defaultdict(list)
for example in test_raw:
    test_query_groups[example['dataset_item_id']].append(example)

print("Generating training data with modified negative sampling...")
probe_train_data_focused = [create_labeled_list_dataset_focused(examples, tokenizer, model, device) for _, examples in tqdm(train_query_groups.items(), desc="Processing Train Queries")]

print("\nGenerating testing data with modified negative sampling...")
probe_test_data_focused = [create_labeled_list_dataset_focused(examples, tokenizer, model, device) for _, examples in tqdm(test_query_groups.items(), desc="Processing Test Queries")]

print(f"\nGenerated {len(probe_train_data_focused)} training examples for the focused dataset (modified sampling).")
print(f"Generated {len(probe_test_data_focused)} testing examples for the focused dataset (modified sampling).")

