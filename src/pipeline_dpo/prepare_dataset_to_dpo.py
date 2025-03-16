import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import torch
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset


def load_dpo_dataset(
    file_path,
    include_scores=False,
    diff_threshold: Optional[float] = None,
    similarity_metric: str = "spearman",  # Can be "spearman" or "cosine"
) -> Dataset:
    """
    Efficiently loads a JSONL dataset into a Hugging Face Dataset format for DPOTrainer.
    Uses `load_dataset` for direct streaming and minimal memory usage.

    Args:
        file_path: Path to the JSONL file containing the dataset
        include_scores: Whether to include the scores in the dataset
        diff_threshold: Optional minimum threshold for difference between
                        best and worst scores. If provided, filters out examples
                        where the difference is below this threshold.
        similarity_metric: Which similarity metric to use for chosen/rejected pairs.
                          Options: "spearman" or "cosine" (default: "spearman")

    Returns:
        A Hugging Face Dataset for DPO training
    """
    # Validate similarity metric
    if similarity_metric not in ["spearman", "cosine"]:
        raise ValueError(f"Invalid similarity metric: {similarity_metric}. Must be 'spearman' or 'cosine'.")

    # Construct the correct keys based on chosen metric
    best_key = f"{similarity_metric}_best"
    worst_key = f"{similarity_metric}_worst"
    diff_key = f"{similarity_metric}_score_diff"
    score_key = f"{similarity_metric}_score"

    # First load the dataset without any filtering
    dataset = load_dataset("json", data_files=file_path, split="train")
    total_count = len(dataset)

    # Filter the dataset if a threshold is provided
    if diff_threshold is not None:
        # Filter based on the pre-calculated score difference
        dataset = dataset.filter(lambda example: example.get(diff_key, 0) >= diff_threshold)
        kept_count = len(dataset)
        filtered_count = total_count - kept_count

        print(f"Total examples: {total_count}")
        print(f"Filtered using {similarity_metric} similarity:")
        print(f"\t- Filtered out {filtered_count} examples with score difference below {diff_threshold}")
        print(f"\t- Kept {kept_count} examples with score difference >= {diff_threshold}")

    def process_example(item):
        """
        Function to convert a single JSON example into the DPO format.
        """
        prompt = [
            {"role": "user", "content": item["decision_prompt"]},
            {
                "role": "assistant",
                "content": item["decision_output"],
            },
            {"role": "user", "content": item["explanation_prompt"]},
        ]

        dpo_entry = {
            "prompt": prompt,
            "chosen": [
                {
                    "role": "assistant",
                    "content": item[best_key]["explanation_output"],
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": item[worst_key]["explanation_output"],
                }
            ],
        }

        if include_scores:
            dpo_entry["score_chosen"] = item[best_key][score_key]
            dpo_entry["score_rejected"] = item[worst_key][score_key]
            # dpo_entry["score_diff"] = item.get(diff_key, dpo_entry["score_chosen"] - dpo_entry["score_rejected"])
            # dpo_entry["similarity_metric"] = similarity_metric

        return dpo_entry

    # Apply processing function
    return dataset.map(process_example, remove_columns=list(dataset.column_names) if dataset.column_names else [])


if __name__ == "__main__":
    # Test the dataset loading function
    dataset_path = "dpo_datasets/cleaned_ecqa_dpo_datasets/cleaned_ecqa_250221_181714_LIME/train.jsonl"
    train_dataset = load_dpo_dataset(dataset_path, include_scores=True)
    print(f"Number of samples: {len(train_dataset)}")
    print(train_dataset.column_names)
    print(train_dataset[0])  # Check the first sample
