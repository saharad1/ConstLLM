import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader


def load_dpo_dataset(
    file_path,
    include_scores=False,
    diff_threshold: Optional[float] = None,
    similarity_metric: str = "spearman",  # Can be "spearman" or "cosine"
) -> Dataset:
    """
    Efficiently loads a JSONL dataset into a Hugging Face Dataset format for DPOTrainer.
    Uses direct JSON loading to handle inconsistent data types in the JSONL file.

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

    # Load the dataset manually using direct JSON parsing
    data_list = []
    total_count = 0
    kept_count = 0
    filtered_count = 0

    print(f"Loading dataset from {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    # Parse JSON line
                    item = json.loads(line.strip())
                    total_count += 1

                    # Check if the example meets the threshold criteria
                    if diff_threshold is not None:
                        score_diff = item.get(diff_key, 0)
                        if score_diff < diff_threshold:
                            filtered_count += 1
                            continue

                    # Create the example for DPO training
                    prompt = [
                        {"role": "user", "content": item["decision_prompt"]},
                        {
                            "role": "assistant",
                            "content": item["decision_output"],
                        },
                        {"role": "user", "content": item["explanation_prompt"]},
                    ]

                    example = {
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

                    # Include scores if requested
                    if include_scores:
                        example["score_chosen"] = item[best_key][score_key]
                        example["score_rejected"] = item[worst_key][score_key]

                    data_list.append(example)
                    kept_count += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_idx+1}: {str(e)}")
                except Exception as e:
                    print(f"Error processing line {line_idx+1}: {str(e)}")
    except Exception as e:
        print(f"Error opening file {file_path}: {str(e)}")

    # Create a Hugging Face Dataset from the loaded data
    dataset = Dataset.from_list(data_list)

    # Print statistics
    if diff_threshold is not None:
        print(f"Total examples: {total_count}")
        print(f"Filtered using {similarity_metric} similarity:")
        print(f"\t- Filtered out {filtered_count} examples with score difference below {diff_threshold}")
        print(f"\t- Kept {kept_count} examples with score difference >= {diff_threshold}")
    else:
        print(f"Loaded {kept_count} examples from {file_path}")

    return dataset


if __name__ == "__main__":
    # Test the dataset loading function
    dataset_path = "dpo_datasets/cleaned_ecqa_dpo_datasets/cleaned_ecqa_250221_181714_LIME/train_7617.jsonl"
    train_dataset = load_dpo_dataset(dataset_path, include_scores=True)
    print(f"Number of samples: {len(train_dataset)}")
    print(train_dataset.column_names)
    print(train_dataset[0])  # Check the first sample
