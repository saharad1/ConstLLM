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
    file_path, include_scores=False, spearman_diff_threshold: Optional[float] = None
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """
    Efficiently loads a JSONL dataset into a Hugging Face Dataset format for DPOTrainer.
    Uses `load_dataset` for direct streaming and minimal memory usage.

    Args:
        file_path: Path to the JSONL file containing the dataset
        include_scores: Whether to include the scores in the dataset
        spearman_diff_threshold: Optional minimum threshold for difference between
                                best and worst Spearman scores. If provided, filters
                                out examples where the difference is below this threshold.

    Returns:
        A Hugging Face Dataset for DPO training
    """
    # First load the dataset without any filtering
    dataset = load_dataset("json", data_files=file_path, split="train")
    total_count = len(dataset)

    # Filter the dataset if a threshold is provided
    if spearman_diff_threshold is not None:
        # Filter based on the pre-calculated score difference
        dataset = dataset.filter(lambda example: example.get("spearman_score_diff", 0) >= spearman_diff_threshold)
        kept_count = len(dataset)
        filtered_count = total_count - kept_count

        print(f"Total examples: {total_count}")
        print(f"Filtered out {filtered_count} examples with score difference below {spearman_diff_threshold}")
        print(f"Kept {kept_count} examples with score difference >= {spearman_diff_threshold}")

    def process_example(item):
        """
        Function to convert a single JSON example into the DPO format.
        """
        prompt = [
            {"role": "user", "content": item["decision_prompt"]},
            {
                "role": "assistant",
                "content": item["explanation_best"]["decision_output"],
            },
            {"role": "user", "content": item["explanation_prompt"]},
        ]

        dpo_entry = {
            "prompt": prompt,
            "chosen": [
                {
                    "role": "assistant",
                    "content": item["explanation_best"]["explanation_output"],
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": item["explanation_worst"]["explanation_output"],
                }
            ],
        }

        if include_scores:
            dpo_entry["score_chosen"] = item["explanation_best"]["spearman_score"]
            dpo_entry["score_rejected"] = item["explanation_worst"]["spearman_score"]
            dpo_entry["score_diff"] = item.get("spearman_score_diff", dpo_entry["score_chosen"] - dpo_entry["score_rejected"])

        return dpo_entry

    # Apply processing function
    return dataset.map(process_example, remove_columns=list(dataset.column_names) if dataset.column_names else [])
