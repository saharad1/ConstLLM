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


def preprocess_jsonl(input_path: Path, output_path: Path, include_scores=True) -> None:
    """
    Cleans the JSONL file to only keep fields required for DPO training.
    Always calculates and includes the Spearman score difference for flexible filtering later.
    """
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_data = []

    with open(input_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)

                # Ensure required fields exist
                if not all(
                    k in item
                    for k in [
                        "decision_prompt",
                        "explanation_best",
                        "explanation_worst",
                        "explanation_prompt",
                    ]
                ):
                    print(f"Skipping row due to missing fields: {item}")
                    continue

                # Construct cleaned entry
                cleaned_entry: dict[str, Any] = {
                    "decision_prompt": item["decision_prompt"],
                    "explanation_prompt": item["explanation_prompt"],
                    "explanation_best": item["explanation_best"],
                    "explanation_worst": item["explanation_worst"],
                }

                # Always include scores and calculate the difference for later filtering
                best_score = float(item["explanation_best"].get("spearman_score", 0.0))
                worst_score = float(item["explanation_worst"].get("spearman_score", 0.0))

                cleaned_entry["explanation_best"]["spearman_score"] = best_score
                cleaned_entry["explanation_worst"]["spearman_score"] = worst_score

                # Add the score difference for easy filtering later
                cleaned_entry["spearman_score_diff"] = best_score - worst_score

                cleaned_data.append(cleaned_entry)

            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON row: {e}")

    # Save cleaned dataset
    with open(output_path, "w") as f:
        for entry in cleaned_data:
            f.write(json.dumps(entry) + "\n")


def split_cleaned_jsonl(input_path: Path, output_dir: Path, train_ratio=0.7, eval_ratio=0.2, test_ratio=0.1, seed=42) -> None:
    """
    Splits a JSONL file into train/eval/test sets based on given ratios.
    Score differences are precalculated in the preprocessing step for
    later filtering at training time.

    Args:
        input_path: Path to the input JSONL file
        output_dir: Directory to save the split files
        train_ratio: Proportion of data to use for training
        eval_ratio: Proportion of data to use for evaluation
        test_ratio: Proportion of data to use for testing
        seed: Random seed for reproducibility
    """
    # Ensure the parent directory exists
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Load all lines
    data = []
    total_count = 0

    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                total_count += 1
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON: {e}")

    print(f"Total examples loaded: {len(data)}")

    # Shuffle data for randomness
    random.seed(seed)
    random.shuffle(data)

    # Split the data into train/eval/test
    total = len(data)
    train_size = int(total * train_ratio)
    eval_size = int(total * eval_ratio)

    # Create initial splits
    train_data = data[:train_size]
    eval_data = data[train_size : train_size + eval_size]
    test_data = data[train_size + eval_size :]

    # Save splits
    def save_jsonl(data, path) -> None:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_data, f"{output_dir}/train.jsonl")
    save_jsonl(eval_data, f"{output_dir}/eval.jsonl")
    save_jsonl(test_data, f"{output_dir}/test.jsonl")

    print(f"Dataset split successfully: {len(train_data)} train, {len(eval_data)} eval, {len(test_data)} test")


# Example usage
if __name__ == "__main__":
    # Example usage - Create clean dataset with preprocessed score differences
    raw_dpo_dataset_path = Path("dpo_datasets") / "codah_dpo_datasets" / "codah_250219_165846_LIME.jsonl"

    # Create output directory
    output_dir = Path("dpo_datasets") / "cleaned_codah_dpo_datasets" / "cleaned_codah_250219_165846_LIME"
    cleaned_dpo_dataset_path = output_dir / "cleaned_codah_250219_165846_LIME.jsonl"

    # Process with score differences always included for later filtering
    preprocess_jsonl(raw_dpo_dataset_path, cleaned_dpo_dataset_path, include_scores=True)

    # Split without applying any threshold - thresholds will be applied during training
    split_cleaned_jsonl(cleaned_dpo_dataset_path, output_dir)

    print("\nTo filter at training time, use:")
    print("python train_dpo_unsloth.py --threshold 0.3")
