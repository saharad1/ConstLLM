import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from src.collect_data.comp_score import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
)


def preprocess_jsonl(input_path: Path, output_path: Path) -> None:
    """
    Cleans the JSONL file to only keep fields required for DPO training.
    Calculates both Spearman correlation and cosine similarity between decision and explanation attributions.
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
                        "decision_attributions",
                        "explanation_attributions",
                        "explanation_outputs",
                        "explanation_prompt",
                    ]
                ):
                    print(f"Skipping row due to missing fields: {item}")
                    continue

                # Construct cleaned entry
                cleaned_entry: Dict[str, Any] = {
                    "decision_prompt": item["decision_prompt"],
                    "explanation_prompt": item["explanation_prompt"],
                    "decision_output": item.get("decision_output", ""),
                    "correct_label": item.get("correct_label", ""),
                }

                # Compute both similarity metrics for each explanation
                decision_attributions = item["decision_attributions"]

                # Initialize variables to track best and worst explanations for both metrics
                spearman_best_score = -2.0  # Spearman ranges from -1 to 1
                spearman_worst_score = 2.0
                cosine_best_score = -1.0  # Cosine ranges from -1 to 1
                cosine_worst_score = 2.0

                spearman_best_explanation = None
                spearman_worst_explanation = None
                cosine_best_explanation = None
                cosine_worst_explanation = None

                explanation_details = []

                for i, explanation_attr in enumerate(item["explanation_attributions"]):
                    # Calculate Spearman correlation
                    spearman_score = calculate_spearman_correlation(decision_attributions, explanation_attr)

                    # Calculate cosine similarity
                    cosine_score = calculate_cosine_similarity(decision_attributions, explanation_attr)

                    # Create an explanation detail object
                    explanation_detail = {
                        "explanation_output": item["explanation_outputs"][i],
                        "spearman_score": spearman_score,
                        "cosine_score": cosine_score,
                        # "decision_output": item.get("decision_output", ""),
                    }

                    explanation_details.append(explanation_detail)

                    # Track best and worst by Spearman
                    if spearman_score > spearman_best_score:
                        spearman_best_score = spearman_score
                        spearman_best_explanation = explanation_detail

                    if spearman_score < spearman_worst_score:
                        spearman_worst_score = spearman_score
                        spearman_worst_explanation = explanation_detail

                    # Track best and worst by cosine
                    if cosine_score > cosine_best_score:
                        cosine_best_score = cosine_score
                        cosine_best_explanation = explanation_detail

                    if cosine_score < cosine_worst_score:
                        cosine_worst_score = cosine_score
                        cosine_worst_explanation = explanation_detail

                # Skip items with no valid explanations
                if (
                    spearman_best_explanation is None
                    or spearman_worst_explanation is None
                    or cosine_best_explanation is None
                    or cosine_worst_explanation is None
                ):
                    print(f"Skipping item with no valid explanations: {item}")
                    continue

                # # Add all explanations and their details
                # cleaned_entry["explanation_details"] = explanation_details

                # Add best and worst explanations by Spearman
                cleaned_entry["spearman_best"] = spearman_best_explanation
                cleaned_entry["spearman_worst"] = spearman_worst_explanation
                cleaned_entry["spearman_score_diff"] = spearman_best_score - spearman_worst_score

                # Add best and worst explanations by cosine
                cleaned_entry["cosine_best"] = cosine_best_explanation
                cleaned_entry["cosine_worst"] = cosine_worst_explanation
                cleaned_entry["cosine_score_diff"] = cosine_best_score - cosine_worst_score

                cleaned_data.append(cleaned_entry)

            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON row: {e}")
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

    # Save cleaned dataset
    with open(output_path, "w") as f:
        for entry in cleaned_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Processed {len(cleaned_data)} entries with both Spearman and cosine metrics")


def split_cleaned_jsonl(input_path: Path, output_dir: Path, train_ratio=0.7, eval_ratio=0.2, test_ratio=0.1, seed=42) -> None:
    """
    Splits a JSONL file into train/eval/test sets based on given ratios.
    Cosine similarities are precalculated in the preprocessing step for
    later filtering at training time.

    Args:
        input_path: Path to the input JSONL file
        output_dir: Directory to save the split files
        train_ratio: Proportion of data to use for training
        eval_ratio: Proportion of data to use for evaluation
        test_ratio: Proportion of data to use for testing
        seed: Random seed for reproducibility
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

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
    # Example usage - Create clean dataset with preprocessed cosine similarities
    raw_dpo_dataset_path = Path("dpo_datasets/ecqa_dpo_datasets/ecqa_250221_181714_LIME.jsonl")

    # Create output directory
    output_dir = Path("dpo_datasets/cleaned_ecqa_dpo_datasets/cleaned_ecqa_250221_181714_LIME")
    cleaned_dpo_dataset_path = output_dir / "cleaned_ecqa_250221_181714_LIME.jsonl"

    # Process with cosine similarities
    preprocess_jsonl(raw_dpo_dataset_path, cleaned_dpo_dataset_path)

    # Split without applying any threshold - thresholds will be applied during training
    split_cleaned_jsonl(cleaned_dpo_dataset_path, output_dir)

    print("Done Cleaning and Splitting the dataset.")
