import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset


def load_dpo_dataset(file_path, include_scores=False) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """
    Efficiently loads a JSONL dataset into a Hugging Face Dataset format for DPOTrainer.
    Uses `load_dataset` for direct streaming and minimal memory usage.
    """

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

        return dpo_entry

    # Load dataset efficiently (streaming mode)
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = load_dataset("json", data_files=file_path, split="train")

    # Apply processing function lazily
    if isinstance(dataset, IterableDatasetDict):
        return dataset.map(process_example)
    else:
        return dataset.map(process_example, remove_columns=list(dataset.column_names) if dataset.column_names else [])


def preprocess_jsonl(input_path: Path, output_path: Path, include_scores=False) -> None:
    """
    Cleans the JSONL file to only keep fields required for DPO training.
    Removes unnecessary fields and ensures schema consistency.
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

                # Include scores if needed
                if include_scores:
                    cleaned_entry["explanation_best"]["spearman_score"] = float(item["explanation_best"].get("spearman_score", 0.0))
                    cleaned_entry["explanation_worst"]["spearman_score"] = float(item["explanation_worst"].get("spearman_score", 0.0))

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
    """
    # assert train_ratio + eval_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Ensure the parent directory exists
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Load all lines
    with open(input_path, "r") as f:
        data: list[Any] = [json.loads(line) for line in f]

    # # Shuffle data for randomness
    # random.seed(seed)
    # random.shuffle(data)

    # Compute split indices
    total = len(data)
    train_size = int(total * train_ratio)
    eval_size = int(total * eval_ratio)

    # Split dataset
    train_data: list[Any] = data[:train_size]
    eval_data: list[Any] = data[train_size : train_size + eval_size]
    test_data: list[Any] = data[train_size + eval_size :]

    # Save splits
    def save_jsonl(data, path) -> None:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_data, f"{output_dir}/train.jsonl")
    save_jsonl(eval_data, f"{output_dir}/eval.jsonl")
    save_jsonl(test_data, f"{output_dir}/test.jsonl")

    print(f"Dataset split successfully: {len(train_data)} train, {len(eval_data)} eval, {len(test_data)} test")


# def split_dataset_jsonl(input_jsonl, output_dir):
#     split_cleaned_jsonl(input_jsonl, output_dir)


# def create_cleaned_data(input_dpo_dataset_path, output_dpo_dataset, include_scores=False):
#     dpo_dataset_path = input_dpo_dataset_path
#     output_dpo_dataset = output_dpo_dataset
#     preprocess_jsonl(str(dpo_dataset_path), str(output_dpo_dataset), include_scores=include_scores)


# Example usage
if __name__ == "__main__":
    # raw_dpo_dataset_path = Path("dpo_datasets") / "codah_dpo_datasets" / "codah_250219_165846_LIME.jsonl"
    # cleaned_dpo_dataset_path = (
    #     Path("dpo_datasets") / "cleaned_codah_dpo_datasets" / "cleaned_codah_250219_165846_LIME" / "cleaned_codah_250219_165846_LIME.jsonl"
    # )
    # split_dpo_dataset = Path("dpo_datasets") / "cleaned_codah_dpo_datasets" / "cleaned_codah_250219_165846_LIME"

    raw_dpo_dataset_path: Path = Path("dpo_datasets") / "ecqa_dpo_datasets" / "ecqa_250221_181714_LIME.jsonl"

    cleaned_dpo_dataset_path: Path = (
        Path("dpo_datasets") / "cleaned_ecqa_dpo_datasets" / "cleaned_ecqa_250221_181714_LIME" / "cleaned_ecqa_250221_181714_LIME.jsonl"
    )
    split_dpo_dataset: Path = Path("dpo_datasets") / "cleaned_ecqa_dpo_datasets" / "cleaned_ecqa_250221_181714_LIME"

    # preprocess raw DPO dataset
    preprocess_jsonl(raw_dpo_dataset_path, cleaned_dpo_dataset_path, include_scores=False)

    # split cleaned dataset to train, eval, test
    split_cleaned_jsonl(cleaned_dpo_dataset_path, split_dpo_dataset, train_ratio=0.7, eval_ratio=0.2, test_ratio=0.1, seed=42)
