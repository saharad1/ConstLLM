import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import Dataset, load_dataset


def load_dpo_dataset(file_path, include_scores=False):
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
    dataset = load_dataset("json", data_files=file_path, split="train")

    # Apply processing function lazily
    return dataset.map(process_example, remove_columns=dataset.column_names)


def preprocess_jsonl(input_path, output_path, include_scores=False):
    """
    Cleans the JSONL file to only keep fields required for DPO training.
    Removes unnecessary fields and ensures schema consistency.
    """
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
                cleaned_entry = {
                    "decision_prompt": item["decision_prompt"],
                    "explanation_prompt": item["explanation_prompt"],
                    "explanation_best": item["explanation_best"],
                    "explanation_worst": item["explanation_worst"],
                }

                # # Include scores if needed
                # if include_scores:
                #     cleaned_entry["explanation_best"]["spearman_score"] = float(
                #         item["explanation_best"].get("spearman_score", 0.0)
                #     )
                #     cleaned_entry["explanation_worst"]["spearman_score"] = float(
                #         item["explanation_worst"].get("spearman_score", 0.0)
                #     )

                cleaned_data.append(cleaned_entry)

            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON row: {e}")

    # Save cleaned dataset
    with open(output_path, "w") as f:
        for entry in cleaned_data:
            f.write(json.dumps(entry) + "\n")


def split_jsonl(
    input_path, output_dir, train_ratio=0.7, eval_ratio=0.2, test_ratio=0.1, seed=42
):
    """
    Splits a JSONL file into train/eval/test sets based on given ratios.
    """
    # assert train_ratio + eval_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load all lines
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]

    # # Shuffle data for randomness
    # random.seed(seed)
    # random.shuffle(data)

    # Compute split indices
    total = len(data)
    train_size = int(total * train_ratio)
    eval_size = int(total * eval_ratio)

    # Split dataset
    train_data = data[:train_size]
    eval_data = data[train_size : train_size + eval_size]
    test_data = data[train_size + eval_size :]

    # Save splits
    def save_jsonl(data, path):
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_data, f"{output_dir}/train.jsonl")
    save_jsonl(eval_data, f"{output_dir}/eval.jsonl")
    save_jsonl(test_data, f"{output_dir}/test.jsonl")

    print(
        f"Dataset split successfully: {len(train_data)} train, {len(eval_data)} eval, {len(test_data)} test"
    )


def split_dataset_jsol():
    input_jsonl = "/home/ahallak/saharad/ConstLLM/dpo_datasets/cleaned_codah_dpo_datasets/cleaned_codah_250213_182318_LLama.jsonl"
    output_dir = "/home/ahallak/saharad/ConstLLM/dpo_datasets/cleaned_codah_dpo_datasets/codah_250213_182318_LLama"  # Change to your directory
    split_jsonl(input_jsonl, output_dir)


def create_cleaned_data():
    dpo_dataset_path = (
        Path("dpo_datasets") / "codah_dpo_datasets" / "codah_250213_182318_LLama.jsonl"
    )

    output_dpo_dataset = (
        Path("dpo_datasets")
        / "cleaned_codah_dpo_datasets"
        / "cleaned_codah_250213_182318_LLama.jsonl"
    )
    preprocess_jsonl(
        str(dpo_dataset_path), str(output_dpo_dataset), include_scores=False
    )


# Example usage
if __name__ == "__main__":
    # dpo_dataset_path = (
    #     Path("dpo_datasets") / "codah_dpo_datasets" / "codah_250213_182318_LLama.jsonl"
    # )

    # output_dpo_dataset = (
    #     Path("dpo_datasets")
    #     / "cleaned_codah_dpo_datasets"
    #     / "cleaned_codah_250213_182318_LLama.jsonl"
    # )
    # preprocess_jsonl(
    #     str(dpo_dataset_path), str(output_dpo_dataset), include_scores=False
    # )
    # # Preprocess the JSONL file
    # train_dataset = load_dpo_dataset(str(output_dpo_dataset), include_scores=False)

    # # Check a sample
    # print(train_dataset[0])
    # create_cleaned_data()

    split_dataset_jsol()
