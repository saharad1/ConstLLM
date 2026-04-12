import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

from src.collect_data.comp_similarity_scores import (
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

                # # Add scenario_id if it exists
                # if "scenario_id" in item:
                cleaned_entry["scenario_id"] = item["scenario_id"]

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

                # Add all explanations and their details
                cleaned_entry["explanation_outputs"] = item["explanation_outputs"]

                # Add decision attributions for correlation calculations
                cleaned_entry["decision_attributions"] = decision_attributions

                # Add all explanation attributions
                cleaned_entry["explanation_attributions"] = item["explanation_attributions"]

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


def split_cleaned_jsonl(
    input_path: Path, output_dir: Path, train_ratio=0.7, eval_ratio=0.2, test_ratio=0.1, seed=42
) -> None:
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

    save_jsonl(train_data, f"{output_dir}/train_{len(train_data)}.jsonl")
    save_jsonl(eval_data, f"{output_dir}/eval_{len(eval_data)}.jsonl")
    save_jsonl(test_data, f"{output_dir}/test_{len(test_data)}.jsonl")

    print(f"Dataset split successfully: {len(train_data)} train, {len(eval_data)} eval, {len(test_data)} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and split a JSONL dataset")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: same as input file directory)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data for training (default: 0.7)")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="Ratio of data for evaluation (default: 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data for testing (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")

    args = parser.parse_args()

    # Convert input file to Path object
    input_path = Path(args.input_file)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use the same directory as the input file
        output_dir = input_path.parent

    # Create cleaned file path
    cleaned_file_path = output_dir / f"{input_path.stem}_cleaned.jsonl"

    print(f"Processing input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Cleaned file will be saved to: {cleaned_file_path}")

    # Clean the dataset
    print("\nCleaning dataset...")
    preprocess_jsonl(input_path, cleaned_file_path)

    # Split the cleaned dataset
    print("\nSplitting dataset...")
    split_cleaned_jsonl(
        cleaned_file_path,
        output_dir,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\nDone! Dataset has been cleaned and split into train/eval/test sets.")
