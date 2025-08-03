import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.collect_data.comp_similarity_scores import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
)


def load_split_indices(split_file: Path) -> Dict:
    """Load split indices from JSON file."""
    print(f"Loading split indices from: {split_file}")

    with open(split_file, "r") as f:
        split_info = json.load(f)

    print(f"Dataset: {split_info['dataset_name']}")
    print(f"Original size: {split_info['original_dataset_size']}")
    print(
        f"Split sizes - Train: {split_info['train_size']}, Eval: {split_info['eval_size']}, Test: {split_info['test_size']}"
    )

    return split_info


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

                # Add scenario_id if it exists
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
    return cleaned_data


def apply_split_indices_to_cleaned_data(cleaned_data: List[Dict], split_info: Dict, output_dir: Path) -> None:
    """
    Apply split indices to cleaned data.
    """
    # Create a mapping from scenario_id to cleaned data
    scenario_id_to_data = {}

    for item in cleaned_data:
        scenario_id = item.get("scenario_id")
        if scenario_id is not None:
            scenario_id_to_data[scenario_id] = item
        else:
            print(f"Warning: Found item without scenario_id: {item}")

    print(f"Found {len(scenario_id_to_data)} items with scenario_id")

    # Get split indices
    train_indices = split_info["train_indices"]
    eval_indices = split_info["eval_indices"]
    test_indices = split_info["test_indices"]

    # Helper function to get data for indices
    def get_data_for_indices(indices: List[int], split_name: str) -> List[Dict]:
        data = []
        missing = []
        for idx in indices:
            if idx in scenario_id_to_data:
                data.append(scenario_id_to_data[idx])
            else:
                missing.append(idx)

        if missing:
            print(f"Warning: {len(missing)} scenarios missing from {split_name} split: {missing[:10]}...")

        return data

    train_data = get_data_for_indices(train_indices, "train")
    eval_data = get_data_for_indices(eval_indices, "eval")
    test_data = get_data_for_indices(test_indices, "test")

    print(f"Split results:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval: {len(eval_data)} examples")
    print(f"  Test: {len(test_data)} examples")

    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(data: List[Dict], path: Path) -> None:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_data, output_dir / f"train_{len(train_data)}.jsonl")
    save_jsonl(eval_data, output_dir / f"eval_{len(eval_data)}.jsonl")
    save_jsonl(test_data, output_dir / f"test_{len(test_data)}.jsonl")

    # Save split info for this specific collection
    collection_info = {
        "split_file_used": str(split_info.get("dataset_name", "unknown")),
        "collected_data_size": len(cleaned_data),
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "test_size": len(test_data),
        "original_split_info": split_info,
    }

    with open(output_dir / "split_info.json", "w") as f:
        json.dump(collection_info, f, indent=2)

    print(f"Saved splits to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Clean collected data and apply split indices")
    parser.add_argument("input_file", type=str, help="Path to collected data JSONL file")
    parser.add_argument("split_file", type=str, help="Path to split indices JSON file")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: same as input file directory)")
    parser.add_argument("--keep_cleaned", action="store_true", help="Keep the intermediate cleaned file")

    args = parser.parse_args()

    # Load split indices
    split_file_path = Path(args.split_file)
    if not split_file_path.exists():
        print(f"Error: Split file not found: {split_file_path}")
        return

    split_info = load_split_indices(split_file_path)

    # Determine output directory
    input_path = Path(args.input_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent

    # Create cleaned file path
    cleaned_file_path = output_dir / f"{input_path.stem}_cleaned.jsonl"

    print(f"Processing input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Cleaned file will be saved to: {cleaned_file_path}")

    # Clean the dataset
    print("\nCleaning dataset...")
    cleaned_data = preprocess_jsonl(input_path, cleaned_file_path)

    # Apply split indices to cleaned data
    print("\nApplying split indices...")
    apply_split_indices_to_cleaned_data(
        cleaned_data=cleaned_data,
        split_info=split_info,
        output_dir=output_dir,
    )

    # Remove intermediate cleaned file if not requested to keep
    if not args.keep_cleaned:
        cleaned_file_path.unlink()
        print(f"Removed intermediate cleaned file: {cleaned_file_path}")

    print("\nDone! Collected data has been cleaned and split using the provided indices.")
    print("This ensures consistent test sets across all data collections.")


if __name__ == "__main__":
    main()
