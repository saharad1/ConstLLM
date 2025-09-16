#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def extract_sample_data(sample):
    """Extract key information from a sample with median explanation"""
    # Handle both formats - collection_data format and eval_results format
    if "spearman_best" in sample:
        # collection_data format - need to find median from multiple explanations
        # This format should have explanation_outputs and spearman_scores
        if "explanation_outputs" in sample and "spearman_scores" in sample:
            explanations = sample["explanation_outputs"]
            spearman_scores = sample["spearman_scores"]

            # Find median explanation
            sorted_pairs = sorted(zip(spearman_scores, explanations))
            median_idx = len(sorted_pairs) // 2
            median_score, median_explanation = sorted_pairs[median_idx]

            return {
                "scenario": sample["decision_prompt"],
                "decision": sample["decision_output"],
                "correct_label": sample["correct_label"],
                "scenario_id": sample["scenario_id"],
                "explanation_median": median_explanation,
                "median_spearman_score": median_score,
            }
        else:
            # Fallback to best explanation if median not available
            return {
                "scenario": sample["decision_prompt"],
                "decision": sample["decision_output"],
                "correct_label": sample["correct_label"],
                "scenario_id": sample["scenario_id"],
                "explanation_median": sample["spearman_best"]["explanation_output"],
                "median_spearman_score": sample["spearman_best"].get("spearman_score", 0.0),
            }
    else:
        # eval_results format - pick median explanation from the list
        explanations = sample["explanation_outputs"]
        spearman_scores = sample["spearman_scores"]

        # Find median explanation
        sorted_pairs = sorted(zip(spearman_scores, explanations))
        median_idx = len(sorted_pairs) // 2
        median_score, median_explanation = sorted_pairs[median_idx]

        return {
            "scenario": sample["decision_prompt"],
            "decision": sample["decision_output"],
            "correct_label": sample["correct_label"],
            "scenario_id": sample["scenario_id"],
            "explanation_median": median_explanation,
            "median_spearman_score": median_score,
        }


def process_eval_file(file_path):
    """Process a single evaluation file and return all samples"""
    samples = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(extract_sample_data(sample))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {file_path}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return samples


def find_evaluation_files(base_dir):
    """Find all evaluation files and categorize by dataset and model"""
    results = defaultdict(list)

    base_path = Path(base_dir)

    # Handle both collection_data and eval_results directories
    for eval_file in base_path.rglob("*results.jsonl"):
        path_parts = eval_file.parts

        if "eval_results" in path_parts:
            # Handle eval_results format
            idx = path_parts.index("eval_results")
            if idx + 2 < len(path_parts):
                dataset = path_parts[idx + 1]  # e.g., arc_challenge

                # Extract model info - could be in different positions
                model_info = path_parts[idx + 2]  # e.g., huggingface, Llama-3.2-3B-Instruct, etc.

                if model_info == "huggingface" and idx + 3 < len(path_parts):
                    # Format: eval_results/dataset/huggingface/model_name/
                    model = path_parts[idx + 3]
                elif model_info in ["huggingface"]:
                    continue  # Skip this pattern for now
                else:
                    # Format: eval_results/dataset/model_name/ or eval_results/dataset/model_name/training_info/
                    model = model_info

                results[(dataset, model)].append(str(eval_file))

        elif "collection_data" in path_parts:
            # Handle collection_data format (original)
            idx = path_parts.index("collection_data")
            if idx + 1 < len(path_parts):
                dataset = path_parts[idx + 1]

                # Extract model name
                if idx + 2 < len(path_parts):
                    model_dir = path_parts[idx + 2]
                    # Extract model name from directory (e.g., "unsloth_Llama-3.2-3B-Instruct" -> "Llama-3.2-3B-Instruct")
                    if model_dir.startswith("unsloth_"):
                        model = model_dir[8:]  # Remove "unsloth_" prefix
                    elif model_dir.startswith("meta-llama_"):
                        model = model_dir[11:]  # Remove "meta-llama_" prefix
                    else:
                        model = model_dir

                    results[(dataset, model)].append(str(eval_file))

    return results


def extract_dataset_model_from_path(file_path):
    """Extract dataset, model, attribution method, and training type from file path"""
    path_parts = Path(file_path).parts
    filename = Path(file_path).name

    if "eval_results" in path_parts:
        # Handle eval_results format
        idx = path_parts.index("eval_results")
        if idx + 2 < len(path_parts):
            dataset = path_parts[idx + 1]  # e.g., arc_easy, ecqa

            # Extract model info - could be in different positions
            model_info = path_parts[idx + 2]  # e.g., huggingface, Llama-3.2-3B-Instruct, etc.

            if model_info == "huggingface" and idx + 3 < len(path_parts):
                # Format: eval_results/dataset/huggingface/model_name/
                model = path_parts[idx + 3]
                training_type = "huggingface"
            else:
                # Format: eval_results/dataset/model_name/ or eval_results/dataset/model_name/training_info/
                model = model_info
                training_type = "fine-tuned"

            # Extract attribution method from filename
            if "LIME" in filename:
                attribution_method = "LIME"
            elif "LIG" in filename:
                attribution_method = "LIG"
            else:
                attribution_method = "unknown"

            return dataset, model, attribution_method, training_type

    elif "collection_data" in path_parts:
        # Handle collection_data format
        idx = path_parts.index("collection_data")
        if idx + 1 < len(path_parts):
            dataset = path_parts[idx + 1]

            # Extract model name
            if idx + 2 < len(path_parts):
                model_dir = path_parts[idx + 2]
                # Extract model name from directory (e.g., "unsloth_Llama-3.2-3B-Instruct" -> "Llama-3.2-3B-Instruct")
                if model_dir.startswith("unsloth_"):
                    model = model_dir[8:]  # Remove "unsloth_" prefix
                elif model_dir.startswith("meta-llama_"):
                    model = model_dir[11:]  # Remove "meta-llama_" prefix
                else:
                    model = model_dir

                # Extract attribution method from filename
                if "LIME" in filename:
                    attribution_method = "LIME"
                elif "LIG" in filename:
                    attribution_method = "LIG"
                else:
                    attribution_method = "unknown"

                return dataset, model, attribution_method, "collection_data"

    # Fallback: try to extract from filename
    print(f"Warning: Could not extract dataset/model from path structure for {file_path}")
    return "unknown", "unknown", "unknown", "unknown"


def create_user_study_samples_from_datasets(
    target_datasets=None, target_files=None, num_samples=5, auto_discover=False
):
    """Create user study file from specific datasets or files"""

    if target_files:
        # Use specific files provided by user - treat each file individually
        eval_files = []
        for file_path in target_files:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            dataset, model, attribution_method, training_type = extract_dataset_model_from_path(file_path)
            eval_files.append((dataset, model, attribution_method, training_type, file_path))
    elif auto_discover:
        # Auto-discover files
        eval_files_collection = find_evaluation_files("/raid/saharad/ConstLLM/data/collection_data")
        eval_files_results = find_evaluation_files("/raid/saharad/ConstLLM/data/eval_results")

        # Merge the results
        eval_files = defaultdict(list)
        for key, files in eval_files_collection.items():
            eval_files[key].extend(files)
        for key, files in eval_files_results.items():
            eval_files[key].extend(files)
    else:
        # Filter by specified datasets
        eval_files_collection = find_evaluation_files("/raid/saharad/ConstLLM/data/collection_data")
        eval_files_results = find_evaluation_files("/raid/saharad/ConstLLM/data/eval_results")

        # Merge the results and filter by target datasets
        all_eval_files = defaultdict(list)
        for key, files in eval_files_collection.items():
            all_eval_files[key].extend(files)
        for key, files in eval_files_results.items():
            all_eval_files[key].extend(files)

        # Filter by target datasets
        eval_files = defaultdict(list)
        for (dataset, model), files in all_eval_files.items():
            if dataset in target_datasets:
                eval_files[(dataset, model)] = files

    if not eval_files:
        print("No evaluation files found!")
        return []

    if target_files:
        # For individual files, show each file
        print(f"Found {len(eval_files)} individual files:")
        for dataset, model, attribution_method, training_type, file_path in eval_files:
            print(f"  {dataset} | {model} | {attribution_method} | {training_type}: {file_path}")
    else:
        # For grouped files, show combinations
        print(f"Found {len(eval_files)} dataset/model combinations:")
        for (dataset, model), files in eval_files.items():
            print(f"  {dataset} with {model}: {len(files)} files")
            for f in files:
                print(f"    - {f}")

    user_study_samples = []

    # Track used scenario IDs GLOBALLY to ensure complete uniqueness across all datasets/models
    used_scenario_ids = set()

    if target_files:
        # Process each file individually
        for i, (dataset, model, attribution_method, training_type, file_path) in enumerate(eval_files, 1):
            print(
                f"  Processing file {i}/{len(eval_files)}: {dataset} | {model} | {attribution_method} | {training_type}"
            )
            print(f"    File: {file_path}")

            samples = process_eval_file(file_path)
            # Add file path to each sample
            for sample in samples:
                sample["file_path"] = file_path

            print(f"    Found {len(samples)} total samples")

            # Filter out already used scenario IDs (globally unique)
            available_samples = [sample for sample in samples if sample["scenario_id"] not in used_scenario_ids]

            print(f"    Found {len(available_samples)} globally unique scenario samples")

            if len(available_samples) >= num_samples:
                # Sort by scenario_id for consistent selection
                available_samples.sort(key=lambda x: x["scenario_id"])
                # Take first N unique samples
                selected_samples = available_samples[:num_samples]
            else:
                # Use all available unique samples if less than N
                selected_samples = available_samples

            # Mark these scenario IDs as used GLOBALLY
            for sample in selected_samples:
                used_scenario_ids.add(sample["scenario_id"])

            # Add metadata and store
            for j, sample in enumerate(selected_samples, 1):
                user_study_sample = {
                    "dataset": dataset,
                    "model": model,
                    "attribution_method": attribution_method,
                    "training_type": training_type,
                    "file_number": i,
                    "sample_number": j,
                    **sample,
                }
                user_study_samples.append(user_study_sample)

            print(f"    Selected {len(selected_samples)} unique samples")
            if selected_samples:
                scenario_ids = [s["scenario_id"] for s in selected_samples]
                print(f"    Scenario IDs: {scenario_ids}")
    else:
        # Process grouped files (original logic for datasets/auto-discover)
        # Collect all combinations and sort for consistent processing
        all_combinations = []
        for (dataset, model), files in eval_files.items():
            all_combinations.append((dataset, model, files))

        # Sort combinations for consistent ordering
        all_combinations.sort(key=lambda x: (x[0], x[1]))  # Sort by dataset, then model

        # Process each dataset/model combination
        for dataset, model, files in all_combinations:
            print(f"  Processing {dataset} with {model}...")

            all_samples = []
            for file_path in files:
                samples = process_eval_file(file_path)
                # Add file path to each sample
                for sample in samples:
                    sample["file_path"] = file_path
                all_samples.extend(samples)

            print(f"    Found {len(all_samples)} total samples")

            # Filter out already used scenario IDs (globally unique)
            available_samples = [sample for sample in all_samples if sample["scenario_id"] not in used_scenario_ids]

            print(f"    Found {len(available_samples)} globally unique scenario samples")

            if len(available_samples) >= num_samples:
                # Sort by scenario_id for consistent selection
                available_samples.sort(key=lambda x: x["scenario_id"])
                # Take first N unique samples
                selected_samples = available_samples[:num_samples]
            else:
                # Use all available unique samples if less than N
                selected_samples = available_samples

            # Mark these scenario IDs as used GLOBALLY
            for sample in selected_samples:
                used_scenario_ids.add(sample["scenario_id"])

            # Add metadata and store
            for i, sample in enumerate(selected_samples, 1):
                user_study_sample = {"dataset": dataset, "model": model, "sample_number": i, **sample}
                user_study_samples.append(user_study_sample)

            print(f"    Selected {len(selected_samples)} unique samples")
            if selected_samples:
                scenario_ids = [s["scenario_id"] for s in selected_samples]
                print(f"    Scenario IDs: {scenario_ids}")

    return user_study_samples


def save_user_study_file(samples, output_file):
    """Save samples to a formatted file for user study"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# User Study: Evaluation Dataset Samples\n\n")
        f.write("This file contains samples from each evaluation dataset, organized by dataset and model.\n")
        f.write("Each scenario ID appears only once globally to ensure no duplicates.\n")
        f.write("Explanations shown are the median-scoring explanations.\n")
        f.write(f"Total samples: {len(samples)}\n\n")

        # Group samples by dataset, model, attribution method, and training type
        grouped = defaultdict(list)
        for sample in samples:
            key = (
                sample["dataset"],
                sample["model"],
                sample.get("attribution_method", "unknown"),
                sample.get("training_type", "unknown"),
            )
            grouped[key].append(sample)

        f.write("## Table of Contents\n\n")
        for i, (dataset, model, attribution_method, training_type) in enumerate(sorted(grouped.keys()), 1):
            anchor = f"{dataset.lower()}-{model.lower().replace('.', '').replace('-', '')}-{attribution_method.lower()}-{training_type.lower().replace('-', '')}"
            f.write(f"{i}. [{dataset} - {model} - {attribution_method} - {training_type}](#{anchor})\n")
        f.write("\n")

        # Write detailed samples
        for (dataset, model, attribution_method, training_type), dataset_samples in sorted(grouped.items()):
            f.write(f"## {dataset} - {model} - {attribution_method} - {training_type}\n\n")

            for sample in dataset_samples:
                f.write(f"### Sample {sample['sample_number']}\n\n")
                f.write(f"**Scenario ID:** {sample['scenario_id']}\n\n")
                f.write("**Scenario:**\n")
                f.write(f"{sample['scenario']}\n\n")
                f.write("**Model Decision:**\n")
                f.write(f"{sample['decision']}\n\n")
                f.write(f"**Correct Label:** {sample['correct_label']}\n\n")
                f.write("**Median Explanation:**\n")
                f.write(f"{sample['explanation_median']}\n\n")
                f.write(f"**Median Spearman Score:** {sample['median_spearman_score']:.4f}\n\n")
                f.write("---\n\n")

    print(f"User study file saved to: {output_file}")


def save_user_study_csv(samples, csv_file):
    """Save samples to a CSV file for user study"""

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            [
                "ScenarioID",
                "AttributionMethod",
                "Dataset",
                "Model",
                "TrainingType",
                "Question",
                "ModelAnswer",
                "CorrectLabel",
                "Explanation",
                "SpearmanScore",
                "DatasetPath",
                "ModelPath",
            ]
        )

        # Write data rows
        for sample in samples:
            # Extract dataset and model paths from file path
            file_path = sample.get("file_path", "")
            dataset_path = f"/raid/saharad/ConstLLM/data/*/{sample['dataset']}"
            model_path = file_path if file_path else f"*/{sample['model']}/*"

            writer.writerow(
                [
                    sample["scenario_id"],
                    sample.get("attribution_method", "unknown"),
                    sample["dataset"],
                    sample["model"],
                    sample.get("training_type", "unknown"),
                    sample["scenario"].replace("\n", " ").replace("\r", " "),  # Clean newlines
                    sample["decision"].replace("\n", " ").replace("\r", " "),
                    sample["correct_label"],
                    sample["explanation_median"].replace("\n", " ").replace("\r", " "),
                    f"{sample['median_spearman_score']:.4f}",
                    dataset_path,
                    model_path,
                ]
            )

    print(f"User study CSV saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract samples for user study from evaluation datasets")
    parser.add_argument(
        "-d", "--datasets", type=str, help='Comma-separated list of dataset names (e.g., "arc_challenge,hellaswag")'
    )
    parser.add_argument("-f", "--files", type=str, help="Comma-separated list of specific jsonl file paths")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=5, help="Number of samples per dataset/model combination (default: 5)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/user_study_extraction/user_study_samples.md",
        help="Markdown output file path",
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        default="outputs/user_study_extraction/user_study_samples.csv",
        help="CSV output file path",
    )
    parser.add_argument(
        "-a", "--auto-discover", action="store_true", help="Auto-discover all evaluation files in standard locations"
    )

    args = parser.parse_args()

    # Parse datasets and files
    target_datasets = []
    if args.datasets:
        target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    target_files = []
    if args.files:
        target_files = [f.strip() for f in args.files.split(",") if f.strip()]

    # Validation
    if not target_datasets and not target_files and not args.auto_discover:
        print("Error: No datasets or files specified. Use --datasets, --files, or --auto-discover")
        parser.print_help()
        return 1

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_dir = os.path.dirname(args.csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # Display configuration
    print("==========================================")
    print("User Study Sample Extraction Configuration")
    print("==========================================")
    if args.auto_discover:
        print("Datasets: Auto-discovering all evaluation datasets")
    elif target_files:
        print(f"Files: {target_files}")
    else:
        print(f"Datasets: {target_datasets}")
    print(f"Samples per combination: {args.num_samples}")
    print(f"Markdown output file: {args.output}")
    print(f"CSV output file: {args.csv}")
    print("==========================================")

    # Extract samples
    print("Extracting samples for user study...")
    samples = create_user_study_samples_from_datasets(
        target_datasets=target_datasets,
        target_files=target_files,
        num_samples=args.num_samples,
        auto_discover=args.auto_discover,
    )

    if not samples:
        print("No samples extracted. Exiting.")
        return 1

    # Save to files
    save_user_study_file(samples, args.output)
    save_user_study_csv(samples, args.csv)

    # Print summary
    datasets = set(sample["dataset"] for sample in samples)
    models = set(sample["model"] for sample in samples)

    print(f"\nSummary:")
    print(f"Total samples: {len(samples)}")
    print(f"Datasets: {sorted(datasets)}")
    print(f"Models: {sorted(models)}")
    print(f"Markdown file: {args.output}")
    print(f"CSV file: {args.csv}")

    return 0


if __name__ == "__main__":
    exit(main())
