#!/usr/bin/env python3
"""
Script to randomly select 300 indices from each dataset's test set for kernel SHAP analysis.

This script:
1. Loads the test indices from each dataset's split indices file
2. Randomly selects 300 indices from each test set
3. Creates a JSON file with the selected indices organized by dataset
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set


def load_test_indices(dataset_name: str, splits_dir: Path) -> List[int]:
    """Load test indices from a dataset's split indices file."""
    split_file = splits_dir / f"{dataset_name}_split_indices.json"

    if not split_file.exists():
        raise FileNotFoundError(f"Split indices file not found: {split_file}")

    with open(split_file, "r") as f:
        split_info = json.load(f)

    test_indices = split_info.get("test_indices", [])
    print(f"Loaded {len(test_indices)} test indices for {dataset_name}")

    return test_indices


def select_random_indices(indices: List[int], num_samples: int, seed: int) -> List[int]:
    """Randomly select a specified number of indices."""
    if len(indices) <= num_samples:
        print(f"Warning: Only {len(indices)} indices available, returning all")
        return indices

    random.seed(seed)
    selected = random.sample(indices, num_samples)
    selected.sort()  # Sort for reproducibility

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select random indices from dataset test sets for kernel SHAP analysis"
    )
    parser.add_argument(
        "--splits_dir", type=str, default="data/dataset_splits", help="Directory containing dataset split indices files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/kernel_shap_indices.json",
        help="Output JSON file path for selected indices",
    )
    parser.add_argument(
        "--num_samples", type=int, default=300, help="Number of indices to select from each test set (default: 300)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Convert to Path objects
    splits_dir = Path(args.splits_dir)
    output_file = Path(args.output_file)

    # Check if splits directory exists
    if not splits_dir.exists():
        print(f"Error: Splits directory does not exist: {splits_dir}")
        return 1

    # Define the datasets to process
    datasets = ["arc_easy", "arc_challenge", "ecqa", "codah"]

    # Dictionary to store selected indices for each dataset
    kernel_shap_indices = {
        "metadata": {
            "num_samples_per_dataset": args.num_samples,
            "seed": args.seed,
            "description": "Randomly selected test indices for kernel SHAP analysis",
        },
        "datasets": {},
    }

    print(f"Selecting {args.num_samples} indices from each dataset's test set...")
    print(f"Using seed: {args.seed}")
    print()

    # Process each dataset
    for dataset_name in datasets:
        try:
            print(f"Processing {dataset_name}...")

            # Load test indices
            test_indices = load_test_indices(dataset_name, splits_dir)

            # Select random indices
            selected_indices = select_random_indices(test_indices, args.num_samples, args.seed)

            # Store in results
            kernel_shap_indices["datasets"][dataset_name] = {
                "total_test_size": len(test_indices),
                "selected_indices": selected_indices,
                "selected_count": len(selected_indices),
            }

            print(f"  Selected {len(selected_indices)} indices from {len(test_indices)} total")

        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            continue

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_file, "w") as f:
        json.dump(kernel_shap_indices, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\nSummary:")
    for dataset_name, data in kernel_shap_indices["datasets"].items():
        print(f"  {dataset_name}: {data['selected_count']} indices selected from {data['total_test_size']} total")

    return 0


if __name__ == "__main__":
    exit(main())
