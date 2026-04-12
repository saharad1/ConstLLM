import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.prepare_datasets.dataset_utils import load_original_dataset


def create_split_indices(
    dataset_name: str,
    subset: int = None,
    train_ratio: float = 0.7,
    eval_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = "data/dataset_splits",
) -> None:
    """
    Create split indices file based on the original dataset.

    This creates a JSON file with train/eval/test indices that can be used
    to consistently split any collected dataset.
    """
    print(f"Loading original dataset: {dataset_name}")
    dataset = load_original_dataset(dataset_name, subset)

    print(f"Original dataset size: {len(dataset)}")

    # Validate ratios
    if abs(train_ratio + eval_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, eval, and test ratios must sum to 1.0")

    # Create indices
    indices = list(range(len(dataset)))

    # Set seed for reproducibility
    random.seed(seed)
    random.shuffle(indices)

    # Calculate split sizes
    train_size = int(len(dataset) * train_ratio)
    eval_size = int(len(dataset) * eval_ratio)

    # Split indices
    train_indices = indices[:train_size]
    eval_indices = indices[train_size : train_size + eval_size]
    test_indices = indices[train_size + eval_size :]

    print(f"Split sizes - Train: {len(train_indices)}, Eval: {len(eval_indices)}, Test: {len(test_indices)}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create split indices file
    split_info = {
        "dataset_name": dataset_name,
        "original_dataset_size": len(dataset),
        "train_size": len(train_indices),
        "eval_size": len(eval_indices),
        "test_size": len(test_indices),
        "subset": subset,
        "train_ratio": train_ratio,
        "eval_ratio": eval_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "test_indices": test_indices,
    }

    # Save split indices file
    split_file = output_path / f"{dataset_name}_split_indices.json"
    with open(split_file, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved split indices to: {split_file}")
    print(f"Use this file to split any collected dataset for {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create split indices file from original dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--subset", type=int, default=None, help="Subset size")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training ratio")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="Evaluation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/dataset_splits", help="Output directory")

    args = parser.parse_args()

    create_split_indices(
        dataset_name=args.dataset_name,
        subset=args.subset,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )
