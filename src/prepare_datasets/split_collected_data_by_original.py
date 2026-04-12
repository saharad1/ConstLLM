import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from prepare_datasets.dataset_utils import load_original_dataset


def get_original_split_indices(
    dataset_name: str,
    subset: int = None,
    train_ratio: float = 0.7,
    eval_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Get split indices based on the original dataset.

    Returns:
        Tuple of (train_indices, eval_indices, test_indices)
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

    return train_indices, eval_indices, test_indices


def load_collected_data(input_file: Path) -> List[Dict]:
    """Load the collected data from JSONL file."""
    data = []
    print(f"Loading collected data from: {input_file}")

    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON: {e}")
                    continue

    print(f"Loaded {len(data)} collected examples")
    return data


def split_collected_data_by_original_indices(
    collected_data: List[Dict],
    train_indices: List[int],
    eval_indices: List[int],
    test_indices: List[int],
    output_dir: Path,
) -> None:
    """
    Split collected data based on original dataset indices.
    """
    # Create a mapping from scenario_id to collected data
    scenario_id_to_data = {}
    missing_scenarios = []

    for item in collected_data:
        scenario_id = item.get("scenario_id")
        if scenario_id is not None:
            scenario_id_to_data[scenario_id] = item
        else:
            print(f"Warning: Found item without scenario_id: {item}")

    print(f"Found {len(scenario_id_to_data)} items with scenario_id")

    # Split data based on original indices
    train_data = []
    eval_data = []
    test_data = []

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

    # Save split indices for reference
    splits_info = {
        "original_dataset_size": len(train_indices) + len(eval_indices) + len(test_indices),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "test_indices": test_indices,
        "collected_data_size": len(collected_data),
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "test_size": len(test_data),
    }

    with open(output_dir / "split_info.json", "w") as f:
        json.dump(splits_info, f, indent=2)

    print(f"Saved splits to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split collected data based on original dataset indices")
    parser.add_argument("input_file", type=str, help="Path to collected data JSONL file")
    parser.add_argument("dataset_name", type=str, help="Name of the original dataset")
    parser.add_argument("--subset", type=int, default=None, help="Subset size")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training ratio")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="Evaluation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: same as input file directory)")

    args = parser.parse_args()

    # Get original split indices
    train_indices, eval_indices, test_indices = get_original_split_indices(
        dataset_name=args.dataset_name,
        subset=args.subset,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Load collected data
    input_path = Path(args.input_file)
    collected_data = load_collected_data(input_path)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent

    # Split collected data based on original indices
    split_collected_data_by_original_indices(
        collected_data=collected_data,
        train_indices=train_indices,
        eval_indices=eval_indices,
        test_indices=test_indices,
        output_dir=output_dir,
    )

    print("\nDone! Collected data has been split based on original dataset indices.")
    print("This ensures consistent test sets across all data collections.")


if __name__ == "__main__":
    main()
