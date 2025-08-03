import argparse
import json
from pathlib import Path
from typing import Dict, List


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


def apply_split_indices(collected_data: List[Dict], split_info: Dict, output_dir: Path) -> None:
    """
    Apply split indices to collected data.
    """
    # Create a mapping from scenario_id to collected data
    scenario_id_to_data = {}

    for item in collected_data:
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
        "collected_data_size": len(collected_data),
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "test_size": len(test_data),
        "original_split_info": split_info,
    }

    with open(output_dir / "split_info.json", "w") as f:
        json.dump(collection_info, f, indent=2)

    print(f"Saved splits to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Apply split indices to collected data")
    parser.add_argument("input_file", type=str, help="Path to collected data JSONL file")
    parser.add_argument("split_file", type=str, help="Path to split indices JSON file")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: same as input file directory)")

    args = parser.parse_args()

    # Load split indices
    split_file_path = Path(args.split_file)
    if not split_file_path.exists():
        print(f"Error: Split file not found: {split_file_path}")
        return

    split_info = load_split_indices(split_file_path)

    # Load collected data
    input_path = Path(args.input_file)
    collected_data = load_collected_data(input_path)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent

    # Apply split indices
    apply_split_indices(collected_data=collected_data, split_info=split_info, output_dir=output_dir)

    print("\nDone! Collected data has been split using the provided indices.")
    print("This ensures consistent test sets across all data collections.")


if __name__ == "__main__":
    main()
