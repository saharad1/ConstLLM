"""
Module to find the intersection of all existing test sets and create a consistent test set.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def load_test_set_scenario_ids(test_file: Path) -> Set[int]:
    """
    Load scenario IDs from a test set file.
    
    Args:
        test_file: Path to the test set JSONL file
        
    Returns:
        Set of scenario IDs in the test set
    """
    scenario_ids = set()
    
    print(f"Loading test set: {test_file}")
    
    with open(test_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    scenario_id = item.get("scenario_id")
                    if scenario_id is not None:
                        scenario_ids.add(scenario_id)
                    else:
                        print(f"Warning: Line {line_num} has no scenario_id")
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                    continue
    
    print(f"Found {len(scenario_ids)} scenario IDs in {test_file.name}")
    return scenario_ids


def find_test_set_intersection(test_files: List[Path]) -> Set[int]:
    """
    Find the intersection of scenario IDs across all test sets.
    
    Args:
        test_files: List of paths to test set files
        
    Returns:
        Set of scenario IDs that appear in ALL test sets
    """
    if not test_files:
        raise ValueError("No test files provided")
    
    print(f"Finding intersection of {len(test_files)} test sets...")
    
    # Load scenario IDs from each test set
    test_sets = []
    for test_file in test_files:
        if test_file.exists():
            scenario_ids = load_test_set_scenario_ids(test_file)
            test_sets.append(scenario_ids)
            print(f"  {test_file.name}: {len(scenario_ids)} scenarios")
        else:
            print(f"Warning: Test file not found: {test_file}")
    
    if not test_sets:
        raise ValueError("No valid test sets found")
    
    # Find intersection
    intersection = test_sets[0]
    for i, test_set in enumerate(test_sets[1:], 1):
        intersection = intersection.intersection(test_set)
        print(f"  After intersection with test set {i+1}: {len(intersection)} scenarios")
    
    return intersection


def create_consistent_test_set(
    test_files: List[Path],
    output_dir: Path,
    dataset_name: str = "unknown"
) -> None:
    """
    Create a consistent test set using the intersection of all provided test sets.
    
    Args:
        test_files: List of paths to test set files
        output_dir: Directory to save the consistent test set
        dataset_name: Name of the dataset for output file naming
    """
    # Find intersection
    intersection_scenario_ids = find_test_set_intersection(test_files)
    
    if not intersection_scenario_ids:
        print("Error: No common scenarios found across all test sets!")
        return
    
    print(f"\nFound {len(intersection_scenario_ids)} common scenarios across all test sets")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save intersection info
    intersection_info = {
        "dataset_name": dataset_name,
        "num_test_files": len(test_files),
        "test_files": [str(f) for f in test_files],
        "intersection_size": len(intersection_scenario_ids),
        "intersection_scenario_ids": sorted(list(intersection_scenario_ids))
    }
    
    info_file = output_dir / f"{dataset_name}_test_intersection_info.json"
    with open(info_file, "w") as f:
        json.dump(intersection_info, f, indent=2)
    
    print(f"Saved intersection info to: {info_file}")
    
    # Create consistent test set from the first test file (using intersection)
    if test_files and test_files[0].exists():
        first_test_file = test_files[0]
        consistent_test_data = []
        
        print(f"Creating consistent test set from: {first_test_file.name}")
        
        with open(first_test_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        scenario_id = item.get("scenario_id")
                        if scenario_id in intersection_scenario_ids:
                            consistent_test_data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        # Save consistent test set
        consistent_test_file = output_dir / f"{dataset_name}_consistent_test_{len(consistent_test_data)}.jsonl"
        with open(consistent_test_file, "w") as f:
            for item in consistent_test_data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Saved consistent test set to: {consistent_test_file}")
        print(f"Consistent test set contains {len(consistent_test_data)} scenarios")
    
    print(f"\nConsistent test set created successfully!")
    print(f"Use this test set for all future evaluations to ensure consistency.")


def find_test_files_in_directory(directory: Path, dataset_name: str = None) -> List[Path]:
    """
    Find all test files in a directory structure.
    
    Args:
        directory: Root directory to search
        dataset_name: Optional dataset name to filter results
        
    Returns:
        List of paths to test files
    """
    test_files = []
    
    print(f"Searching for test files in: {directory}")
    
    # Search for files matching test_*.jsonl pattern
    for test_file in directory.rglob("test_*.jsonl"):
        if dataset_name is None or dataset_name in str(test_file):
            test_files.append(test_file)
            print(f"  Found: {test_file}")
    
    return test_files


def main():
    parser = argparse.ArgumentParser(description="Find intersection of test sets and create consistent test set")
    parser.add_argument("--test_files", nargs="+", help="List of test file paths")
    parser.add_argument("--search_dir", type=str, help="Directory to search for test files")
    parser.add_argument("--dataset_name", type=str, help="Dataset name for filtering and output naming")
    parser.add_argument("--output_dir", type=str, default="data/consistent_test_sets", help="Output directory")
    
    args = parser.parse_args()
    
    test_files = []
    
    if args.test_files:
        # Use explicitly provided test files
        test_files = [Path(f) for f in args.test_files]
    elif args.search_dir:
        # Search for test files in directory
        search_dir = Path(args.search_dir)
        if not search_dir.exists():
            print(f"Error: Search directory not found: {search_dir}")
            return
        
        test_files = find_test_files_in_directory(search_dir, args.dataset_name)
    else:
        print("Error: Must provide either --test_files or --search_dir")
        return
    
    if not test_files:
        print("Error: No test files found!")
        return
    
    print(f"\nFound {len(test_files)} test files:")
    for f in test_files:
        print(f"  {f}")
    
    # Create consistent test set
    output_dir = Path(args.output_dir)
    dataset_name = args.dataset_name or "dataset"
    
    create_consistent_test_set(
        test_files=test_files,
        output_dir=output_dir,
        dataset_name=dataset_name
    )


if __name__ == "__main__":
    main() 