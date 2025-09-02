#!/usr/bin/env python3
"""
Apply kernel SHAP selected indices to dataset test sets to create filtered datasets.

This script:
- Loads selected indices per dataset from a JSON file (produced by select_kernel_shap_indices)
- Searches for test_*.jsonl files under the collection_data root per dataset
- Matches the correct test file by comparing its number of lines to the recorded total_test_size
- Filters entries whose scenario_id is in the selected indices for that dataset
- Writes filtered JSONL files under an output root, mirroring the input directory structure

Defaults:
- indices_file: data/dataset_splits/kernel_shap_indices.json
- collection_root: data/collection_data
- output_root: data/kernel_shap_datasets

Usage example:
  python src/apply_kernel_shap_indices.py \
    --indices_file data/dataset_splits/kernel_shap_indices.json \
    --collection_root data/collection_data \
    --output_root data/kernel_shap_datasets
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_jsonl_count(path: Path) -> int:
    count = 0
    with open(path, "r") as f:
        for _ in f:
            count += 1
    return count


def load_indices(indices_file: Path) -> Dict:
    with open(indices_file, "r") as f:
        return json.load(f)


def find_test_files_for_dataset(collection_root: Path, dataset_name: str) -> List[Path]:
    dataset_dir = collection_root / dataset_name
    if not dataset_dir.exists():
        return []
    # Walk and collect test_*.jsonl
    test_files: List[Path] = []
    for root, _dirs, files in os.walk(dataset_dir):
        for fname in files:
            if fname.startswith("test_") and fname.endswith(".jsonl"):
                test_files.append(Path(root) / fname)
    # Sort for determinism
    test_files.sort()
    return test_files


def choose_test_file_by_size(test_files: List[Path], expected_size: int) -> Optional[Path]:
    for path in test_files:
        try:
            n = read_jsonl_count(path)
            if n == expected_size:
                return path
        except Exception:
            continue
    return None


def filter_jsonl_by_indices(input_path: Path, output_path: Path, selected_indices: set[int]) -> Tuple[int, int]:
    kept = 0
    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario_id = obj.get("scenario_id")
            if scenario_id in selected_indices:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    return kept, total


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply kernel SHAP indices to test sets to create filtered datasets")
    parser.add_argument(
        "--indices_file",
        type=str,
        default="data/dataset_splits/kernel_shap_indices.json",
        help="Path to selected indices JSON file",
    )
    parser.add_argument(
        "--collection_root", type=str, default="data/collection_data", help="Root directory for collected datasets"
    )
    parser.add_argument(
        "--output_root", type=str, default="data/kernel_shap_datasets", help="Root directory to write filtered datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["arc_easy", "arc_challenge", "ecqa", "codah"],
        help="Datasets to process",
    )

    args = parser.parse_args()

    indices_file = Path(args.indices_file)
    collection_root = Path(args.collection_root)
    output_root = Path(args.output_root)

    if not indices_file.exists():
        print(f"Error: indices_file not found: {indices_file}")
        return 1
    if not collection_root.exists():
        print(f"Error: collection_root not found: {collection_root}")
        return 1

    data = load_indices(indices_file)
    ds_info: Dict = data.get("datasets", {})

    manifest: Dict = {
        "indices_file": str(indices_file),
        "output_root": str(output_root),
        "datasets": {},
    }

    for dataset_name in args.datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        if dataset_name not in ds_info:
            print(f"  Warning: no indices found for {dataset_name} in {indices_file}")
            continue
        selected = ds_info[dataset_name].get("selected_indices", [])
        total_test_size = ds_info[dataset_name].get("total_test_size")
        if not isinstance(total_test_size, int):
            print(f"  Error: total_test_size missing for {dataset_name}")
            continue

        selected_set = set(int(x) for x in selected)
        print(f"  Selected indices: {len(selected_set)}; expected test size: {total_test_size}")

        # Find test files and choose the matching one by size
        test_files = find_test_files_for_dataset(collection_root, dataset_name)
        if not test_files:
            print(f"  Error: no test_*.jsonl files found under {collection_root / dataset_name}")
            continue

        chosen = choose_test_file_by_size(test_files, total_test_size)
        if chosen is None:
            print("  Error: could not find a test file whose line count matches total_test_size; skipping")
            continue

        print(f"  Chosen test file: {chosen}")

        # Build output path mirroring input dir structure under output_root
        rel = chosen.relative_to(collection_root)
        out_path = output_root / rel.parent / f"test_kernel_shap_{len(selected_set)}.jsonl"
        kept, total = filter_jsonl_by_indices(chosen, out_path, selected_set)
        print(f"  Wrote {kept} of {total} entries to {out_path}")

        manifest["datasets"][dataset_name] = {
            "input_test_file": str(chosen),
            "output_file": str(out_path),
            "kept": kept,
            "selected_requested": len(selected_set),
            "total_in_input_test": total,
        }

    # Save manifest
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
