#!/usr/bin/env python3
"""
Script to deduplicate JSONL dataset files.
Removes duplicate entries based on specified key fields.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def create_dedup_key(entry: Dict[str, Any], key_fields: List[str]) -> str:
    """
    Create a unique key for deduplication based on specified fields.

    Args:
        entry: JSON object from the dataset
        key_fields: List of field names to use for deduplication

    Returns:
        String representation of the key fields for hashing
    """
    key_parts = []
    for field in key_fields:
        if field in entry:
            key_parts.append(str(entry[field]))
        else:
            key_parts.append("")  # Empty string for missing fields

    return "|".join(key_parts)


def deduplicate_jsonl(
    input_file: str, output_file: str, key_fields: List[str], verbose: bool = False
) -> Dict[str, int]:
    """
    Deduplicate a JSONL file based on specified key fields.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output deduplicated JSONL file
        key_fields: List of field names to use for deduplication
        verbose: Whether to print progress information

    Returns:
        Dictionary with statistics about the deduplication process
    """
    seen_keys: Set[str] = set()
    total_lines = 0
    unique_lines = 0
    duplicate_lines = 0

    if verbose:
        print(f"Reading from: {input_file}")
        print(f"Writing to: {output_file}")
        print(f"Deduplicating based on fields: {key_fields}")

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            total_lines += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue

            # Create deduplication key
            dedup_key = create_dedup_key(entry, key_fields)

            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                outfile.write(line + "\n")
                unique_lines += 1
            else:
                duplicate_lines += 1
                if verbose:
                    print(f"Duplicate found on line {line_num}: {dedup_key}")

    stats = {
        "total_lines": total_lines,
        "unique_lines": unique_lines,
        "duplicate_lines": duplicate_lines,
        "duplication_rate": duplicate_lines / total_lines if total_lines > 0 else 0,
    }

    if verbose:
        print(f"\nDeduplication complete!")
        print(f"Total lines processed: {stats['total_lines']}")
        print(f"Unique lines kept: {stats['unique_lines']}")
        print(f"Duplicate lines removed: {stats['duplicate_lines']}")
        print(f"Duplication rate: {stats['duplication_rate']:.2%}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Deduplicate JSONL dataset files")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("-o", "--output", help="Output file path (default: input_file_dedup.jsonl)")
    parser.add_argument(
        "-k",
        "--key-fields",
        nargs="+",
        default=["scenario_id", "decision_prompt", "decision_output"],
        help="Fields to use for deduplication (default: scenario_id decision_prompt decision_output)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--backup", action="store_true", help="Create backup of original file before deduplicating")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1

    # Set output file path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_dedup{input_path.suffix}"

    # Create backup if requested
    if args.backup:
        backup_path = input_path.parent / f"{input_path.stem}_backup{input_path.suffix}"
        if args.verbose:
            print(f"Creating backup: {backup_path}")
        import shutil

        shutil.copy2(input_path, backup_path)

    # Perform deduplication
    try:
        stats = deduplicate_jsonl(
            input_file=str(input_path), output_file=str(output_path), key_fields=args.key_fields, verbose=args.verbose
        )

        if args.verbose:
            print(f"\nOutput saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"Error during deduplication: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
