"""
Utilities for managing run environment and checkpoints in data collection.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def setup_run_environment(dataset_name, model_id, attribution_method_name, resume_run=None):
    """
    Set up the run environment including directories and filenames.

    Args:
        dataset_name: Name of the dataset being used
        attribution_method_name: Name of the attribution method
        resume_run: Optional name of a run to resume

    Returns:
        Tuple of (run_name, attribution_method_name, output_dir, jsonl_filename,
                 checkpoint_file, progress_file)
    """
    # Set up run name and directories
    if resume_run:
        run_name = resume_run
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract a simplified model name for the run name
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id

        # Create a shorter model identifier for the run name
        if "llama" in model_name.lower():
            # Extract version from names like "Meta-Llama-3.1-8B-Instruct"
            short_model = "llama"  # Default value
            if "-" in model_name:
                parts = model_name.split("-")
                for part in parts:
                    if "." in part and part[0].isdigit():  # Version number like 3.1
                        short_model += part
                        break
        else:
            # For other models, use first 10 chars of model name
            short_model = model_name.split("-")[0][:10]

        run_name = f"{dataset_name}_{current_time}_{attribution_method_name}_{short_model}"

    # Create output directory
    # Use the model ID with slashes replaced by underscores for the directory
    safe_model_id = model_id.replace("/", "_")
    output_dir = Path(f"data/collection_data/{dataset_name}/{safe_model_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create run directory
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up filenames
    jsonl_filename = run_dir / f"{run_name}.jsonl"
    checkpoint_file = run_dir / f"{run_name}_checkpoint.json"
    progress_file = run_dir / f"{run_name}_progress.json"

    return run_name, attribution_method_name, run_dir, jsonl_filename, checkpoint_file, progress_file


def load_checkpoints(checkpoint_file, progress_file, total_scenarios):
    """
    Load checkpoint and progress data from files.

    Args:
        checkpoint_file: Path to the checkpoint file
        progress_file: Path to the progress file
        total_scenarios: Total number of scenarios in the dataset

    Returns:
        Tuple of (processed_scenarios, progress_data)
    """
    # Initialize empty sets and dictionaries
    processed_scenarios = set()
    progress_data = {
        "start_time": datetime.now().timestamp(),
        "total_scenarios": total_scenarios,
        "processed_count": 0,
        "failed_scenarios": [],
    }

    # Load checkpoint file if it exists
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                processed_scenarios = set(checkpoint_data.get("processed_scenarios", []))
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")

    # Load progress file if it exists
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
                # Ensure failed_scenarios exists
                if "failed_scenarios" not in progress_data:
                    progress_data["failed_scenarios"] = []
        except Exception as e:
            print(f"Error loading progress file: {e}")

    # Update progress data
    progress_data["processed_count"] = len(processed_scenarios)

    return processed_scenarios, progress_data


def save_checkpoint(checkpoint_file, processed_scenarios):
    """
    Save checkpoint data to file.

    Args:
        checkpoint_file: Path to the checkpoint file
        processed_scenarios: Set of processed scenario IDs
    """
    checkpoint_data = {"processed_scenarios": list(processed_scenarios)}
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)


def save_progress(progress_file, progress_data):
    """
    Save progress data to file.

    Args:
        progress_file: Path to the progress file
        progress_data: Dictionary containing progress information
    """
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2, separators=(",", ": "))


def update_progress(progress_data, processed_scenarios, failed_scenarios=None):
    """
    Update progress data with current status.

    Args:
        progress_data: Dictionary containing progress information
        processed_scenarios: Set of processed scenario IDs
        failed_scenarios: List of failed scenario IDs

    Returns:
        Updated progress_data dictionary
    """
    current_time = datetime.now().timestamp()
    start_time = progress_data["start_time"]
    total_scenarios = progress_data["total_scenarios"]
    processed_count = len(processed_scenarios)

    # Calculate elapsed and estimated time
    elapsed_time = current_time - start_time
    if processed_count > 0:
        scenarios_per_second = processed_count / elapsed_time
        remaining_scenarios = total_scenarios - processed_count
        estimated_remaining_time = remaining_scenarios / scenarios_per_second if scenarios_per_second > 0 else 0
    else:
        estimated_remaining_time = 0

    # Update progress data
    progress_data.update(
        {
            "processed_count": processed_count,
            "elapsed_time": elapsed_time,
            "estimated_remaining_time": estimated_remaining_time,
            "completion_percentage": (processed_count / total_scenarios) * 100 if total_scenarios > 0 else 0,
        }
    )

    if failed_scenarios is not None:
        progress_data["failed_scenarios"] = failed_scenarios

    return progress_data
