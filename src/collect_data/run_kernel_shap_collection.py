#!/usr/bin/env python3
"""
Script for collecting data specifically for kernel SHAP analysis using selected scenario IDs.

This script:
1. Loads the kernel SHAP indices for a specific dataset
2. Filters the original dataset to only include those specific scenario IDs
3. Runs the full data collection pipeline (decision + explanation + attribution) on just those scenarios
4. Saves results in a kernel SHAP specific directory structure
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import psutil
import torch
from tqdm import tqdm

import wandb
from src.collect_data.attribution_config import get_attribution_methods_params
from src.collect_data.collection_metrics import calculate_metrics
from src.collect_data.dataset_loader import load_and_prepare_dataset
from src.collect_data.run_collection_utils import (
    load_checkpoints,
    save_checkpoint,
    save_progress,
    update_progress,
)
from src.collect_data.scenario_runner import (
    process_single_scenario,
    save_scenario_result,
)
from src.llm_attribution.LLMAnalyzer import LLMAnalyzer

# Constants
MEMORY_CHECK_INTERVAL = 20
RELOAD_MODEL_EVERY = 50


def get_scenario_attribute(scenario_item, dict_key, obj_attr, default=""):
    """
    Helper function to get an attribute from either a dictionary or an object.

    Args:
        scenario_item: Either a dictionary or an object
        dict_key: The key to use if scenario_item is a dictionary
        obj_attr: The attribute name to use if scenario_item is an object
        default: Default value to return if attribute is not found

    Returns:
        The value from the scenario_item
    """
    if isinstance(scenario_item, dict):
        return scenario_item.get(dict_key, default)
    else:
        return getattr(scenario_item, obj_attr, default)


def load_kernel_shap_indices(indices_file: str, dataset_name: str) -> list:
    """Load the selected scenario IDs for kernel SHAP analysis."""
    with open(indices_file, "r") as f:
        data = json.load(f)

    if dataset_name not in data.get("datasets", {}):
        raise ValueError(f"Dataset {dataset_name} not found in indices file")

    selected_indices = data["datasets"][dataset_name].get("selected_indices", [])
    print(f"Loaded {len(selected_indices)} selected indices for {dataset_name}")

    return selected_indices


def filter_dataset_by_indices(dataset, selected_indices: list):
    """
    Filter the dataset to only include scenarios with the selected indices.

    Args:
        dataset: The prepared dataset object
        selected_indices: List of scenario IDs to include

    Returns:
        Filtered dataset containing only the selected scenarios
    """
    # Convert selected_indices to a set for faster lookup
    selected_set = set(selected_indices)

    # Create a list to store filtered scenarios
    filtered_scenarios = []

    print(f"Filtering dataset to include only {len(selected_indices)} selected scenarios...")

    for idx, scenario in enumerate(dataset):
        # Get the scenario ID - this might be the index itself or a specific field
        # For now, we'll use the index as the scenario ID since that's how the split indices work
        if idx in selected_set:
            filtered_scenarios.append(scenario)

    print(f"Filtered dataset contains {len(filtered_scenarios)} scenarios")
    return filtered_scenarios


def setup_kernel_shap_run_environment(dataset_name, model_id, attribution_method_name, resume_run=None):
    """
    Set up the run environment for kernel SHAP data collection.
    Similar to the original but with kernel SHAP specific naming.
    """
    # Set up run name and directories
    if resume_run:
        run_name = resume_run
    else:
        current_time = time.strftime("%Y%m%d_%H%M%S")
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

        run_name = f"{dataset_name}_kernel_shap_{current_time}_{attribution_method_name}_{short_model}"

    # Create output directory with kernel SHAP specific path
    safe_model_id = model_id.replace("/", "_")
    output_dir = Path(f"data/kernel_shap_collection/{dataset_name}/{safe_model_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create run directory
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up filenames
    jsonl_filename = run_dir / f"{run_name}_results.jsonl"
    checkpoint_file = run_dir / f"{run_name}_checkpoint.json"
    progress_file = run_dir / f"{run_name}_progress.json"

    return run_name, attribution_method_name, run_dir, jsonl_filename, checkpoint_file, progress_file


def run_kernel_shap_collection(
    model_id,
    dataset_name,
    indices_file,
    attribution_method_name="LIME",
    num_dec_exp=5,
    use_wandb=True,
    resume_run=None,
    temperature=0.7,
    base_seed=42,
):
    """
    Run the kernel SHAP data collection process for selected scenario IDs.

    Args:
        model_id: ID of the model to use
        dataset_name: Name of the dataset to use
        indices_file: Path to the kernel SHAP indices JSON file
        attribution_method_name: Name of the attribution method to use
        num_dec_exp: Number of decision explanations to generate
        use_wandb: Whether to use wandb for logging
        resume_run: Optional name of a run to resume
        temperature: Temperature for model generation (default: 0.7)
        base_seed: Base seed for reproducible experiments (default: 42)
    """

    # Set up signal handler for graceful termination
    def signal_handler(sig, frame):
        print("\nReceived termination signal. Saving checkpoint and exiting...")
        save_checkpoint(checkpoint_file, processed_scenarios)
        save_progress(progress_file, progress_data)
        if use_wandb:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load kernel SHAP indices
    print(f"Loading kernel SHAP indices from: {indices_file}")
    selected_indices = load_kernel_shap_indices(indices_file, dataset_name)

    # Set up run environment
    run_name, attribution_method_name, output_dir, jsonl_filename, checkpoint_file, progress_file = (
        setup_kernel_shap_run_environment(
            dataset_name=dataset_name,
            model_id=model_id,
            attribution_method_name=attribution_method_name,
            resume_run=resume_run,
        )
    )

    # Set up wandb if enabled
    if use_wandb:
        wandb.init(
            project="constllm_kernel_shap_collection",
            name=run_name,
            config={
                "model_id": model_id,
                "dataset": dataset_name,
                "attribution_method": attribution_method_name,
                "num_dec_exp": num_dec_exp,
                "temperature": temperature,
                "kernel_shap_indices_file": indices_file,
                "num_selected_scenarios": len(selected_indices),
            },
        )

    # Load full dataset first
    print(f"Loading full dataset: {dataset_name}")
    full_dataset = load_and_prepare_dataset(dataset_name)

    # Filter dataset to only include selected scenario IDs
    filtered_dataset = filter_dataset_by_indices(full_dataset, selected_indices)

    # Load checkpoints
    processed_scenarios, progress_data = load_checkpoints(
        checkpoint_file=checkpoint_file, progress_file=progress_file, total_scenarios=len(filtered_dataset)
    )

    # Initialize tracking variables
    last_model_reload = 0
    last_memory_check = 0
    success_sum = 0
    spearman_sums = {"best": 0, "worst": 0, "median": 0}
    cosine_sums = {"best": 0, "worst": 0, "median": 0}
    total_time_sum = 0

    # Set up attribution methods
    methods_params_decision, methods_params_explanation = get_attribution_methods_params(attribution_method_name)

    # Initialize LLM analyzer with the model
    print(f"Loading model: {model_id}")
    llm_analyzer = LLMAnalyzer(model_id=model_id, temperature=temperature)

    # Compute deterministic seeds from the base seed for each generation
    print(f"Using base seed: {base_seed}")
    # Create sequential seeds for each explanation generation
    generation_seeds = [base_seed + i for i in range(num_dec_exp)]
    print(f"Generated seeds for explanations: {generation_seeds}")

    # Store original parameters for reset after retries
    original_params = {
        "model_id": model_id,
        "decision": methods_params_decision.copy(),
        "explanation": methods_params_explanation.copy(),
    }

    # Process scenarios
    for iteration, scenario_item in enumerate(tqdm(filtered_dataset, desc="Processing scenarios"), 1):
        # Skip if already processed
        if iteration in processed_scenarios:
            print(f"Skipping already processed scenario {iteration}")
            continue

        # Memory and model management
        if iteration - last_memory_check >= MEMORY_CHECK_INTERVAL:
            memory_usage = psutil.virtual_memory().percent
            print(f"Memory usage: {memory_usage:.1f}%")
            last_memory_check = iteration

            if memory_usage > 90:
                print("High memory usage detected. Consider restarting the script.")

        if iteration - last_model_reload >= RELOAD_MODEL_EVERY:
            print("Reloading model to free memory...")
            del llm_analyzer
            torch.cuda.empty_cache()
            llm_analyzer = LLMAnalyzer(model_id=model_id, temperature=temperature)
            last_model_reload = iteration

        # Process the scenario
        start_time = time.time()
        try:
            scenario_result = process_single_scenario(
                scenario_item=scenario_item,
                llm_analyzer=llm_analyzer,
                methods_params_decision=methods_params_explanation,
                methods_params_explanation=methods_params_explanation,
                generation_seeds=generation_seeds,
                num_dec_exp=num_dec_exp,
                iteration=iteration,
            )

            if scenario_result:
                # Calculate metrics
                metrics = calculate_metrics(scenario_result)

                # Update sums
                success_sum += 1
                for metric_type in ["best", "worst", "median"]:
                    spearman_sums[metric_type] += metrics.get(f"spearman_{metric_type}", 0)
                    cosine_sums[metric_type] += metrics.get(f"cosine_{metric_type}", 0)

                # Save scenario result
                save_scenario_result(scenario_result, jsonl_filename)

                # Update progress
                processed_scenarios.add(iteration)
                progress_data = update_progress(
                    progress_data, iteration, len(filtered_dataset), metrics, time.time() - start_time
                )

                # Save checkpoint and progress
                save_checkpoint(checkpoint_file, processed_scenarios)
                save_progress(progress_file, progress_data)

                # Log to wandb if enabled
                if use_wandb:
                    wandb.log(
                        {
                            "iteration": iteration,
                            "success_rate": success_sum / iteration,
                            "spearman_best_avg": spearman_sums["best"] / success_sum,
                            "spearman_worst_avg": spearman_sums["worst"] / success_sum,
                            "spearman_median_avg": spearman_sums["median"] / success_sum,
                            "cosine_best_avg": cosine_sums["best"] / success_sum,
                            "cosine_worst_avg": cosine_sums["worst"] / success_sum,
                            "cosine_median_avg": cosine_sums["median"] / success_sum,
                        }
                    )

                total_time_sum += time.time() - start_time

        except Exception as e:
            print(f"Error processing scenario {iteration}: {e}")
            continue

    # Final summary
    print(f"\nKernel SHAP data collection completed!")
    print(f"Successfully processed: {success_sum}/{len(filtered_dataset)} scenarios")
    if success_sum > 0:
        print(
            f"Average Spearman - Best: {spearman_sums['best']/success_sum:.3f}, "
            f"Worst: {spearman_sums['worst']/success_sum:.3f}, "
            f"Median: {spearman_sums['median']/success_sum:.3f}"
        )
        print(
            f"Average Cosine - Best: {cosine_sums['best']/success_sum:.3f}, "
            f"Worst: {cosine_sums['worst']/success_sum:.3f}, "
            f"Median: {cosine_sums['median']/success_sum:.3f}"
        )
        print(f"Average processing time per scenario: {total_time_sum/success_sum:.2f}s")

    if use_wandb:
        wandb.finish()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect kernel SHAP data using selected scenario IDs")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--indices_file", type=str, required=True, help="Path to kernel SHAP indices JSON file")
    parser.add_argument("--attribution_method", type=str, default="LIME", help="Attribution method to use")
    parser.add_argument("--num_dec_exp", type=int, default=5, help="Number of explanations per decision")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume_run", type=str, help="Name of run to resume")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible experiments")

    args = parser.parse_args()

    run_kernel_shap_collection(
        model_id=args.model_id,
        dataset_name=args.dataset,
        indices_file=args.indices_file,
        attribution_method_name=args.attribution_method,
        num_dec_exp=args.num_dec_exp,
        use_wandb=not args.no_wandb,
        resume_run=args.resume_run,
        temperature=args.temperature,
        base_seed=args.seed,
    )
