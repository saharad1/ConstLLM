"""
Script for evaluating trained DPO models by computing attributions on test datasets.
This script loads a trained model and runs the same evaluation process as in data collection.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

import wandb
from src.collect_data.attribution_config import get_attribution_methods_params
from src.collect_data.collection_metrics import calculate_metrics
from src.collect_data.dataset_loader import load_and_prepare_dataset
from src.collect_data.run_collection_utils import save_progress, update_progress
from src.collect_data.scenario_core_processor import process_scenario
from src.collect_data.scenario_runner import save_scenario_result
from src.llm_attribution.LLMAnalyzer import LLMAnalyzer


def eval_trained_dpo(
    model_path,
    dataset_path,
    attribution_method_name="LIME",
    num_dec_exp=5,
    subset=None,
    use_wandb=True,
    temperature=0.7,
    output_dir=None,
):
    """
    Evaluate a trained DPO model on a test dataset by computing attributions.

    Args:
        model_path: Path to the trained model
        dataset_path: Path to the dataset file (JSONL format)
        attribution_method_name: Name of the attribution method to use
        num_dec_exp: Number of decision explanations to generate
        subset: Optional subset size of the dataset to use
        use_wandb: Whether to use wandb for logging
        temperature: Temperature for model generation
        output_dir: Optional custom output directory
    """
    # Convert paths to Path objects if they're strings
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    # Set up signal handler for graceful termination
    def signal_handler(sig, frame):
        print("\nReceived termination signal. Saving progress and exiting...")
        save_progress(progress_file, progress_data)
        if use_wandb:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Extract model name and dataset name for run naming
    model_name = model_path.parent.name
    dataset_name = dataset_path.stem

    # Set up run environment
    if output_dir is None:
        base_output_dir = Path("data") / "eval_results" / model_name
    else:
        base_output_dir = Path(output_dir)

    # Create a unique run name
    timestamp = time.strftime("%y%m%d_%H%M%S")
    run_name = f"eval_{dataset_name}_{attribution_method_name}_{timestamp}"

    # Create output directories
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up output files
    jsonl_filename = output_dir / f"{run_name}_results.jsonl"
    progress_file = output_dir / "progress.json"

    # Set up wandb if enabled
    if use_wandb:
        wandb.init(
            project="constllm_eval_dpo",
            name=run_name,
            config={
                "model_path": str(model_path),
                "dataset_path": str(dataset_path),
                "attribution_method": attribution_method_name,
                "num_dec_exp": num_dec_exp,
                "temperature": temperature,
            },
        )

    # Load dataset from file
    print(f"Loading dataset from: {dataset_path}")
    try:
        # Load the dataset from the provided path
        dataset = []
        with open(dataset_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    dataset.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in dataset: {e}")
                    continue

        # Apply subset if specified
        if subset is not None and subset < len(dataset):
            dataset = dataset[:subset]

        print(f"Loaded {len(dataset)} scenarios")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if use_wandb:
            wandb.finish()
        return None

    # Initialize progress tracking
    processed_scenarios = set()
    progress_data = {
        "total_scenarios": len(dataset),
        "processed_scenarios": 0,
        "failed_scenarios": [],
        "success_rate": 0.0,
        "avg_spearman_best": 0.0,
        "avg_spearman_worst": 0.0,
        "avg_cosine_best": 0.0,
        "avg_cosine_worst": 0.0,
        "avg_spearman_median": 0.0,
        "avg_cosine_median": 0.0,
    }

    # Initialize tracking variables
    success_sum = 0
    spearman_sums = {"best": 0, "worst": 0, "median": 0}
    cosine_sums = {"best": 0, "worst": 0, "median": 0}

    # Set up attribution methods
    methods_params_decision, methods_params_explanation = get_attribution_methods_params(attribution_method_name)

    # Initialize LLM analyzer with the trained model
    print(f"Loading model from: {model_path}")
    llm_analyzer = LLMAnalyzer(model_id=str(model_path), temperature=temperature)

    # Store original parameters for reset after retries
    original_params = {
        "model_id": str(model_path),
        "decision": methods_params_decision.copy(),
        "explanation": methods_params_explanation.copy(),
    }

    # Process scenarios
    for iteration, scenario_item in enumerate(tqdm(dataset, desc="Processing scenarios"), 1):
        # Get scenario ID (ensure it exists)
        if hasattr(scenario_item, "scenario_id"):
            scenario_id = scenario_item.scenario_id
        elif isinstance(scenario_item, dict) and "scenario_id" in scenario_item:
            scenario_id = scenario_item["scenario_id"]
        else:
            scenario_id = f"scenario_{iteration}"

        if scenario_id in processed_scenarios:
            continue

        try:
            # Process the scenario directly using the core processor
            scenario_res = process_scenario(
                llm_analyzer=llm_analyzer,
                scenario_item=scenario_item,
                methods_params_decision=methods_params_decision,
                methods_params_explanation=methods_params_explanation,
                num_dec_exp=num_dec_exp,
            )

            # Ensure scenario_id is set
            if not hasattr(scenario_res, "scenario_id") or not scenario_res.scenario_id:
                scenario_res.scenario_id = scenario_id

            # Save the result
            save_scenario_result(scenario_res, jsonl_filename)

            # Mark as processed
            processed_scenarios.add(scenario_id)

            # Update progress
            failed_scenarios = progress_data.get("failed_scenarios", [])
            progress_data = update_progress(progress_data, processed_scenarios, failed_scenarios)
            save_progress(progress_file, progress_data)

            # Calculate metrics
            metrics, success_sum = calculate_metrics(
                scenario_res=scenario_res,
                success_sum=success_sum,
                iteration=iteration,
                spearman_sums=spearman_sums,
                cosine_sums=cosine_sums,
            )

            # Calculate median values and add to sums
            if len(scenario_res.spearman_scores) > 0:
                spearman_median = np.median(scenario_res.spearman_scores)
                cosine_median = np.median(scenario_res.cosine_scores)
                spearman_sums["median"] += spearman_median
                cosine_sums["median"] += cosine_median

                # Add median metrics to wandb logging
                metrics["spearman/median"] = spearman_median
                metrics["cosine/median"] = cosine_median

            # Log metrics
            if use_wandb:
                wandb.log(metrics)

        except Exception as e:
            print(f"Error processing scenario {scenario_id}: {str(e)}")
            # Log error to file
            error_file = output_dir / "errors.log"
            with open(error_file, "a") as f:
                f.write(f"Scenario {scenario_id} (iteration {iteration}): {str(e)}\n")

            # Add to failed scenarios in progress data
            if "failed_scenarios" not in progress_data:
                progress_data["failed_scenarios"] = []
            progress_data["failed_scenarios"].append(scenario_id)
            save_progress(progress_file, progress_data)
            continue

    # Finalize
    print(f"Evaluation completed. Processed {len(processed_scenarios)} scenarios.")

    # Update final averages for median values
    if len(processed_scenarios) > 0:
        progress_data["avg_spearman_median"] = spearman_sums["median"] / len(processed_scenarios)
        progress_data["avg_cosine_median"] = cosine_sums["median"] / len(processed_scenarios)

    save_progress(progress_file, progress_data)

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total scenarios: {len(dataset)}")
    print(f"Processed scenarios: {len(processed_scenarios)}")
    print(f"Failed scenarios: {len(progress_data.get('failed_scenarios', []))}")
    print(f"Success rate: {progress_data.get('success_rate', 0):.2f}%")
    print(f"Average Spearman (best): {progress_data.get('avg_spearman_best', 0):.4f}")
    print(f"Average Spearman (median): {progress_data.get('avg_spearman_median', 0):.4f}")
    print(f"Average Spearman (worst): {progress_data.get('avg_spearman_worst', 0):.4f}")
    print(f"Average Cosine (best): {progress_data.get('avg_cosine_best', 0):.4f}")
    print(f"Average Cosine (median): {progress_data.get('avg_cosine_median', 0):.4f}")
    print(f"Average Cosine (worst): {progress_data.get('avg_cosine_worst', 0):.4f}")

    if use_wandb:
        wandb.finish()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DPO model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file (JSONL format)")
    parser.add_argument("--attribution_method", type=str, default="LIME", help="Attribution method to use")
    parser.add_argument("--num_dec_exp", type=int, default=5, help="Number of explanations per decision")
    parser.add_argument("--subset", type=int, default=None, help="Size of dataset subset to use")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory for results")

    args = parser.parse_args()

    eval_trained_dpo(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        attribution_method_name=args.attribution_method,
        num_dec_exp=args.num_dec_exp,
        subset=args.subset,
        use_wandb=not args.no_wandb,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )
