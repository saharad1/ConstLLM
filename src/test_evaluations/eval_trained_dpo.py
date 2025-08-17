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
    is_model_id=False,
    base_seed=42,
    ignore_pre_generated=False,
    device_map=None,
    resume_run=None,
):
    """
    Evaluate a trained DPO model on a test dataset by computing attributions.

    Args:
        model_path: Path to the trained model or a model ID (from HuggingFace)
        dataset_path: Path to the dataset file (JSONL format)
        attribution_method_name: Name of the attribution method to use
        num_dec_exp: Number of decision explanations to generate
        subset: Optional subset size of the dataset to use
        use_wandb: Whether to use wandb for logging
        temperature: Temperature for model generation
        output_dir: Optional custom output directory
        is_model_id: If True, model_path is treated as a HuggingFace model ID instead of a local path
        base_seed: Base seed for reproducible experiments, default is 42
        ignore_pre_generated: If True, ignore any pre-generated attributions in the dataset
        resume_run: Optional name of a run to resume from
    """
    # Convert paths to Path objects if they're strings and not a model ID
    if not is_model_id:
        if isinstance(model_path, str):
            model_path = Path(model_path)

    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    # Set up signal handler for graceful termination
    def signal_handler(sig, frame):
        print("\nReceived termination signal. Saving progress and exiting...")
        # Add processed scenarios to progress data before saving
        progress_data["processed_scenarios"] = list(processed_scenarios)
        save_progress(progress_file, progress_data)
        if use_wandb:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Extract model name and dataset name for run naming
    if is_model_id:
        # For model IDs, use the model ID itself (or the last part if it contains slashes)
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        grandparent_name = "huggingface"
    else:
        # For local paths, use the directory structure
        model_name = model_path.parent.name
        grandparent_name = model_path.parent.parent.name

    # Extract dataset name (e.g., ecqa, codah) from the dataset path
    # First try to extract from the path structure
    dataset_path_str = str(dataset_path)
    if "ecqa" in dataset_path_str.lower():
        dataset_type = "ecqa"
    elif "codah" in dataset_path_str.lower():
        dataset_type = "codah"
    elif "arc_easy" in dataset_path_str.lower():
        dataset_type = "arc_easy"
    elif "arc_challenge" in dataset_path_str.lower():
        dataset_type = "arc_challenge"
    else:
        # Fallback to using the stem of the filename
        dataset_type = dataset_path.stem.split("_")[0]

    # Get the full dataset filename for reference
    dataset_name = dataset_path.stem

    # Set up run environment
    if output_dir is None:
        # Use clean dataset type (ecqa/codah) and model name for directory structure
        base_output_dir = Path("data") / "eval_results" / dataset_type / grandparent_name / model_name
    else:
        base_output_dir = Path(output_dir)

    print(f"Output directory: {base_output_dir}")

    # Create a unique run name
    if resume_run:
        run_name = resume_run
        print(f"Resuming run: {run_name}")
    else:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        pregen_signal = "no_pregen" if ignore_pre_generated else "with_pregen"
        run_name = f"eval_{timestamp}_{dataset_name}_{attribution_method_name}_{pregen_signal}"

    # Create output directories
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up output files
    jsonl_filename = output_dir / f"{run_name}_results.jsonl"
    progress_file = output_dir / "progress.json"

    # Initialize wandb if enabled
    if use_wandb:
        print("Initializing WandB...")
        try:
            # Initialize wandb
            wandb.init(
                project="constllm_eval_dpo",
                name=run_name,
                config={
                    "model_path": str(model_path),
                    "dataset_path": str(dataset_path),
                    "attribution_method": attribution_method_name,
                    "num_dec_exp": num_dec_exp,
                    "temperature": temperature,
                    "ignore_pre_generated": ignore_pre_generated,
                },
            )
            print("WandB initialized successfully!")
        except Exception as e:
            print(f"Error initializing WandB: {str(e)}")
            print("Continuing without WandB logging...")
            use_wandb = False

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
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "total_scenarios": len(dataset),
        "processed_count": 0,
        "failed_scenarios": [],
        "success_rate": 0.0,
        "avg_spearman_best": 0.0,
        "avg_spearman_worst": 0.0,
        "avg_cosine_best": 0.0,
        "avg_cosine_worst": 0.0,
        "avg_spearman_median": 0.0,
        "avg_cosine_median": 0.0,
        "start_time": time.time(),
        "elapsed_time": 0.0,
        "estimated_remaining_time": 0.0,
        "completion_percentage": 0.0,
    }

    # Load existing progress if resuming
    if resume_run and progress_file.exists():
        print(f"Loading existing progress from {progress_file}")
        try:
            with open(progress_file, "r") as f:
                existing_progress = json.load(f)
                processed_scenarios = set(existing_progress.get("processed_scenarios", []))
                progress_data.update(existing_progress)
                print(f"Loaded {len(processed_scenarios)} already processed scenarios")
        except Exception as e:
            print(f"Error loading progress file: {e}")
            print("Starting fresh...")

    # Initialize tracking variables
    success_sum = 0
    spearman_sums = {"best": 0, "worst": 0, "median": 0}
    cosine_sums = {"best": 0, "worst": 0, "median": 0}

    # Set up attribution methods
    methods_params_decision, methods_params_explanation = get_attribution_methods_params(attribution_method_name)

    # Initialize LLM analyzer with the model
    if is_model_id:
        print(f"Loading model from HuggingFace with ID: {model_path}")
    else:
        print(f"Loading model from local path: {model_path}")

    # Convert model_path to string to ensure compatibility with LLMAnalyzer
    model_path_str = str(model_path)
    llm_analyzer = LLMAnalyzer(model_id=model_path_str, device_map=device_map, temperature=temperature)

    # Compute deterministic seeds from the base seed for each generation
    print(f"Using base seed: {base_seed}")
    # Create sequential seeds for each explanation generation
    generation_seeds = [base_seed + i for i in range(num_dec_exp)]
    print(f"Generated seeds for explanations: {generation_seeds}")

    # Process scenarios
    start_time_total = time.time()
    total_time_sum = 0
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
            # Track time for this specific scenario (individual timing)
            start_time_scenario = time.time()

            # Get pre-generated outputs and attributions if we're not ignoring them
            pre_generated_decision_output = None
            pre_generated_decision_attributions = None
            if not ignore_pre_generated:
                if isinstance(scenario_item, dict):
                    pre_generated_decision_output = scenario_item.get("decision_output")
                    pre_generated_decision_attributions = scenario_item.get("decision_attributions")
                else:
                    pre_generated_decision_output = getattr(scenario_item, "decision_output", None)
                    pre_generated_decision_attributions = getattr(scenario_item, "decision_attributions", None)

            # Process the scenario directly using the core processor
            scenario_res = process_scenario(
                llm_analyzer=llm_analyzer,
                scenario_item=scenario_item,
                methods_params_decision=methods_params_decision,
                methods_params_explanation=methods_params_explanation,
                num_dec_exp=num_dec_exp,
                generation_seeds=generation_seeds,
                pre_generated_decision_output=pre_generated_decision_output,
                pre_generated_decision_attributions=pre_generated_decision_attributions,
            )

            # Ensure scenario_id is set
            if not hasattr(scenario_res, "scenario_id") or not scenario_res.scenario_id:
                scenario_res.scenario_id = scenario_id

            # Save the result
            save_scenario_result(scenario_res, jsonl_filename)

            # Mark as processed
            processed_scenarios.add(scenario_id)

            # Calculate and add timing information
            end_time_scenario = time.time()
            scenario_duration = end_time_scenario - start_time_scenario
            total_time_sum += scenario_duration  # Update total time sum

            # Calculate metrics, now including timing arguments
            metrics, success_sum = calculate_metrics(
                scenario_res=scenario_res,
                success_sum=success_sum,
                iteration=iteration,
                spearman_sums=spearman_sums,
                cosine_sums=cosine_sums,
                scenario_time=scenario_duration,  # Pass scenario duration
                total_time_sum=total_time_sum,  # Pass total time sum
            )

            # Add timing metrics (per-scenario timing for detailed analysis)
            metrics["scenario/duration_seconds"] = scenario_duration

            # Update progress data with the latest metrics
            progress_data.update(
                {
                    "success_rate": (success_sum / iteration) * 100,
                    "avg_spearman_best": spearman_sums["best"] / iteration,
                    "avg_spearman_median": spearman_sums["median"] / iteration,
                    "avg_spearman_worst": spearman_sums["worst"] / iteration,
                    "avg_cosine_best": cosine_sums["best"] / iteration,
                    "avg_cosine_median": cosine_sums["median"] / iteration,
                    "avg_cosine_worst": cosine_sums["worst"] / iteration,
                }
            )

            # Update progress tracking information
            failed_scenarios = progress_data.get("failed_scenarios", [])
            progress_data = update_progress(progress_data, processed_scenarios, failed_scenarios)

            # Add processed scenarios to progress data for resuming
            progress_data["processed_scenarios"] = list(processed_scenarios)

            # Save the updated progress data
            save_progress(progress_file, progress_data)

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

            # Add processed scenarios to progress data for resuming
            progress_data["processed_scenarios"] = list(processed_scenarios)
            save_progress(progress_file, progress_data)
            continue

    # Finalize
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    print(f"Evaluation completed. Processed {len(processed_scenarios)} scenarios in {total_duration:.2f} seconds.")
    print(f"Average time per scenario: {total_duration / len(processed_scenarios):.2f} seconds.")

    # Add processed scenarios to progress data for final save
    progress_data["processed_scenarios"] = list(processed_scenarios)
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

    # Log summary metrics to wandb
    if use_wandb:
        try:
            summary_metrics = {
                "summary/total_scenarios": len(dataset),
                "summary/processed_scenarios": len(processed_scenarios),
                "summary/failed_scenarios": len(progress_data.get("failed_scenarios", [])),
                "summary/success_rate": progress_data.get("success_rate", 0),
                "summary/avg_spearman_best": progress_data.get("avg_spearman_best", 0),
                "summary/avg_spearman_median": progress_data.get("avg_spearman_median", 0),
                "summary/avg_spearman_worst": progress_data.get("avg_spearman_worst", 0),
                "summary/avg_cosine_best": progress_data.get("avg_cosine_best", 0),
                "summary/avg_cosine_median": progress_data.get("avg_cosine_median", 0),
                "summary/avg_cosine_worst": progress_data.get("avg_cosine_worst", 0),
                "summary/total_duration_seconds": total_duration,
                "summary/avg_scenario_duration_seconds": (
                    total_duration / len(processed_scenarios) if processed_scenarios else 0
                ),
            }
            # print("Logging summary metrics to WandB...")
            # wandb.log(summary_metrics)
            # print("Successfully logged summary metrics to WandB")

            # Also set these as wandb summary values (appear at the top of the run page)
            for key, value in summary_metrics.items():
                wandb.run.summary[key] = value
            print("Successfully set WandB summary values")
        except Exception as e:
            print(f"Error logging summary metrics to WandB: {str(e)}")

    if use_wandb:
        try:
            print("Finishing WandB run...")
            wandb.finish()
            print("WandB run finished successfully")
        except Exception as e:
            print(f"Error finishing WandB run: {str(e)}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DPO model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model directory or HuggingFace model ID"
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file (JSONL format)")
    parser.add_argument("--attribution_method", type=str, default="LIME", help="Attribution method to use")
    parser.add_argument("--num_dec_exp", type=int, default=5, help="Number of explanations per decision")
    parser.add_argument("--subset", type=int, default=None, help="Size of dataset subset to use")
    parser.add_argument("--wandb", type=str, default="false", help="Enable/disable wandb logging (true/false)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory for results")
    parser.add_argument(
        "--is_model_id", action="store_true", help="Treat model_path as a HuggingFace model ID instead of a local path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible experiments")
    parser.add_argument(
        "--ignore_pre_generated", action="store_true", help="Ignore any pre-generated attributions in the dataset"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device mapping for multi-GPU (auto, balanced, sequential, or specific mapping)",
    )
    parser.add_argument(
        "--resume_run",
        type=str,
        default=None,
        help="Name of a previous run to resume from",
    )

    args = parser.parse_args()

    # Convert wandb string to boolean
    use_wandb = args.wandb.lower() == "true"

    eval_trained_dpo(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        attribution_method_name=args.attribution_method,
        num_dec_exp=args.num_dec_exp,
        subset=args.subset,
        use_wandb=use_wandb,
        temperature=args.temperature,
        output_dir=args.output_dir,
        is_model_id=args.is_model_id,
        base_seed=args.seed,
        ignore_pre_generated=args.ignore_pre_generated,
        device_map=args.device_map,
        resume_run=args.resume_run,
    )
