# src/collect_data/run_collect_data_new.py
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.collect_data.attribution_config import configure_attribution_methods
from src.collect_data.dataset_loader import load_and_prepare_dataset
from src.collect_data.process_scenario import process_scenario
from src.collect_data.system_utils import (
    clear_memory,
    get_memory_usage,
    setup_signal_handlers,
)
from src.llm_attribution.LLMAnalyzer import LLMAnalyzer
from src.llm_attribution.utils_attribution import AttributionMethod
from src.utils.general import print_gpu_info

# Set up logging and GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
MEMORY_CHECK_INTERVAL = 20
RELOAD_MODEL_EVERY = 50


def extract_choice(output: str):
    """
    Extracts the choice (e.g., 'A', 'B') from the model's output.

    Args:
        output: The raw model output string.

    Returns:
        The extracted choice.
    """
    output = output.strip().upper()

    # First check for standalone letters at the beginning
    if output and output[0] in "ABCDE" and (len(output) == 1 or output[1] in [".", ")", " ", "\n"]):
        return output[0]

    # Look for patterns like "A.", "A)", "A ", "Answer: A"
    for pattern in ["A", "B", "C", "D", "E"]:
        if pattern in output:
            for i in range(len(output) - len(pattern) + 1):
                if output[i : i + len(pattern)] == pattern:
                    if i + len(pattern) == len(output) or output[i + len(pattern)] in [".", ")", " ", "\n"]:
                        return pattern

    # If no clear choice is found, return the first letter as a fallback
    if output:
        return output[0]
    return ""


def setup_run_environment(dataset_name, attribution_method_name, resume_run=None):
    """
    Set up the run environment, including run name and output directories.

    Args:
        dataset_name: Name of the dataset
        attribution_method_name: Name of the attribution method
        resume_run: Name of run to resume, if any

    Returns:
        Tuple of (run_name, output_dir, jsonl_filename, checkpoint_file, progress_file)
    """
    # Generate run name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if resume_run:
        run_name = resume_run
        logger.info(f"Resuming run: {run_name}")
        # Extract attribution method from run name
        try:
            extracted_method = run_name.split("_")[-1]  # Get last part after underscore
            logger.info(f"Using attribution method from run name: {extracted_method}")
            attribution_method_name = extracted_method
        except:
            logger.warning(f"Could not extract attribution method from run name, using default: {attribution_method_name}")
    else:
        run_name = f"{dataset_name}_{timestamp}_{attribution_method_name}"
        logger.info(f"Starting new run: {run_name}")

    # Set up output directories and files
    output_dir = Path(f"collected_data/{dataset_name}/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_filename = output_dir / "results.jsonl"
    checkpoint_file = output_dir / "checkpoint.txt"
    progress_file = output_dir / "progress.json"

    return run_name, attribution_method_name, output_dir, jsonl_filename, checkpoint_file, progress_file


def load_checkpoints(checkpoint_file, progress_file, total_scenarios):
    """
    Load checkpoints and progress data if available.

    Args:
        checkpoint_file: Path to checkpoint file
        progress_file: Path to progress file
        total_scenarios: Total number of scenarios

    Returns:
        Tuple of (processed_scenarios, progress_data)
    """
    # Initialize tracking variables
    processed_scenarios = set()

    # Load checkpoint if resuming
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            processed_scenarios = set(line.strip() for line in f)
        logger.info(f"Loaded {len(processed_scenarios)} processed scenarios from checkpoint")

    # Initialize progress tracking
    progress_data = {
        "total_scenarios": total_scenarios,
        "completed_scenarios": len(processed_scenarios),
        "failed_scenarios": [],
        "total_processing_time": 0,
        "avg_scenario_time": 0,
        "estimated_completion_time": None,
    }

    # Load progress data if resuming
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                progress_data.update(json.load(f))
            logger.info(f"Loaded progress data: {progress_data}")
        except:
            logger.warning("Could not load progress data, using defaults")

    return processed_scenarios, progress_data


def process_single_scenario(
    scenario_item, llm_analyzer, methods_params_decision, methods_params_explanation, num_dec_exp, original_params, iteration
):
    """
    Process a single scenario with retries.

    Args:
        scenario_item: The scenario to process
        llm_analyzer: The LLM analyzer to use
        methods_params_decision: Parameters for decision attribution
        methods_params_explanation: Parameters for explanation attribution
        num_dec_exp: Number of explanations to generate
        original_params: Original parameters for reset after retries
        iteration: Current iteration number

    Returns:
        Tuple of (scenario_result, error_message) where error_message is None if successful
    """
    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            logger.info(f"\n=== Running Scenario {iteration} (Attempt {retry_count + 1}/{MAX_RETRIES}) ===")

            # Clear memory
            clear_memory()

            # Process scenario
            scenario_result = process_scenario(
                llm_analyzer=llm_analyzer,
                scenario_item=scenario_item,
                methods_params_decision=methods_params_decision,
                methods_params_explanation=methods_params_explanation,
                num_dec_exp=num_dec_exp,
                custom_logger=logger,
            )

            return scenario_result, None  # Success

        except Exception as e:
            retry_count += 1
            error_msg = f"Error processing scenario {iteration} (Attempt {retry_count}/{MAX_RETRIES}): {str(e)}"
            logger.error(error_msg, exc_info=True)

            if retry_count < MAX_RETRIES:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
                # Reset parameters to original values
                methods_params_decision.clear()
                methods_params_decision.update(original_params["decision"])
                methods_params_explanation.clear()
                methods_params_explanation.update(original_params["explanation"])
            else:
                return None, error_msg  # Failed after all retries


def calculate_metrics(scenario_res, success_sum, iteration, spearman_sums, cosine_sums):
    """
    Calculate metrics for a scenario result.

    Args:
        scenario_res: The scenario result
        success_sum: Running sum of successful decisions
        iteration: Current iteration number
        spearman_sums: Dictionary with running sums for Spearman metrics
        cosine_sums: Dictionary with running sums for cosine metrics

    Returns:
        Dictionary of calculated metrics
    """
    # Compute metrics
    correct_choice = extract_choice(scenario_res.correct_label)
    decision_choice = extract_choice(scenario_res.decision_output)
    success_sum += decision_choice == correct_choice
    accuracy_label = success_sum / iteration

    # Spearman correlations
    spearman_best_score = np.max(scenario_res.spearman_scores)
    spearman_worst_score = np.min(scenario_res.spearman_scores)
    spearman_sums["best"] += spearman_best_score
    spearman_best_score_avg = spearman_sums["best"] / iteration
    spearman_sums["worst"] += spearman_worst_score
    spearman_worst_score_avg = spearman_sums["worst"] / iteration

    # Cosine similarities
    cosine_best_score = np.max(scenario_res.cosine_scores)
    cosine_worst_score = np.min(scenario_res.cosine_scores)
    cosine_sums["best"] += cosine_best_score
    cosine_best_score_avg = cosine_sums["best"] / iteration
    cosine_sums["worst"] += cosine_worst_score
    cosine_worst_score_avg = cosine_sums["worst"] / iteration

    # Prepare metrics dictionary
    metrics = {
        # General metrics
        "accuracy": accuracy_label,
        "spearman_best_score_avg": spearman_best_score_avg,
        "spearman_worst_score_avg": spearman_worst_score_avg,
        "cosine_best_score_avg": cosine_best_score_avg,
        "cosine_worst_score_avg": cosine_worst_score_avg,
        # Scenario metrics
        "scenario/scenario_id": scenario_res.scenario_id,
        # Spearman metrics
        "scenario/spearman/mean": np.mean(scenario_res.spearman_scores),
        "scenario/spearman/std": np.std(scenario_res.spearman_scores, ddof=1),
        "scenario/spearman/best_score": spearman_best_score,
        "scenario/spearman/worst_score": spearman_worst_score,
        # Cosine metrics
        "scenario/cosine/mean": np.mean(scenario_res.cosine_scores),
        "scenario/cosine/std": np.std(scenario_res.cosine_scores, ddof=1),
        "scenario/cosine/best_score": cosine_best_score,
        "scenario/cosine/worst_score": cosine_worst_score,
    }

    return metrics, success_sum


def run_collect_data(
    model_id,
    dataset_name="ecqa",
    attribution_method_name="LIME",
    num_dec_exp=5,
    temperature=0.7,
    subset=None,
    wandb_mode=True,
    resume_run=None,
):
    """
    Main function to collect data using the specified model and parameters.

    Args:
        model_id: ID of the model to use
        dataset_name: Name of the dataset to use
        attribution_method_name: Name of the attribution method to use
        num_dec_exp: Number of explanations to generate per decision
        temperature: Temperature for model generation
        subset: Size of dataset subset to use
        wandb_mode: Whether to use wandb for logging
        resume_run: Name of run to resume, if any
    """
    # Initialize variables
    running = True
    termination_reason = None
    device = "auto"

    # Set up signal handlers
    def handle_termination(reason):
        nonlocal running, termination_reason
        termination_reason = reason
        running = False

    setup_signal_handlers(handle_termination)

    # Set up run environment
    run_name, attribution_method_name, output_dir, jsonl_filename, checkpoint_file, progress_file = setup_run_environment(
        dataset_name, attribution_method_name, resume_run
    )

    # Set up wandb if enabled
    if wandb_mode:
        wandb.init(
            project="constllm-data-collection",
            name=run_name,
            config={
                "model_id": model_id,
                "dataset": dataset_name,
                "attribution_method": attribution_method_name,
                "num_dec_exp": num_dec_exp,
                "temperature": temperature,
                "subset": subset,
            },
            resume="allow",
        )

    # Load dataset
    dataset = load_and_prepare_dataset(dataset_name, subset)

    # Initialize model
    print_gpu_info()
    llm_analyzer = LLMAnalyzer(model_id=model_id, temperature=temperature, device=device)

    # Configure attribution methods
    methods_params_decision = configure_attribution_methods(attribution_method_name, "decision")
    methods_params_explanation = configure_attribution_methods(attribution_method_name, "explanation")

    # Store original parameters for reset after retries
    original_params = {
        "decision": methods_params_decision.copy(),
        "explanation": methods_params_explanation.copy(),
    }

    # Load checkpoints
    processed_scenarios, progress_data = load_checkpoints(checkpoint_file, progress_file, len(dataset))

    # Initialize tracking variables
    last_model_reload = 0
    success_sum = 0
    spearman_sums = {"best": 0, "worst": 0}
    cosine_sums = {"best": 0, "worst": 0}

    # Process scenarios
    for iteration, scenario_item in tqdm(
        enumerate(dataset, 1),
        total=len(dataset),
        desc="Processing Scenarios",
    ):
        if not running:
            logger.warning(f"Graceful shutdown initiated. Reason: {termination_reason}")
            logger.warning(f"Saving progress at scenario {iteration}/{len(dataset)}")
            break

        # Skip processed scenarios
        scenario_id = getattr(scenario_item, "scenario_id", None)
        if isinstance(scenario_item, dict):
            scenario_id = scenario_item.get("scenario_id", None)

        if scenario_id and str(scenario_id) in processed_scenarios:
            logger.info(f"Skipping processed scenario {scenario_id}")
            continue

        # Check if model needs reloading
        if iteration - last_model_reload >= RELOAD_MODEL_EVERY:
            logger.info("Performing periodic model reload...")
            del llm_analyzer
            torch.cuda.empty_cache()
            clear_memory()
            llm_analyzer = LLMAnalyzer(model_id=model_id, temperature=temperature, device=device)
            last_model_reload = iteration

        # Monitor resources
        if iteration % MEMORY_CHECK_INTERVAL == 0:
            memory_stats = get_memory_usage()
            logger.info(f"Memory usage: RAM {memory_stats['ram_used_gb']:.2f} GB, GPU {memory_stats['gpu_memory_used']:.2f} GB")

        start_time = time.time()

        # Process the scenario
        scenario_res, error_msg = process_single_scenario(
            scenario_item=scenario_item,
            llm_analyzer=llm_analyzer,
            methods_params_decision=methods_params_decision,
            methods_params_explanation=methods_params_explanation,
            num_dec_exp=num_dec_exp,
            original_params=original_params,
            iteration=iteration,
        )

        if scenario_res:  # Successful processing
            # Save result
            if wandb_mode:
                with open(jsonl_filename, "a") as f:
                    f.write(json.dumps(asdict(scenario_res)) + "\n")

                # Update checkpoint
                with open(checkpoint_file, "a") as f:
                    f.write(f"{scenario_res.scenario_id}\n")
                processed_scenarios.add(str(scenario_res.scenario_id))

            # Update progress tracking
            scenario_time = time.time() - start_time
            progress_data["total_processing_time"] += scenario_time
            progress_data["completed_scenarios"] = len(processed_scenarios)

            if progress_data["completed_scenarios"] > 0:
                progress_data["avg_scenario_time"] = progress_data["total_processing_time"] / progress_data["completed_scenarios"]

                # Estimate completion time
                remaining_scenarios = progress_data["total_scenarios"] - progress_data["completed_scenarios"]
                estimated_remaining_time = remaining_scenarios * progress_data["avg_scenario_time"]
                progress_data["estimated_completion_time"] = (datetime.now() + timedelta(seconds=estimated_remaining_time)).isoformat()

            # Save progress
            with open(progress_file, "w") as f:
                json.dump(progress_data, f)

            # Calculate metrics
            metrics, success_sum = calculate_metrics(
                scenario_res=scenario_res,
                success_sum=success_sum,
                iteration=iteration,
                spearman_sums=spearman_sums,
                cosine_sums=cosine_sums,
            )

            # Add timing metrics
            metrics["iteration"] = iteration
            metrics["scenario/iteration_time_seconds"] = time.time() - start_time

            # Log metrics
            if wandb_mode:
                wandb.log(metrics)

        else:  # Failed processing
            # Log final failure
            error_file = output_dir / "errors.log"
            with open(error_file, "a") as f:
                f.write(f"Scenario {scenario_id}: {error_msg}\n")
            progress_data["failed_scenarios"].append(str(scenario_id))
            with open(progress_file, "w") as f:
                json.dump(progress_data, f)

    logger.info(f"Data collection complete. Results saved to {jsonl_filename}")

    if wandb_mode:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect data using a specified model")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use")
    parser.add_argument("--dataset", type=str, default="ecqa", help="Dataset to use")
    parser.add_argument("--attribution_method", type=str, default="LIME", help="Attribution method to use")
    parser.add_argument("--num_explanations", type=int, default=5, help="Number of explanations per decision")
    parser.add_argument("--subset", type=int, default=None, help="Size of dataset subset to use")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume_run", type=str, help="Name of run to resume")

    args = parser.parse_args()

    run_collect_data(
        model_id=args.model_id,
        dataset_name=args.dataset,
        attribution_method_name=args.attribution_method,
        num_dec_exp=args.num_explanations,
        subset=args.subset,
        wandb_mode=not args.no_wandb,
        resume_run=args.resume_run,
    )
