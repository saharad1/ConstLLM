"""
Script for collecting data by running models on scenarios and computing attributions.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from datasets import load_dataset
from tqdm import tqdm

import wandb
from src.collect_data.attribution_config import get_attribution_methods_params
from src.collect_data.collection_metrics import calculate_metrics, extract_choice
from src.collect_data.dataset_loader import load_and_prepare_dataset
from src.collect_data.run_collection_utils import (
    load_checkpoints,
    save_checkpoint,
    save_progress,
    setup_run_environment,
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


def run_collect_data(
    model_id,
    dataset_name,
    attribution_method_name="LIME",
    num_dec_exp=5,
    subset=None,
    use_wandb=True,
    resume_run=None,
    temperature=0.7,
    base_seed=42,
):
    """
    Run the data collection process for a given model and dataset.

    Args:
        model_id: ID of the model to use
        dataset_name: Name of the dataset to use
        attribution_method_name: Name of the attribution method to use
        num_dec_exp: Number of decision explanations to generate
        subset: Optional subset size of the dataset to use
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

    # Set up run environment
    run_name, attribution_method_name, output_dir, jsonl_filename, checkpoint_file, progress_file = (
        setup_run_environment(
            dataset_name=dataset_name,
            model_id=model_id,
            attribution_method_name=attribution_method_name,
            resume_run=resume_run,
        )
    )

    # Set up wandb if enabled
    if use_wandb:
        wandb.init(
            project="constllm_collect_data",
            name=run_name,
            config={
                "model_id": model_id,
                "dataset": dataset_name,
                "attribution_method": attribution_method_name,
                "num_dec_exp": num_dec_exp,
                "temperature": temperature,
            },
        )

    # Load dataset
    dataset = load_and_prepare_dataset(dataset_name, subset)

    # Load checkpoints
    processed_scenarios, progress_data = load_checkpoints(
        checkpoint_file=checkpoint_file, progress_file=progress_file, total_scenarios=len(dataset)
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
    for iteration, scenario_item in enumerate(tqdm(dataset, desc="Processing scenarios"), 1):
        scenario_start_time = time.time()
        # Skip already processed scenarios
        scenario_id = get_scenario_attribute(scenario_item, "scenario_id", "scenario_id", f"scenario_{iteration}")
        if scenario_id in processed_scenarios:
            continue

        # Check memory usage periodically
        if iteration - last_memory_check >= MEMORY_CHECK_INTERVAL:
            last_memory_check = iteration
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            if memory_percent > 90:
                print(f"High memory usage detected ({memory_percent}%). Saving checkpoint and exiting...")
                save_checkpoint(checkpoint_file, processed_scenarios)
                save_progress(progress_file, progress_data)
                if use_wandb:
                    wandb.finish()
                return

        # Reload model periodically to prevent memory leaks
        if iteration - last_model_reload >= RELOAD_MODEL_EVERY:
            last_model_reload = iteration
            print("Reloading model to prevent memory leaks...")
            del llm_analyzer
            torch.cuda.empty_cache()
            time.sleep(2)  # Give some time for memory to be freed
            llm_analyzer = LLMAnalyzer(model_id, temperature=temperature)

        # Process the scenario
        scenario_res, error_msg = process_single_scenario(
            scenario_item=scenario_item,
            llm_analyzer=llm_analyzer,
            methods_params_decision=methods_params_decision,
            methods_params_explanation=methods_params_explanation,
            num_dec_exp=num_dec_exp,
            generation_seeds=generation_seeds,
            original_params=original_params,
            iteration=iteration,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )

        # Add timing end and logging here
        scenario_time = time.time() - scenario_start_time
        total_time_sum += scenario_time
        # Skip if processing failed
        if scenario_res is None:
            # Log error to file
            error_file = output_dir / "errors.log"
            with open(error_file, "a") as f:
                f.write(f"Scenario {scenario_id} (iteration {iteration}): {error_msg}\n")

            # Add to failed scenarios in progress data
            if "failed_scenarios" not in progress_data:
                progress_data["failed_scenarios"] = []
            progress_data["failed_scenarios"].append(scenario_id)
            save_progress(progress_file, progress_data)
            continue

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
            scenario_time=scenario_time,
            total_time_sum=total_time_sum,
        )

        # Log metrics
        if use_wandb:
            wandb.log(metrics)

        # Save checkpoint periodically
        if iteration % 10 == 0:
            save_checkpoint(checkpoint_file, processed_scenarios)

    # Finalize
    print(f"Data collection completed. Processed {len(processed_scenarios)} scenarios.")
    save_checkpoint(checkpoint_file, processed_scenarios)
    save_progress(progress_file, progress_data)
    if use_wandb:
        wandb.finish()


def save_scenario_details(scenario_res, output_dir):
    """Save detailed scenario information to a log file."""
    # Create a log file in the output directory
    log_file = output_dir / "scenario_details.log"

    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"\nScenario {scenario_res.get('scenario_id', 'unknown')}:\n")
        f.write("-" * 50 + "\n")

        # Print question
        f.write(f"Question: {scenario_res.get('decision_prompt', 'N/A')}\n")

        # Print correct answer and model's decision
        f.write(f"\nCorrect Answer: {scenario_res.get('correct_label', 'N/A')}\n")
        decision_output = scenario_res.get("decision_output", "")
        if ")" in decision_output:
            choice = decision_output.split(")", 1)[0] + ")"
            f.write(f"Model's Decision: {choice.strip()}\n")
        else:
            f.write(f"Model's Decision: {decision_output}\n")

        # Print explanations and their scores
        if "explanation_attributions" in scenario_res and "decision_attributions" in scenario_res:
            decision_attr = scenario_res["decision_attributions"]
            explanation_attrs = scenario_res["explanation_attributions"]

            # Calculate scores for all explanations
            explanation_scores = []
            for j, expl_attr in enumerate(explanation_attrs):
                spearman_score = calculate_spearman_correlation(decision_attr, expl_attr)
                cosine_score = calculate_cosine_similarity(decision_attr, expl_attr)

                explanation_text = (
                    scenario_res.get("explanation_outputs", ["N/A"])[j]
                    if "explanation_outputs" in scenario_res
                    else scenario_res.get("explanation", "N/A")
                )

                explanation_scores.append(
                    {"index": j, "text": explanation_text, "spearman": spearman_score, "cosine": cosine_score}
                )

            # Sort explanations by Spearman correlation score
            explanation_scores.sort(
                key=lambda x: x["spearman"] if x["spearman"] is not None else float("-inf"), reverse=True
            )

            f.write("\nExplanations and Scores (sorted by Spearman correlation):\n")
            for j, score_info in enumerate(explanation_scores):
                f.write(f"\nExplanation {j+1}:\n")
                f.write(f"Text: {score_info['text']}\n")
                f.write(
                    f"Spearman Correlation: {score_info['spearman']:.4f}\n"
                    if score_info["spearman"] is not None
                    else "Spearman Correlation: N/A\n"
                )
                f.write(
                    f"Cosine Similarity: {score_info['cosine']:.4f}\n"
                    if score_info["cosine"] is not None
                    else "Cosine Similarity: N/A\n"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect data using a specified model")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use")
    parser.add_argument("--dataset", type=str, default="ecqa", help="Dataset to use")
    parser.add_argument("--attribution_method", type=str, default="LIME", help="Attribution method to use")
    parser.add_argument("--num_dec_exp", type=int, default=5, help="Number of explanations per decision")
    parser.add_argument("--subset", type=int, default=None, help="Size of dataset subset to use")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume_run", type=str, help="Name of run to resume")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible experiments")

    args = parser.parse_args()

    run_collect_data(
        model_id=args.model_id,
        dataset_name=args.dataset,
        attribution_method_name=args.attribution_method,
        num_dec_exp=args.num_dec_exp,
        subset=args.subset,
        use_wandb=not args.no_wandb,
        resume_run=args.resume_run,
        temperature=args.temperature,
        base_seed=args.seed,
    )
