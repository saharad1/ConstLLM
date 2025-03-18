# Add imports
import gc
import json
import logging
import os
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psutil
import torch
from tqdm import tqdm

import wandb
from datasets import load_dataset
from src.collect_data.comp_score import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
    compute_kl_divergence,
)
from src.llm_attribution.LLMAnalyzer import LLMAnalyzer
from src.llm_attribution.utils_attribution import AttributionMethod
from src.prepare_datasets.prepare_arc import PreparedARCDataset
from src.prepare_datasets.prepare_choice75 import PreparedCHOICE75Dataset
from src.prepare_datasets.prepare_codah import PreparedCODAHDataset
from src.prepare_datasets.prepare_ecqa import PreparedECQADataset
from src.utils.custom_chat_template import custom_apply_chat_template
from src.utils.data_models import ExplanationRanking, ScenarioScores
from src.utils.general import print_gpu_info
from src.utils.phase_run import MethodParams, run_phase

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print_gpu_info()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_choice(output: str) -> str:
    """
    Extracts the choice (e.g., 'A', 'B') from the model's output.

    Args:
        output: The raw model output string.

    Returns:
        The extracted choice.
    """
    if not output:
        return ""
    # Remove leading/trailing spaces and special characters
    output = output.strip().replace("\u200b", "").lower()
    # Extract the first character (e.g., 'a' from 'a) description')
    choice = output[0]
    return choice


# Dataset preparation
def load_and_prepare_dataset(dataset_name, subset=20):
    logger.info("Loading and preparing the dataset...")
    if dataset_name == "codah":
        raw_dataset = load_dataset(path="jaredfern/codah", name="codah", split="all")
        prepared_dataset = PreparedCODAHDataset(raw_dataset, subset=subset)
    elif dataset_name == "choice75":
        prepared_dataset = PreparedCHOICE75Dataset(subset=subset)
    elif dataset_name == "ecqa":
        raw_dataset = load_dataset(path="yangdong/ecqa", split="all")
        prepared_dataset = PreparedECQADataset(raw_dataset, subset=subset)
    elif dataset_name == "arc_easy":
        raw_dataset = load_dataset(path="ai2_arc", name="ARC-Easy", split="all")
        prepared_dataset = PreparedARCDataset(raw_dataset, subset=subset)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    logger.info(f"Number of scenarios: {len(prepared_dataset)}")
    return prepared_dataset


def process_scenario(
    llm_analyzer,
    scenario_item,
    methods_params_decision,
    methods_params_explanation,
    num_dec_exp,
    logger=None,
) -> ScenarioScores:
    """
    Process a single scenario with the given analyzer, methods, and parameters.

    Args:
        llm_analyzer: The LLM analyzer to use
        scenario_item: The scenario item to process
        methods_params_decision: Parameters for the decision phase
        methods_params_explanation: Parameters for the explanation phase
        num_dec_exp: Number of explanation generations to try
        logger: Optional logger for tracking progress

    Returns:
        ScenarioScores object with the results
    """
    # Logging function
    log_info = logger.info if logger else print

    # Prepare for tracking results
    spearman_scores = []
    cosine_scores = []
    explanation_outputs = []
    current_method = next(iter(methods_params_decision))

    assert current_method == next(iter(methods_params_explanation)), "Mismatched methods"

    log_info(f"Current method: {current_method}")

    # Decision Phase
    decision_prompt = custom_apply_chat_template([{"role": "user", "content": scenario_item.scenario_string}])
    decision_output, decision_result = run_phase(
        llm_analyzer=llm_analyzer,
        prompt=decision_prompt,
        methods_params=methods_params_decision,
        phase="decision",
    )

    decision_attributions = decision_result.methods_scores[current_method]
    explanation_attributions_list = []

    # Multiple explanation generation
    for i in range(num_dec_exp):
        log_info(f"Processing decision and explanation for repetition {i+1}/{num_dec_exp}...")

        # Explanation Phase
        explanation_prompt = custom_apply_chat_template(
            [
                {"role": "user", "content": scenario_item.scenario_string},
                {"role": "assistant", "content": decision_output},
                {"role": "user", "content": scenario_item.explanation_string},
            ]
        )
        explanation_output, explanation_result = run_phase(
            llm_analyzer=llm_analyzer,
            prompt=explanation_prompt,
            methods_params=methods_params_explanation,
            phase="explanation",
        )

        explanation_outputs.append(explanation_output)
        explanation_attributions = explanation_result.methods_scores[current_method]
        explanation_attributions_list.append(explanation_attributions)

        # Compute Spearman correlation
        curr_spearman_score = calculate_spearman_correlation(
            decision_attributions=decision_attributions,
            explanation_attributions=explanation_attributions,
        )

        # Compute Cosine similarity
        curr_cosine_score = calculate_cosine_similarity(
            decision_attributions=decision_attributions,
            explanation_attributions=explanation_attributions,
        )

        log_info(f"Spearman Score for repetition {i+1}: {curr_spearman_score}")
        log_info(f"Cosine Score for repetition {i+1}: {curr_cosine_score}")

        # Store results
        spearman_scores.append(curr_spearman_score)
        cosine_scores.append(curr_cosine_score)

    # Create scenario results object
    scenario_result = ScenarioScores(
        scenario_id=scenario_item.scenario_id,
        correct_label=scenario_item.label,
        decision_prompt=scenario_item.scenario_string,
        decision_output=decision_output,
        explanation_prompt=scenario_item.explanation_string,
        explanation_outputs=explanation_outputs,
        decision_attributions=decision_attributions,
        explanation_attributions=explanation_attributions_list,
        spearman_scores=spearman_scores,
        cosine_scores=cosine_scores,
    )

    return scenario_result


def get_memory_usage():
    """Get current memory usage of the process"""

    process = psutil.Process()
    return {
        "ram_percent": process.memory_percent(),
        "ram_used_gb": process.memory_info().rss / (1024 * 1024 * 1024),
        "gpu_memory_used": torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024) if torch.cuda.is_available() else 0,
    }


def adjust_batch_params(memory_stats, methods_params):
    """Dynamically adjust parameters based on memory usage"""
    if memory_stats["ram_percent"] > 85 or memory_stats["gpu_memory_used"] > 30:
        # Reduce sample size if memory usage is too high
        for method in methods_params:
            # if "n_samples" in methods_params[method]:
            #     methods_params[method]["n_samples"] = max(100, methods_params[method]["n_samples"] // 2)
            if "perturbations_per_eval" in methods_params[method]:
                methods_params[method]["perturbations_per_eval"] = max(100, methods_params[method]["perturbations_per_eval"] // 2)
        return True
    return False


def run_collect_d(model_id: str, wandb_mode: bool = True):

    # set configurations
    wandb_mode = True
    dataset_name = "arc_easy"  # Set to "codah" or "choice75" or "ecqa" or "arc_easy"
    num_dec_exp = 5
    subset = None  # Set to None to process the entire dataset
    attribution_method = AttributionMethod.LIME.name
    device = "cuda"

    # Add configuration for periodic model reload
    RELOAD_MODEL_EVERY = 50  # Reload model every 50 scenarios
    MEMORY_CHECK_INTERVAL = 5  # Check memory every 5 scenarios
    MAX_RETRIES = 3  # Maximum number of retries per scenario

    # Setup signal handler for graceful shutdown
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        print("\nReceived shutdown signal. Completing current scenario...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set parameters using a single function call per method
    if attribution_method == AttributionMethod.LIME.name:
        methods_params_decision = MethodParams.set_params(AttributionMethod.LIME.name, n_samples=500, perturbations_per_eval=500)
        methods_params_explanation = MethodParams.set_params(AttributionMethod.LIME.name, n_samples=500, perturbations_per_eval=500)
    elif attribution_method == AttributionMethod.LIG.name:
        methods_params_decision = MethodParams.set_params(AttributionMethod.LIG.name, n_steps=25)
        methods_params_explanation = MethodParams.set_params(AttributionMethod.LIG.name, n_steps=25)
        device = "auto"
    elif attribution_method == AttributionMethod.SHAPLEY_VALUE_SAMPLING.name:
        methods_params_decision = MethodParams.set_params(
            AttributionMethod.SHAPLEY_VALUE_SAMPLING.name,
            n_samples=25,
            perturbations_per_eval=25,
        )
        methods_params_explanation = MethodParams.set_params(
            AttributionMethod.SHAPLEY_VALUE_SAMPLING.name,
            n_samples=25,
            perturbations_per_eval=25,
        )
        device = "auto"
    elif attribution_method == AttributionMethod.FEATURE_ABLATION.name:
        methods_params_decision = MethodParams.set_params(
            AttributionMethod.FEATURE_ABLATION.name,
            perturbations_per_eval=500,
        )
        methods_params_explanation = MethodParams.set_params(
            AttributionMethod.FEATURE_ABLATION.name,
            perturbations_per_eval=500,
        )
    else:
        raise ValueError(f"Invalid attribution method: {attribution_method}")

    # Generate a timestamped run name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"{dataset_name}_{timestamp}_{attribution_method}"
    jsonl_filename = Path("dpo_datasets") / f"{dataset_name}_dpo_datasets" / f"{run_name}.jsonl"
    # Ensure directories exist
    if wandb_mode:
        jsonl_filename.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": run_name,
        "model_id": model_id,
        "num_dec_exp": num_dec_exp,
        "dataset": dataset_name,
        "subset": subset,
        "attribution_method": attribution_method,
        "methods_params_decision": methods_params_decision,
        "methods_params_explanation": methods_params_explanation,
    }

    # Initialize LLM analyzer
    logger.info("Initializing LLM analyzer...")
    llm_analyzer = LLMAnalyzer(model_id=model_id, device=device)
    last_model_reload = 0

    # Load and prepare dataset
    prepared_dataset = load_and_prepare_dataset(dataset_name=dataset_name, subset=subset)

    # Initialize wandb
    wandb.init(
        project=f"ConstLLM_Collect_Data",
        name=run_name,
        tags=[dataset_name, attribution_method],
        config=config,
        mode="online" if wandb_mode else "disabled",
    )

    # Load checkpoint and initialize progress tracking
    checkpoint_file = jsonl_filename.with_suffix(".checkpoint")
    progress_file = jsonl_filename.with_suffix(".progress")
    processed_scenarios = set()

    # Load checkpoint and progress data
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            processed_scenarios = set(f.read().splitlines())
        logger.info(f"Loaded {len(processed_scenarios)} processed scenarios from checkpoint")

    # Initialize or load progress tracking
    progress_data = {
        "start_time": datetime.now().isoformat(),
        "total_processing_time": 0,
        "total_scenarios": len(prepared_dataset),
        "completed_scenarios": len(processed_scenarios),
        "failed_scenarios": [],
        "avg_scenario_time": 0,
        "estimated_completion_time": None,
    }

    if progress_file.exists():
        with open(progress_file, "r") as f:
            saved_progress = json.load(f)
            progress_data.update(saved_progress)

    success_sum = len(processed_scenarios)
    spearman_best_score_sum, spearman_worst_score_sum = 0, 0
    cosine_best_score_sum, cosine_worst_score_sum = 0, 0

    # Track original parameters for potential adjustment
    original_params = {"decision": methods_params_decision.copy(), "explanation": methods_params_explanation.copy()}

    for iteration, scenario_item in tqdm(
        enumerate(prepared_dataset, 1),
        total=len(prepared_dataset),
        desc="Processing Scenarios",
    ):
        if not running:
            logger.info("Graceful shutdown initiated. Saving progress...")
            break

        # Skip processed scenarios
        if str(scenario_item.scenario_id) in processed_scenarios:
            continue

        # Check if model needs reloading
        if iteration - last_model_reload >= RELOAD_MODEL_EVERY:
            logger.info("Performing periodic model reload...")
            del llm_analyzer
            torch.cuda.empty_cache()
            gc.collect()
            llm_analyzer = LLMAnalyzer(model_id=model_id, device=device)
            last_model_reload = iteration

        # Monitor and adjust resources
        if iteration % MEMORY_CHECK_INTERVAL == 0:
            memory_stats = get_memory_usage()
            if adjust_batch_params(memory_stats, methods_params_decision):
                logger.warning("Adjusted parameters due to high memory usage")
                # wandb.log({"memory_adjustment": True, **memory_stats})

        start_time = time.time()
        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                logger.info(f"\n=== Running Scenario {iteration} (Attempt {retry_count + 1}/{MAX_RETRIES}) ===")

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                scenario_res = process_scenario(
                    llm_analyzer=llm_analyzer,
                    scenario_item=scenario_item,
                    methods_params_decision=methods_params_decision,
                    methods_params_explanation=methods_params_explanation,
                    num_dec_exp=num_dec_exp,
                    logger=logger,
                )

                # Successful processing
                if wandb_mode:
                    with open(jsonl_filename, "a") as f:
                        f.write(json.dumps(asdict(scenario_res)) + "\n")

                    # Update checkpoint
                    with open(checkpoint_file, "a") as f:
                        f.write(f"{scenario_item.scenario_id}\n")
                    processed_scenarios.add(str(scenario_item.scenario_id))

                # Update progress tracking
                scenario_time = time.time() - start_time
                progress_data["total_processing_time"] += scenario_time
                progress_data["completed_scenarios"] = len(processed_scenarios)
                progress_data["avg_scenario_time"] = progress_data["total_processing_time"] / progress_data["completed_scenarios"]

                # Estimate completion time
                remaining_scenarios = progress_data["total_scenarios"] - progress_data["completed_scenarios"]
                estimated_remaining_time = remaining_scenarios * progress_data["avg_scenario_time"]
                progress_data["estimated_completion_time"] = (datetime.now() + timedelta(seconds=estimated_remaining_time)).isoformat()

                # Save progress
                with open(progress_file, "w") as f:
                    json.dump(progress_data, f)

                # Log memory stats
                memory_stats = get_memory_usage()
                wandb.log(
                    {
                        # "memory/ram_percent": memory_stats["ram_percent"],
                        "memory/ram_used_gb": memory_stats["ram_used_gb"],
                        "memory/gpu_memory_used": memory_stats["gpu_memory_used"],
                        # "progress/estimated_completion_time": progress_data["estimated_completion_time"],
                        # "progress/avg_scenario_time": progress_data["avg_scenario_time"],
                    }
                )

                break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                error_msg = f"Error processing scenario {iteration} (Attempt {retry_count}/{MAX_RETRIES}): {str(e)}"
                logger.error(error_msg, exc_info=True)

                if retry_count < MAX_RETRIES:
                    logger.info(f"Retrying in 5 seconds...")
                    time.sleep(5)
                    # Try resetting parameters if they were adjusted
                    methods_params_decision = original_params["decision"].copy()
                    methods_params_explanation = original_params["explanation"].copy()
                else:
                    # Log final failure
                    with open(jsonl_filename.with_suffix(".errors"), "a") as f:
                        f.write(f"Scenario {scenario_item.scenario_id}: {error_msg}\n")
                    progress_data["failed_scenarios"].append(str(scenario_item.scenario_id))
                    with open(progress_file, "w") as f:
                        json.dump(progress_data, f)
                    continue

        # Compute key results
        # Compute iteration time and accuracy
        iteration_time = time.time() - start_time
        correct_choice = extract_choice(scenario_res.correct_label)
        decision_choice = extract_choice(scenario_res.decision_output)
        success_sum += decision_choice == correct_choice
        accuracy_label = success_sum / iteration

        # spearman correlations
        spearman_best_score = np.max(scenario_res.spearman_scores)
        spearman_worst_score = np.min(scenario_res.spearman_scores)
        spearman_best_score_sum += spearman_best_score
        spearman_best_score_avg = spearman_best_score_sum / iteration
        spearman_worst_score_sum += spearman_worst_score
        spearman_worst_score_avg = spearman_worst_score_sum / iteration

        # cosine similarities
        cosine_best_score = np.max(scenario_res.cosine_scores)
        cosine_worst_score = np.min(scenario_res.cosine_scores)
        cosine_best_score_sum += cosine_best_score
        cosine_best_score_avg = cosine_best_score_sum / iteration
        cosine_worst_score_sum += cosine_worst_score
        cosine_worst_score_avg = cosine_worst_score_sum / iteration

        # Log both Spearman and Cosine metrics in Wandb
        wandb.log(
            {
                "iteration": iteration,
                "accuracy": accuracy_label,
                "spearman_best_score_avg": spearman_best_score_avg,
                "spearman_worst_score_avg": spearman_worst_score_avg,
                "cosine_best_score_avg": cosine_best_score_avg,
                "cosine_worst_score_avg": cosine_worst_score_avg,
                # Spearman Metrics
                "scenario/spearman/mean": np.mean(scenario_res.spearman_scores),
                "scenario/spearman/std": np.std(scenario_res.spearman_scores, ddof=1),
                "scenario/spearman/best_score": spearman_best_score,
                "scenario/spearman/worst_score": spearman_worst_score,
                # Cosine Metrics
                "scenario/cosine/mean": np.mean(scenario_res.cosine_scores),
                "scenario/cosine/std": np.std(scenario_res.cosine_scores, ddof=1),
                "scenario/cosine/best_score": cosine_best_score,
                "scenario/cosine/worst_score": cosine_worst_score,
                # Other existing metrics
                "scenario/scenario_id": scenario_res.scenario_id,
                "scenario/iteration_time_seconds": time.time() - start_time,
            }
        )


if __name__ == "__main__":
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    run_collect_d(model_id=model_id, wandb_mode=True)
