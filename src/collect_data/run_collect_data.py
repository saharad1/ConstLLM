import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
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
        raw_dataset = load_dataset(path="ai2_arc", name="ARC-Easy", split="train")
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


# def process_scenario(
#     llm_analyzer,
#     scenario_item,
#     methods_params_decision,
#     methods_params_explanation,
#     num_dec_exp,
#     similarity_metric="spearman",  # Add similarity_metric parameter with default "spearman"
# ):
#     """
#     Process a single scenario with the given analyzer, methods, and parameters.

#     Args:
#         llm_analyzer: The LLM analyzer to use
#         scenario_item: The scenario item to process
#         methods_params_decision: Parameters for the decision phase
#         methods_params_explanation: Parameters for the explanation phase
#         num_dec_exp: Number of explanation generations to try
#         similarity_metric: Which similarity metric to use ("spearman" or "cosine")

#     Returns:
#         ScenarioScores object with the results
#     """
#     similarity_triplet = []
#     explanation_outputs = []
#     similarity_scores = []
#     current_method = next(iter(methods_params_decision))

#     assert current_method == next(iter(methods_params_explanation)), "Mismatched methods"

#     print(f"Current method: {current_method}")
#     print(f"Using similarity metric: {similarity_metric}")

#     # Decision Phase
#     decision_prompt = custom_apply_chat_template([{"role": "user", "content": scenario_item.scenario_string}])
#     decision_output, decision_result = run_phase(
#         llm_analyzer=llm_analyzer,
#         prompt=decision_prompt,
#         methods_params=methods_params_decision,
#         phase="decision",
#     )

#     decision_attributions = decision_result.methods_scores[current_method]
#     explanation_attributions_list = []
#     for i in range(num_dec_exp):
#         logger.info(f"Processing decision and explanation for repetition {i+1}/{num_dec_exp}...")
#         # Explanation Phase
#         explanation_prompt = custom_apply_chat_template(
#             [
#                 {"role": "user", "content": scenario_item.scenario_string},
#                 {"role": "assistant", "content": decision_output},
#                 {"role": "user", "content": scenario_item.explanation_string},
#             ]
#         )
#         explanation_output, explanation_result = run_phase(
#             llm_analyzer=llm_analyzer,
#             prompt=explanation_prompt,
#             methods_params=methods_params_explanation,
#             phase="explanation",
#         )
#         explanation_outputs.append(explanation_output)
#         explanation_attributions = explanation_result.methods_scores[current_method]
#         explanation_attributions_list.append(explanation_attributions)

#         # Compute similarity score based on the selected metric
#         if similarity_metric == "cosine":
#             curr_similarity_score = calculate_cosine_similarity(
#                 decision_attributions=decision_attributions,
#                 explanation_attributions=explanation_attributions,
#             )
#             score_name = "Cosine Similarity"
#         else:  # Default to Spearman
#             curr_similarity_score = calculate_spearman_correlation(
#                 decision_attributions=decision_attributions,
#                 explanation_attributions=explanation_attributions,
#             )
#             score_name = "Spearman Score"

#         logger.info(f"{score_name} for repetition {i+1}: {curr_similarity_score}")
#         similarity_scores.append(curr_similarity_score)
#         similarity_triplet.append(ExplanationRanking(decision_output, explanation_output, curr_similarity_score))

#     # Best and worst explanations based on the selected metric
#     explanation_best = max(similarity_triplet, key=lambda x: x.similarity_score)
#     explanation_worst = min(similarity_triplet, key=lambda x: x.similarity_score)

#     scenario_result = ScenarioScores(
#         scenario_id=scenario_item.scenario_id,
#         correct_label=scenario_item.label,
#         decision_prompt=scenario_item.scenario_string,
#         decision_output=decision_output,
#         explanation_prompt=scenario_item.explanation_string,
#         explanation_outputs=explanation_outputs,
#         decision_attributions=decision_attributions,
#         explanation_attributions=explanation_attributions_list,
#         spearman_scores=similarity_scores,  # Renamed from spearman_scores but keeping the same field
#         explanation_best=explanation_best,
#         explanation_worst=explanation_worst,
#     )

#     return scenario_result


# Main function
def run_collect_d(model_id: str, wandb_mode: bool = True):

    # set configurations
    wandb_mode = True
    dataset_name = "arc_easy"  # Set to "codah" or "choice75" or "ecqa" or "arc_easy"
    num_dec_exp = 5
    subset = None  # Set to None to process the entire dataset
    attribution_method = AttributionMethod.LIME.name
    device = "cuda"

    assert dataset_name in [
        "codah",
        "choice75",
        "ecqa",
        "arc_easy",
    ], f"Invalid dataset name: {dataset_name}"

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

    success_sum = 0
    spearman_best_score_sum, spearman_worst_score_sum = 0, 0
    cosine_best_score_sum, cosine_worst_score_sum = 0, 0
    for iteration, scenario_item in tqdm(
        enumerate(prepared_dataset, 1),
        total=len(prepared_dataset),
        desc="Processing Scenarios",
    ):
        start_time = time.time()  # Start timing

        try:
            logger.info(f"\n=== Running Scenario {iteration} ===")
            scenario_res = process_scenario(
                llm_analyzer=llm_analyzer,
                scenario_item=scenario_item,
                methods_params_decision=methods_params_decision,
                methods_params_explanation=methods_params_explanation,
                num_dec_exp=num_dec_exp,  # Number of decision-explanation repetitions
            )
            if wandb_mode:
                with open(jsonl_filename, "a") as f:
                    f.write(json.dumps(asdict(scenario_res)) + "\n")
        except ValueError as e:
            logger.error(f"Error processing scenario {iteration}: {e}")
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

        # Log key results and iteration time for tracking progress in wandb
        # wandb.log(
        #     {
        #         "iteration": iteration,
        #         "accuracy": accuracy_label,
        #         "spearman_best_score_avg": spearman_best_score_avg,
        #         "spearman_worst_score_avg": spearman_worst_score_avg,
        #         "scenario/scenario_id": scenario_res.scenario_id,
        #         "scenario/iteration_time_seconds": iteration_time,
        #         "scenario/best_spearman_score": spearman_best_score,
        #         "scenario/worst_spearman_score": spearman_worst_score,
        #         "scenario/spearman_diff": spearman_best_score - spearman_worst_score,
        #         "scenario/spearman_std": std_spearman,
        #     }
        # )


if __name__ == "__main__":
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    run_collect_d(model_id=model_id, wandb_mode=True)
