import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import wandb
from datasets import load_dataset
from llm_attribution.LLMAnalyzer import LLMAnalyzer
from llm_attribution.utils_attribution import AttributionMethod
from pipeline_dpo.comp_score import compute_kl_divergence, compute_spearman_score
from prepare_datasets.prepare_codah import PreparedCODAHDataset
from utils.cutom_chat_template import custom_apply_chat_template
from utils.data_models import (
    ExplanationRanking,
    ScenarioResult,
    ScenarioScores,
    ScenarioSummary,
)
from utils.general import print_gpu_info

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print_gpu_info()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Utility functions
def ensure_output_directory(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_lime_params(n_samples=500, perturbations_per_eval=500):
    return {
        AttributionMethod.LIME.name: {
            "n_samples": n_samples,
            "perturbations_per_eval": perturbations_per_eval,
        }
    }


def prepare_lig_params(n_steps=50):
    return {AttributionMethod.LIG.name: {"n_steps": n_steps}}


# Dataset preparation
def load_and_prepare_dataset(subset=20):
    logger.info("Loading and preparing the dataset...")
    raw_dataset = load_dataset(path="jaredfern/codah", name="codah", split="all")
    prepared_dataset = PreparedCODAHDataset(raw_dataset, mode="exp1", subset=subset)
    logger.info(f"Number of scenarios: {len(prepared_dataset)}")
    return prepared_dataset


# LLM analyzer initialization
def initialize_llm_analyzer():
    logger.info("Initializing LLM analyzer...")
    return LLMAnalyzer(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda")


# Generalized phase function
def run_phase(llm_analyzer, prompt, methods_params, phase="decision"):
    logger.info(f"Running {phase} phase...")
    output = llm_analyzer.generate_output(prompt)
    result = llm_analyzer.analyze(
        input_text=prompt, target=output, method_params=methods_params
    )
    if not result:
        raise ValueError(f"{phase.capitalize()} phase returned invalid results")
    return output, result


# Process individual scenario
def process_scenario(
    llm_analyzer,
    scenario_item,
    methods_params_decision,
    methods_params_explanation,
    num_dec_exp,
):
    spearman_triplet = []
    decision_outputs = []
    explanation_outputs = []
    spearman_scores = []
    current_method = next(iter(methods_params_decision))
    print(f"Current method: {current_method}")
    for i in range(num_dec_exp):
        logger.info(
            f"Processing decision and explanation for repetition {i+1}/{num_dec_exp}..."
        )
        # Decision Phase
        decision_prompt = custom_apply_chat_template(
            [{"role": "user", "content": scenario_item.scenario_string}]
        )
        decision_output, decision_result = run_phase(
            llm_analyzer=llm_analyzer,
            prompt=decision_prompt,
            methods_params=methods_params_decision,
            phase="decision",
        )
        decision_outputs.append(decision_output)

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

        # Scenario summary and scores
        scenario_summary = ScenarioSummary(
            scenario_id=scenario_item.scenario_id,
            correct_label=scenario_item.label,
            decision_prompt=scenario_item.scenario_string,
            decision_output=decision_output,
            decision_scores=decision_result.methods_scores,
            explanation_prompt=scenario_item.explanation_string,
            explanation_output=explanation_output,
            explanation_scores=explanation_result.methods_scores,
        )

        # Compute Spearman score
        decision_attributions = scenario_summary.decision_scores[current_method]
        explanation_attributions = scenario_summary.explanation_scores[current_method]
        print(f"Decision attributions: {decision_attributions}")
        sys.exit()
        curr_spearman_score = compute_spearman_score(
            decision_attributions=decision_attributions,
            explanation_attributions=explanation_attributions,
        )
        logger.info(f"Spearman Score for repetition {i+1}: {curr_spearman_score}")
        spearman_scores.append(curr_spearman_score)
        spearman_triplet.append(
            ExplanationRanking(decision_output, explanation_output, curr_spearman_score)
        )

    # Best and worst explanations
    explanation_best = max(spearman_triplet, key=lambda x: x.spearman_score)
    explanation_worst = min(spearman_triplet, key=lambda x: x.spearman_score)
    # scenario_result= ScenarioResult(
    #     scenario_id=scenario_item.scenario_id,
    #     correct_label=scenario_item.label,
    #     decision_prompt=scenario_item.scenario_string,
    #     decision_output=decision_output,
    #     explanation_prompt=scenario_item.explanation_string,
    #     explanation_best_output=explanation_best[0],
    #     explanation_best_score=explanation_best[1],
    #     explanation_worst_output=explanation_worst[0],
    #     explanation_worst_score=explanation_worst[1],
    # )
    scenario_result = ScenarioScores(
        scenario_id=scenario_item.scenario_id,
        correct_label=scenario_item.label,
        decision_prompt=scenario_item.scenario_string,
        decisions_outputs=decision_outputs,
        explanation_prompt=scenario_item.explanation_string,
        explanation_outputs=explanation_outputs,
        decision_attributions=decision_attributions,
        explanation_attributions=explanation_attributions,
        spearman_scores=spearman_scores,
        explanation_best=explanation_best,
        explanation_worst=explanation_worst,
    )

    return scenario_result


# Main function
def run_collect_d():
    # Initialize LLM analyzer
    llm_analyzer = initialize_llm_analyzer()

    # Prepare attribution method parameters
    methods_params_decision = prepare_lime_params(
        n_samples=500, perturbations_per_eval=500
    )
    methods_params_explanation = prepare_lime_params(
        n_samples=500, perturbations_per_eval=500
    )

    # Generate a timestamped run name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"codah_{timestamp}_LLama"
    jsonl_filename = Path("dpo_datasets") / "codah_dpo_datasets" / f"{run_name}.jsonl"

    # Ensure directories exist
    jsonl_filename.parent.mkdir(parents=True, exist_ok=True)

    num_dec_exp = 5
    subset = None  # Set to None to process the entire dataset

    # Load and prepare dataset
    prepared_dataset = load_and_prepare_dataset(subset=subset)

    config = {
        "run_name": run_name,
        "num_dec_exp": num_dec_exp,
        "subset": subset,
        "methods_params_decision": methods_params_decision,
        "methods_params_explanation": methods_params_explanation,
    }

    # Initialize wandb
    wandb_log = False
    wandb.init(
        project="codah-dataset-dpo",
        name=run_name,
        config=config,
        mode="online" if wandb_log else "disabled",
    )

    with open(jsonl_filename, "a") as f:
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
                f.write(json.dumps(asdict(scenario_res)) + "\n")
            except ValueError as e:
                logger.error(f"Error processing scenario {iteration}: {e}")
                continue

            # Compute iteration time
            iteration_time = time.time() - start_time  # End timing

            spearman_best_score = scenario_res.explanation_best.spearman_score
            spearman_worst_score = scenario_res.explanation_worst.spearman_score
            std_spearman = np.std(np.array(scenario_res.spearman_scores), ddof=1)

            # Log key results and iteration time for tracking progress in wandb
            wandb.log(
                {
                    "iteration": iteration,
                    "scenario_id": scenario_res.scenario_id,
                    "spearman_best_score": spearman_best_score,
                    "worst_spearman_score": spearman_worst_score,
                    "spearman_diff": spearman_best_score - spearman_worst_score,
                    "spearman_std": std_spearman,
                    "iteration_time_seconds": iteration_time,  # Log the time taken per scenario
                }
            )


if __name__ == "__main__":
    run_collect_d()
