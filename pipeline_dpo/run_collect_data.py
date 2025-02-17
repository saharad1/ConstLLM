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
from llm_attribution.LLMAnalyzer import LLMAnalyzer
from llm_attribution.utils_attribution import AttributionMethod
from pipeline_dpo.comp_score import compute_kl_divergence, compute_spearman_score
from prepare_datasets.prepare_choice75 import PreparedCHOICE75Dataset
from prepare_datasets.prepare_codah import PreparedCODAHDataset
from utils.custom_chat_template import custom_apply_chat_template
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


class MethodParams:
    _METHOD_PARAMS_FUNCTIONS = {
        AttributionMethod.LIME.name: lambda n_samples=500, perturbations_per_eval=250: {
            AttributionMethod.LIME.name: {
                "n_samples": n_samples,
                "perturbations_per_eval": perturbations_per_eval,
            }
        },
        AttributionMethod.LIG.name: lambda n_steps=50: {
            AttributionMethod.LIG.name: {
                "n_steps": n_steps,
            }
        },
    }

    @classmethod
    def set_params(cls, method_name: str, **kwargs):
        """Sets the parameters for a specific attribution method."""
        if method_name not in cls._METHOD_PARAMS_FUNCTIONS:
            raise ValueError(f"Invalid method name: {method_name}")
        return cls._METHOD_PARAMS_FUNCTIONS[method_name](**kwargs)


# def prepare_lime_params(n_samples=500, perturbations_per_eval=500):
#     return {
#         AttributionMethod.LIME.name: {
#             "n_samples": n_samples,
#             "perturbations_per_eval": perturbations_per_eval,
#         }
#     }


# def prepare_lig_params(n_steps=50):
#     return {AttributionMethod.LIG.name: {"n_steps": n_steps}}


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
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
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

    assert current_method == next(
        iter(methods_params_explanation)
    ), "Mismatched methods"

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
def run_collect_d(wandb_mode: bool = True):
    # Initialize LLM analyzer
    llm_analyzer = initialize_llm_analyzer()

    # # Prepare attribution for LIME
    # methods_params_decision = prepare_lime_params(
    #     n_samples=500, perturbations_per_eval=250
    # )
    # methods_params_explanation = prepare_lime_params(
    #     n_samples=500, perturbations_per_eval=250
    # )

    # # Prepare attribution for LIG
    # methods_params_decision = prepare_lig_params(n_steps=50)
    # methods_params_explanation = prepare_lig_params(n_steps=50)

    # set configurations
    dataset_name = "codah"  # Set to "codah" or "choice75"
    num_dec_exp = 1
    subset = None  # Set to None to process the entire dataset

    assert dataset_name in [
        "codah",
        "choice75",
    ], f"Invalid dataset name: {dataset_name}"

    # Set parameters using a single function call per method
    attribution_method = AttributionMethod.LIG.name
    if attribution_method == AttributionMethod.LIME.name:
        methods_params_decision = MethodParams.set_params(
            AttributionMethod.LIME.name, n_samples=500, perturbations_per_eval=250
        )
        methods_params_explanation = MethodParams.set_params(
            AttributionMethod.LIME.name, n_samples=500, perturbations_per_eval=250
        )
    elif attribution_method == AttributionMethod.LIG.name:
        methods_params_decision = MethodParams.set_params(
            AttributionMethod.LIG.name, n_steps=1
        )
        methods_params_explanation = MethodParams.set_params(
            AttributionMethod.LIG.name, n_steps=1
        )
    else:
        raise ValueError(f"Invalid attribution method: {attribution_method}")

    # Generate a timestamped run name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"{dataset_name}_{timestamp}_{attribution_method}"
    jsonl_filename = (
        Path("dpo_datasets") / f"{dataset_name}_dpo_datasets" / f"{run_name}.jsonl"
    )

    # Ensure directories exist
    if wandb_mode:
        jsonl_filename.parent.mkdir(parents=True, exist_ok=True)

    # Load and prepare dataset
    prepared_dataset = load_and_prepare_dataset(
        dataset_name=dataset_name, subset=subset
    )

    config = {
        "run_name": run_name,
        "num_dec_exp": num_dec_exp,
        "dataset": dataset_name,
        "subset": subset,
        "attribution_method": attribution_method,
        "methods_params_decision": methods_params_decision,
        "methods_params_explanation": methods_params_explanation,
    }

    # Initialize wandb
    wandb.init(
        project=f"{dataset_name}-dataset-dpo",
        name=run_name,
        config=config,
        mode="online" if wandb_mode else "disabled",
    )

    success_sum = 0
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

        # Compute iteration time
        iteration_time = time.time() - start_time  # End timing

        # Compute key results
        correct_choice = extract_choice(scenario_res.correct_label)
        decision_choices = [
            extract_choice(decision) for decision in scenario_res.decisions_outputs
        ]
        scenario_success_count = sum(
            decision == correct_choice for decision in decision_choices
        )
        most_common_choice, _ = Counter(decision_choices).most_common(1)[0]
        most_common_choice_correct = most_common_choice == correct_choice
        success_sum += most_common_choice_correct
        success_rate = success_sum / iteration

        # spearman correlations
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
                "scenario_sucess_count": scenario_success_count,
                "success_rate": success_rate,
            }
        )


if __name__ == "__main__":
    run_collect_d(wandb_mode=False)
