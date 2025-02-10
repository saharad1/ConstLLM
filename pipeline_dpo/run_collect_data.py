import json
import logging
import os
import time

from datasets import load_dataset
from tqdm import tqdm

from llm_attribution.LLMAnalyzer import LLMAnalyzer
from llm_attribution.utils_attribution import AttributionMethod
from pipeline_dpo.comp_score import compute_kl_divergence, compute_spearman_score
from prepare_datasets.prepare_codah import PreparedCODAHDataset
from utils.cutom_chat_template import custom_apply_chat_template
from utils.data_models import ScenarioResult, ScenarioSummary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Utility functions
def ensure_output_directory(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_methods_params(n_samples=20, perturbations_per_eval=20):
    return {
        AttributionMethod.LIME.name: {
            "n_samples": n_samples,
            "perturbations_per_eval": perturbations_per_eval,
        }
    }


# Dataset preparation
def load_and_prepare_dataset():
    logger.info("Loading and preparing the dataset...")
    raw_dataset = load_dataset(path="jaredfern/codah", name="codah", split="all")
    prepared_dataset = PreparedCODAHDataset(raw_dataset, mode="exp1", subset=20)
    logger.info(f"Number of scenarios: {len(prepared_dataset)}")
    return prepared_dataset


# LLM analyzer initialization
def initialize_llm_analyzer():
    logger.info("Initializing LLM analyzer...")
    return LLMAnalyzer(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda:1"
    )


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
    spearman_scores = []
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
        decision_attributions = scenario_summary.decision_scores[
            AttributionMethod.LIME.name
        ]
        explanation_attributions = scenario_summary.explanation_scores[
            AttributionMethod.LIME.name
        ]
        curr_spearman_score = compute_spearman_score(
            decision_attributions=decision_attributions,
            explanation_attributions=explanation_attributions,
        )
        logger.info(f"Spearman Score for repetition {i+1}: {curr_spearman_score}")
        spearman_scores.append([explanation_output, curr_spearman_score])

    # Best and worst explanations
    explanation_best = max(spearman_scores, key=lambda x: x[1])
    explanation_worst = min(spearman_scores, key=lambda x: x[1])
    return ScenarioResult(
        scenario_id=scenario_item.scenario_id,
        correct_label=scenario_item.label,
        decision_prompt=scenario_item.scenario_string,
        decision_output=decision_output,
        explanation_prompt=scenario_item.explanation_string,
        explanation_best_output=explanation_best[0],
        explanation_best_score=explanation_best[1],
        explanation_worst_output=explanation_worst[0],
        explanation_worst_score=explanation_worst[1],
    )


# Main function
def run_collect_d():
    ensure_output_directory("results/codah_res3")
    prepared_dataset = load_and_prepare_dataset()
    llm_analyzer = initialize_llm_analyzer()

    # Prepare attribution method parameters
    methods_params_decision = prepare_methods_params(
        n_samples=20, perturbations_per_eval=20
    )
    methods_params_explanation = prepare_methods_params(
        n_samples=20, perturbations_per_eval=20
    )

    jsonl_filename = "results/codah_res/codah_results2.jsonl"
    num_dec_exp = 5

    with open(jsonl_filename, "a") as f:
        for idx, scenario_item in tqdm(
            enumerate(prepared_dataset, 1),
            total=len(prepared_dataset),
            desc="Processing Scenarios",
        ):
            try:
                logger.info(f"\n=== Running Scenario {idx} ===")
                scenario_res = process_scenario(
                    llm_analyzer=llm_analyzer,
                    scenario_item=scenario_item,
                    methods_params_decision=methods_params_decision,
                    methods_params_explanation=methods_params_explanation,
                    num_dec_exp=num_dec_exp,
                )
                f.write(json.dumps(scenario_res.to_dict()) + "\n")
            except ValueError as e:
                logger.error(f"Error processing scenario {idx}: {e}")
                continue


if __name__ == "__main__":
    run_collect_d()
