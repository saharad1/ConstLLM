"""
Core processor for individual scenarios in the data collection pipeline.

This module contains the core logic for processing a single scenario through both
decision and explanation phases. It handles attribution methods, prompt generation,
and calculates similarity metrics between decision and explanation attributions.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from src.collect_data.comp_similarity_scores import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
)
from src.utils.custom_chat_template import custom_apply_chat_template
from src.utils.data_models import ScenarioScores
from src.utils.phase_run import MethodParams, run_phase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def process_scenario(
    llm_analyzer,
    scenario_item,
    methods_params_decision,
    methods_params_explanation,
    num_dec_exp,
    custom_logger=None,
    pre_generated_decision_output=None,
    pre_generated_decision_attributions=None,
    pre_generated_explanation_outputs=None,
    generation_seeds=None,
) -> ScenarioScores:
    """
    Process a single scenario with the given analyzer, methods, and parameters.
    Works with both class-based and dictionary-based scenario items.

    Args:
        llm_analyzer: The LLM analyzer to use
        scenario_item: The scenario item to process (either a dictionary or an object)
        methods_params_decision: Parameters for the decision phase
        methods_params_explanation: Parameters for the explanation phase
        num_dec_exp: Number of explanation generations to try
        custom_logger: Optional custom logger for tracking progress
        pre_generated_decision_output: Optional pre-generated output for the decision phase
        pre_generated_decision_attributions: Optional pre-generated attributions for the decision phase
        pre_generated_explanation_outputs: Optional list of pre-generated outputs for explanation phases
        generation_seeds: Optional list of seeds for reproducible explanation generations

    Returns:
        ScenarioScores object with the results
    """
    # Use the provided logger or the module logger
    log_info = custom_logger.info if custom_logger else logger.info

    # Prepare for tracking results
    spearman_scores = []
    cosine_scores = []
    explanation_outputs = []
    current_method = next(iter(methods_params_decision))

    assert current_method == next(iter(methods_params_explanation)), "Mismatched methods"

    log_info(f"Current method: {current_method}")

    # Get scenario attributes based on whether it's a dictionary or object
    scenario_id = get_scenario_attribute(scenario_item, "scenario_id", "scenario_id")
    decision_prompt = get_scenario_attribute(scenario_item, "decision_prompt", "scenario_string")
    explanation_prompt = get_scenario_attribute(scenario_item, "explanation_prompt", "explanation_string")
    correct_label = get_scenario_attribute(scenario_item, "correct_label", "label")

    # Decision Phase
    decision_prompt_template = custom_apply_chat_template([{"role": "user", "content": decision_prompt}], tokenizer=llm_analyzer.tokenizer)
    decision_output, decision_result = run_phase(
        llm_analyzer=llm_analyzer,
        prompt=decision_prompt_template,
        methods_params=methods_params_decision,
        phase="decision",
        pre_generated_output=pre_generated_decision_output,
        pre_generated_attributions=pre_generated_decision_attributions,
    )

    decision_attributions = decision_result.methods_scores[current_method]
    explanation_attributions_list = []

    # Multiple explanation generation
    for i in range(num_dec_exp):
        log_info(f"Processing decision and explanation for repetition {i+1}/{num_dec_exp}...")

        # Set seed for this generation if seeds are provided
        if generation_seeds is not None and i < len(generation_seeds):
            current_seed = int(generation_seeds[i])
            log_info(f"Using seed {current_seed} for explanation generation {i+1}")

            torch.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
            np.random.seed(current_seed)

        # Get pre-generated explanation output if available
        current_pre_generated_explanation = None
        if pre_generated_explanation_outputs and i < len(pre_generated_explanation_outputs):
            current_pre_generated_explanation = pre_generated_explanation_outputs[i]

        # Explanation Phase
        explanation_prompt_template = custom_apply_chat_template(
            [
                {"role": "user", "content": decision_prompt},
                {"role": "assistant", "content": decision_output},
                {"role": "user", "content": explanation_prompt},
            ],
            tokenizer=llm_analyzer.tokenizer,
        )
        explanation_output, explanation_result = run_phase(
            llm_analyzer=llm_analyzer,
            prompt=explanation_prompt_template,
            methods_params=methods_params_explanation,
            phase="explanation",
            pre_generated_output=current_pre_generated_explanation,
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
        scenario_id=scenario_id,
        correct_label=correct_label,
        decision_prompt=decision_prompt,
        decision_output=decision_output,
        explanation_prompt=explanation_prompt,
        explanation_outputs=explanation_outputs,
        decision_attributions=decision_attributions,
        explanation_attributions=explanation_attributions_list,
        spearman_scores=spearman_scores,
        cosine_scores=cosine_scores,
    )

    return scenario_result
