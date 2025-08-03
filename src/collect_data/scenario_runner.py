"""
Runner for executing scenarios in the data collection pipeline.

This module handles the execution flow for scenarios, including retry logic,
error handling, and saving results to files. It acts as a wrapper around the
core processing functionality, providing robustness and persistence.
"""

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from src.collect_data.comp_similarity_scores import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
)

# Import the process_scenario function from the renamed module
from src.collect_data.scenario_core_processor import (
    get_scenario_attribute,
    process_scenario,
)


def process_single_scenario(
    scenario_item,
    llm_analyzer,
    methods_params_decision,
    methods_params_explanation,
    num_dec_exp,
    generation_seeds,
    original_params,
    iteration,
    output_dir,
    jsonl_filename,
    max_retries=3,
):
    """
    Process a single scenario with retries.

    Args:
        scenario_item: The scenario item to process
        llm_analyzer: The LLM analyzer instance
        methods_params_decision: Parameters for decision attribution methods
        methods_params_explanation: Parameters for explanation attribution methods
        num_dec_exp: Number of decision explanations to generate
        generation_seeds: List of seeds for generation
        original_params: Original parameters for the scenario
        iteration: Current iteration number
        output_dir: Directory to save output files
        jsonl_filename: Path to the JSONL file for saving results
        max_retries: Maximum number of retries for processing a scenario

    Returns:
        Tuple of (scenario_result, error_message)
    """
    # Extract scenario ID for tracking
    scenario_id = get_scenario_attribute(scenario_item, "scenario_id", "scenario_id", f"scenario_{iteration}")

    # Initialize error message
    error_msg = None

    # Try processing with retries
    for retry in range(max_retries):
        try:
            # Process the scenario using the process_scenario function
            scenario_res = process_scenario(
                scenario_item=scenario_item,
                llm_analyzer=llm_analyzer,
                methods_params_decision=methods_params_decision,
                methods_params_explanation=methods_params_explanation,
                num_dec_exp=num_dec_exp,
                generation_seeds=generation_seeds,
            )

            # Ensure scenario has an ID
            if not hasattr(scenario_res, "scenario_id") or not scenario_res.scenario_id:
                scenario_res.scenario_id = scenario_id

            # After processing is successful, save both the result and detailed log
            # This ensures scenarios are saved even if they succeed after retries
            save_scenario_result(scenario_res, jsonl_filename)
            save_scenario_details(scenario_res, output_dir)

            return scenario_res, None

        except Exception as e:
            error_msg = f"Error processing scenario (retry {retry+1}/{max_retries}): {str(e)}\n{traceback.format_exc()}"
            print(error_msg)

            # Wait before retrying
            time.sleep(2)

            # Reset parameters for retry
            methods_params_decision = original_params["decision"].copy()
            methods_params_explanation = original_params["explanation"].copy()

    # If we get here, all retries failed
    return None, error_msg


# def save_scenario_result(scenario_res, jsonl_filename):
#     """
#     Save a scenario result to a JSONL file.

#     Args:
#         scenario_res: The scenario result to save
#         jsonl_filename: Path to the JSONL file

#     Returns:
#         True if successful, False otherwise
#     """
#     try:
#         # Convert dataclass to dictionary if needed
#         scenario_dict = asdict(scenario_res) if hasattr(scenario_res, "__dataclass_fields__") else scenario_res

#         # Write to JSONL file
#         with open(jsonl_filename, "a") as f:
#             f.write(f"{scenario_dict}\n")
#         return True
#     except Exception as e:
#         print(f"Error saving scenario result: {e}")
#         return False


def save_scenario_result(scenario_res, jsonl_filename):
    """
    Save a scenario result to a JSONL file.

    Args:
        scenario_res: The scenario result to save
        jsonl_filename: Path to the JSONL file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert dataclass to dictionary if needed
        scenario_dict = asdict(scenario_res) if hasattr(scenario_res, "__dataclass_fields__") else scenario_res

        # Write to JSONL file using json module
        with open(jsonl_filename, "a") as f:
            json.dump(scenario_dict, f)
            f.write("\n")
        return True
    except Exception as e:
        print(f"Error saving scenario result: {e}")
        return False


def save_scenario_details(scenario_res, output_dir):
    """Save detailed scenario information to a log file."""
    # Create a log file in the output directory
    log_file = output_dir / "scenario_details.log"

    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"\nScenario {scenario_res.scenario_id}:\n")
        f.write("-" * 50 + "\n")

        # Print question
        f.write(f"Question: {scenario_res.decision_prompt}\n")

        # Print correct answer and model's decision
        f.write(f"\nCorrect Answer: {scenario_res.correct_label}\n")
        decision_output = scenario_res.decision_output
        f.write(f"Model's Decision: {decision_output}\n")

        # Print explanations and their scores
        if hasattr(scenario_res, "explanation_attributions") and hasattr(scenario_res, "decision_attributions"):
            decision_attr = scenario_res.decision_attributions
            explanation_attrs = scenario_res.explanation_attributions

            # Calculate scores for all explanations
            explanation_scores = []
            for j, expl_attr in enumerate(explanation_attrs):
                spearman_score = calculate_spearman_correlation(decision_attr, expl_attr)
                cosine_score = calculate_cosine_similarity(decision_attr, expl_attr)

                explanation_text = (
                    scenario_res.explanation_outputs[j]
                    if hasattr(scenario_res, "explanation_outputs")
                    else scenario_res.explanation
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
