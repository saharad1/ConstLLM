"""
Module for calculating and tracking metrics in the data collection process.
"""

import numpy as np


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
    spearman_median_score = np.median(scenario_res.spearman_scores)
    spearman_sums["best"] += spearman_best_score
    spearman_best_score_avg = spearman_sums["best"] / iteration
    spearman_sums["worst"] += spearman_worst_score
    spearman_worst_score_avg = spearman_sums["worst"] / iteration
    spearman_sums["median"] += spearman_median_score
    spearman_median_score_avg = spearman_sums["median"] / iteration

    # Cosine similarities
    cosine_best_score = np.max(scenario_res.cosine_scores)
    cosine_worst_score = np.min(scenario_res.cosine_scores)
    cosine_median_score = np.median(scenario_res.cosine_scores)
    cosine_sums["best"] += cosine_best_score
    cosine_best_score_avg = cosine_sums["best"] / iteration
    cosine_sums["worst"] += cosine_worst_score
    cosine_worst_score_avg = cosine_sums["worst"] / iteration
    cosine_sums["median"] += cosine_median_score
    cosine_median_score_avg = cosine_sums["median"] / iteration

    # Prepare metrics dictionary
    metrics = {
        # Tracking general metrics
        "tracking/accuracy": accuracy_label,
        "tracking/spearman_best_score_avg": spearman_best_score_avg,
        "tracking/spearman_worst_score_avg": spearman_worst_score_avg,
        "tracking/spearman_median_score_avg": spearman_median_score_avg,
        "tracking/cosine_best_score_avg": cosine_best_score_avg,
        "tracking/cosine_worst_score_avg": cosine_worst_score_avg,
        "tracking/cosine_median_score_avg": cosine_median_score_avg,
        # Scenario metrics
        "scenario/scenario_id": scenario_res.scenario_id,
        # Spearman metrics
        "scenario/spearman/mean": np.mean(scenario_res.spearman_scores),
        "scenario/spearman/std": np.std(scenario_res.spearman_scores, ddof=1),
        "scenario/spearman/best_score": spearman_best_score,
        "scenario/spearman/worst_score": spearman_worst_score,
        "scenario/spearman/median": spearman_median_score,
        # Cosine metrics
        "scenario/cosine/mean": np.mean(scenario_res.cosine_scores),
        "scenario/cosine/std": np.std(scenario_res.cosine_scores, ddof=1),
        "scenario/cosine/best_score": cosine_best_score,
        "scenario/cosine/worst_score": cosine_worst_score,
        "scenario/cosine/median": cosine_median_score,
    }

    return metrics, success_sum
