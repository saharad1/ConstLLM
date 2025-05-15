import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analyze_data.analysis_utils import (
    compute_explanation_ranks,
    extract_choice,
    parse_line,
)
from src.collect_data.comp_similarity_scores import (
    calculate_cosine_similarity,
    calculate_lma,
    calculate_spearman_correlation,
)


def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """
    Analyze a dataset file and compute various metrics including accuracy, cosine similarity,
    and spearman correlation statistics.

    Args:
        file_path: Path to the dataset file (can be JSONL or Python dictionary format)

    Returns:
        Dictionary containing the computed metrics
    """
    # Initialize counters and lists
    total_scenarios = 0
    correct_decisions = 0
    spearman_scores = []
    cosine_scores = []
    lma_scores = []

    # Dataset-level statistics
    dataset_spearman_worst = []
    dataset_spearman_median = []
    dataset_spearman_best = []
    dataset_spearman_mean = []
    dataset_cosine_worst = []
    dataset_cosine_median = []
    dataset_cosine_best = []
    dataset_cosine_mean = []
    dataset_lma_worst = []
    dataset_lma_median = []
    dataset_lma_best = []
    dataset_lma_mean = []

    # Separate lists for correct and incorrect model decisions
    spearman_worst_right = []
    spearman_best_right = []
    spearman_mean_right = []
    spearman_worst_wrong = []
    spearman_best_wrong = []
    spearman_mean_wrong = []
    cosine_worst_right = []
    cosine_best_right = []
    cosine_mean_right = []
    cosine_worst_wrong = []
    cosine_best_wrong = []
    cosine_mean_wrong = []
    lma_worst_right = []
    lma_best_right = []
    lma_mean_right = []
    lma_worst_wrong = []
    lma_best_wrong = []
    lma_mean_wrong = []

    # Read and process the dataset
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                scenario = parse_line(line)
                total_scenarios += 1

                # Check accuracy using the same method as collection phase
                correct_choice = extract_choice(scenario.get("correct_label", ""))
                decision_choice = extract_choice(scenario.get("decision_output", ""))
                is_correct = correct_choice and decision_choice and correct_choice == decision_choice
                if is_correct:
                    correct_decisions += 1

                # Process explanation attributions
                if "explanation_attributions" in scenario and "decision_attributions" in scenario:
                    decision_attr = scenario["decision_attributions"]
                    explanation_attrs = scenario["explanation_attributions"]

                    # Calculate scores for each explanation
                    scenario_spearman = []
                    scenario_cosine = []
                    scenario_lma = []

                    for expl_attr in explanation_attrs:
                        # Calculate Spearman correlation
                        spearman_score = calculate_spearman_correlation(decision_attr, expl_attr)
                        if spearman_score is not None:
                            scenario_spearman.append(spearman_score)
                            spearman_scores.append(spearman_score)

                        # Calculate Cosine similarity
                        cosine_score = calculate_cosine_similarity(decision_attr, expl_attr)
                        if cosine_score is not None:
                            scenario_cosine.append(cosine_score)
                            cosine_scores.append(cosine_score)

                        # Calculate LMA
                        lma_score = calculate_lma(decision_attr, expl_attr)
                        if lma_score is not None:
                            scenario_lma.append(lma_score)
                            lma_scores.append(lma_score)

                    # Add per-scenario statistics
                    if scenario_spearman:
                        dataset_spearman_worst.append(min(scenario_spearman))
                        dataset_spearman_median.append(np.median(scenario_spearman))
                        dataset_spearman_best.append(max(scenario_spearman))
                        dataset_spearman_mean.append(np.mean(scenario_spearman))
                        # Split by correctness
                        if is_correct:
                            spearman_worst_right.append(min(scenario_spearman))
                            spearman_best_right.append(max(scenario_spearman))
                            spearman_mean_right.append(np.mean(scenario_spearman))
                        else:
                            spearman_worst_wrong.append(min(scenario_spearman))
                            spearman_best_wrong.append(max(scenario_spearman))
                            spearman_mean_wrong.append(np.mean(scenario_spearman))

                    if scenario_cosine:
                        dataset_cosine_worst.append(min(scenario_cosine))
                        dataset_cosine_median.append(np.median(scenario_cosine))
                        dataset_cosine_best.append(max(scenario_cosine))
                        dataset_cosine_mean.append(np.mean(scenario_cosine))
                        # Split by correctness
                        if is_correct:
                            cosine_worst_right.append(min(scenario_cosine))
                            cosine_best_right.append(max(scenario_cosine))
                            cosine_mean_right.append(np.mean(scenario_cosine))
                        else:
                            cosine_worst_wrong.append(min(scenario_cosine))
                            cosine_best_wrong.append(max(scenario_cosine))
                            cosine_mean_wrong.append(np.mean(scenario_cosine))

                    if scenario_lma:
                        dataset_lma_worst.append(min(scenario_lma))
                        dataset_lma_median.append(np.median(scenario_lma))
                        dataset_lma_best.append(max(scenario_lma))
                        dataset_lma_mean.append(np.mean(scenario_lma))
                        # Split by correctness
                        if is_correct:
                            lma_worst_right.append(min(scenario_lma))
                            lma_best_right.append(max(scenario_lma))
                            lma_mean_right.append(np.mean(scenario_lma))
                        else:
                            lma_worst_wrong.append(min(scenario_lma))
                            lma_best_wrong.append(max(scenario_lma))
                            lma_mean_wrong.append(np.mean(scenario_lma))

            except ValueError as e:
                print(f"Error on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing scenario on line {line_num}: {e}")
                continue

    # Compute final metrics
    metrics = {
        "total_scenarios": total_scenarios,
        "accuracy": correct_decisions / total_scenarios if total_scenarios > 0 else 0,
        # Cosine similarity statistics
        "cosine": {
            "worst": {
                "mean": np.mean(dataset_cosine_worst) if dataset_cosine_worst else 0,
                "median": np.median(dataset_cosine_worst) if dataset_cosine_worst else 0,
                "min": min(dataset_cosine_worst) if dataset_cosine_worst else 0,
                "max": max(dataset_cosine_worst) if dataset_cosine_worst else 0,
            },
            # "median": {
            #     "mean": np.mean(dataset_cosine_median) if dataset_cosine_median else 0,
            #     "median": np.median(dataset_cosine_median) if dataset_cosine_median else 0,
            #     "min": min(dataset_cosine_median) if dataset_cosine_median else 0,
            #     "max": max(dataset_cosine_median) if dataset_cosine_median else 0,
            # },
            "best": {
                "mean": np.mean(dataset_cosine_best) if dataset_cosine_best else 0,
                "median": np.median(dataset_cosine_best) if dataset_cosine_best else 0,
                "min": min(dataset_cosine_best) if dataset_cosine_best else 0,
                "max": max(dataset_cosine_best) if dataset_cosine_best else 0,
            },
            "mean": {
                "mean": np.mean(dataset_cosine_mean) if dataset_cosine_mean else 0,
                "median": np.median(dataset_cosine_mean) if dataset_cosine_mean else 0,
                "min": min(dataset_cosine_mean) if dataset_cosine_mean else 0,
                "max": max(dataset_cosine_mean) if dataset_cosine_mean else 0,
            },
        },
        # Spearman correlation statistics
        "spearman": {
            "worst": {
                "mean": np.mean(dataset_spearman_worst) if dataset_spearman_worst else 0,
                "median": np.median(dataset_spearman_worst) if dataset_spearman_worst else 0,
                "min": min(dataset_spearman_worst) if dataset_spearman_worst else 0,
                "max": max(dataset_spearman_worst) if dataset_spearman_worst else 0,
            },
            "median": {
                "mean": np.mean(dataset_spearman_median) if dataset_spearman_median else 0,
                "median": np.median(dataset_spearman_median) if dataset_spearman_median else 0,
                "min": min(dataset_spearman_median) if dataset_spearman_median else 0,
                "max": max(dataset_spearman_median) if dataset_spearman_median else 0,
            },
            "best": {
                "mean": np.mean(dataset_spearman_best) if dataset_spearman_best else 0,
                "median": np.median(dataset_spearman_best) if dataset_spearman_best else 0,
                "min": min(dataset_spearman_best) if dataset_spearman_best else 0,
                "max": max(dataset_spearman_best) if dataset_spearman_best else 0,
            },
            "mean": {
                "mean": np.mean(dataset_spearman_mean) if dataset_spearman_mean else 0,
                "median": np.median(dataset_spearman_mean) if dataset_spearman_mean else 0,
                "min": min(dataset_spearman_mean) if dataset_spearman_mean else 0,
                "max": max(dataset_spearman_mean) if dataset_spearman_mean else 0,
            },
        },
        "lma": {
            "worst": {
                "mean": np.mean(dataset_lma_worst) if dataset_lma_worst else 0,
                "median": np.median(dataset_lma_worst) if dataset_lma_worst else 0,
                "min": min(dataset_lma_worst) if dataset_lma_worst else 0,
                "max": max(dataset_lma_worst) if dataset_lma_worst else 0,
            },
            # "median": {
            #     "mean": np.mean(dataset_lma_median) if dataset_lma_median else 0,
            #     "median": np.median(dataset_lma_median) if dataset_lma_median else 0,
            #     "min": min(dataset_lma_median) if dataset_lma_median else 0,
            #     "max": max(dataset_lma_median) if dataset_lma_median else 0,
            # },
            "best": {
                "mean": np.mean(dataset_lma_best) if dataset_lma_best else 0,
                "median": np.median(dataset_lma_best) if dataset_lma_best else 0,
                "min": min(dataset_lma_best) if dataset_lma_best else 0,
                "max": max(dataset_lma_best) if dataset_lma_best else 0,
            },
            "mean": {
                "mean": np.mean(dataset_lma_mean) if dataset_lma_mean else 0,
                "median": np.median(dataset_lma_mean) if dataset_lma_mean else 0,
                "min": min(dataset_lma_mean) if dataset_lma_mean else 0,
                "max": max(dataset_lma_mean) if dataset_lma_mean else 0,
            },
        },
        # Add new metrics for correct/incorrect cases
        "cosine_right": {
            "worst_mean": np.mean(cosine_worst_right) if cosine_worst_right else 0,
            "mean_mean": np.mean(cosine_mean_right) if cosine_mean_right else 0,
            "best_mean": np.mean(cosine_best_right) if cosine_best_right else 0,
        },
        "cosine_wrong": {
            "worst_mean": np.mean(cosine_worst_wrong) if cosine_worst_wrong else 0,
            "mean_mean": np.mean(cosine_mean_wrong) if cosine_mean_wrong else 0,
            "best_mean": np.mean(cosine_best_wrong) if cosine_best_wrong else 0,
        },
        "spearman_right": {
            "worst_mean": np.mean(spearman_worst_right) if spearman_worst_right else 0,
            "mean_mean": np.mean(spearman_mean_right) if spearman_mean_right else 0,
            "best_mean": np.mean(spearman_best_right) if spearman_best_right else 0,
        },
        "spearman_wrong": {
            "worst_mean": np.mean(spearman_worst_wrong) if spearman_worst_wrong else 0,
            "mean_mean": np.mean(spearman_mean_wrong) if spearman_mean_wrong else 0,
            "best_mean": np.mean(spearman_best_wrong) if spearman_best_wrong else 0,
        },
        "lma_right": {
            "worst_mean": np.mean(lma_worst_right) if lma_worst_right else 0,
            "mean_mean": np.mean(lma_mean_right) if lma_mean_right else 0,
            "best_mean": np.mean(lma_best_right) if lma_best_right else 0,
        },
        "lma_wrong": {
            "worst_mean": np.mean(lma_worst_wrong) if lma_worst_wrong else 0,
            "mean_mean": np.mean(lma_mean_wrong) if lma_mean_wrong else 0,
            "best_mean": np.mean(lma_best_wrong) if lma_best_wrong else 0,
        },
    }

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print the metrics in a readable format."""
    print("\n=== Dataset Analysis Results ===")
    print(f"Total scenarios analyzed: {metrics['total_scenarios']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")

    print("\n=== Spearman Correlation Statistics ===")
    for category in ["worst", "best", "mean"]:
        print(f"\n{category.title()} values:")
        stats = metrics["spearman"][category]
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n=== Cosine Similarity Statistics ===")
    for category in ["worst", "best", "mean"]:
        print(f"\n{category.title()} values:")
        stats = metrics["cosine"][category]
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n=== LMA Statistics ===")
    for category in ["worst", "best", "mean"]:
        print(f"\n{category.title()} values:")
        stats = metrics["lma"][category]
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # Print correct/incorrect split metrics
    for metric in ["cosine", "spearman", "lma"]:
        print(f"\n=== {metric.title()} Means by Model Correctness ===")
        for correctness in ["right", "wrong"]:
            key = f"{metric}_{correctness}"
            print(f"\nWhen model was {correctness}:")
            print(f"  Mean of Worst: {metrics[key]['worst_mean']:.4f}")
            print(f"  Mean of Mean: {metrics[key]['mean_mean']:.4f}")
            print(f"  Mean of Best: {metrics[key]['best_mean']:.4f}")


def print_scenario_details(file_path: str, num_scenarios: int = 20, output_file: str = None) -> None:
    """
    Print detailed information about a specified number of scenarios from the dataset.

    Args:
        file_path: Path to the dataset file
        num_scenarios: Number of scenarios to print (default: 20)
        output_file: Optional path to output file. If provided, results will be written to this file
    """
    # Set up output stream
    import sys

    original_stdout = sys.stdout
    if output_file:
        f = open(output_file, "w")
        sys.stdout = f

    try:
        print(f"\n=== Printing details for {num_scenarios} scenarios ===")

        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= num_scenarios:
                    break

                try:
                    scenario = parse_line(line)
                    print(f"\nScenario {i+1}:")
                    print("-" * 50)

                    # Print question
                    print("Question:", scenario.get("decision_prompt", "N/A"))

                    # Print correct answer and model's decision
                    print("\nCorrect Answer:", scenario.get("correct_label", "N/A"))
                    decision_output = scenario.get("decision_output", "")
                    # Extract only the choice part (e.g., "E)" from "E) friend's house")
                    if ")" in decision_output:
                        choice = decision_output.split(")", 1)[0] + ")"
                        print("Model's Decision:", choice.strip())
                    else:
                        print("Model's Decision:", decision_output)

                    explanation_ranks = compute_explanation_ranks(scenario)

                    print("\nExplanations and Scores (sorted by Spearman correlation):")
                    for j, score_info in enumerate(explanation_ranks):
                        print(f"\nExplanation {j+1}:")
                        print(f"Text: {score_info['text']}")
                        print(
                            f"Spearman Correlation: {score_info['spearman']:.4f}"
                            if score_info["spearman"] is not None
                            else "Spearman Correlation: N/A"
                        )
                        print(
                            f"Cosine Similarity: {score_info['cosine']:.4f}"
                            if score_info["cosine"] is not None
                            else "Cosine Similarity: N/A"
                        )

                    print("\n" + "=" * 50)

                except Exception as e:
                    print(f"Error processing scenario {i+1}: {e}")
                    continue
    finally:
        # Restore stdout and close file if it was opened
        if output_file:
            sys.stdout = original_stdout
            f.close()


if __name__ == "__main__":
    # file_path = "data/collection_data/codah/unsloth_Llama-3.2-3B-Instruct/codah_20250506_085629_LIME_llama3.2/codah_20250506_085629_LIME_llama3.2.jsonl"
    file_path = "data/collection_data/codah/unsloth_Llama-3.2-3B-Instruct/codah_20250506_085629_LIME_llama3.2/codah_20250506_085629_LIME_llama3.2.jsonl"

    # Print overall metrics
    metrics = analyze_dataset(file_path)
    print_metrics(metrics)

    # Print detailed scenario information to console and file
    # print_scenario_details(file_path, num_scenarios=20)
    # print_scenario_details(file_path, num_scenarios=20, output_file="outputs/scenario_details.txt")
