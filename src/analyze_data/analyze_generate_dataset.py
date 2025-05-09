import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.collect_data.comp_similarity_scores import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
)


def extract_choice(output: str) -> str:
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


def parse_line(line: str) -> Dict[str, Any]:
    """
    Parse a line from the dataset file, handling both JSON and Python dictionary formats.

    Args:
        line: A line from the dataset file

    Returns:
        Parsed dictionary from the line

    Raises:
        ValueError: If the line cannot be parsed in either format
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")

    # Try parsing as JSON first
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # If JSON parsing fails, try parsing as Python dictionary
        try:
            return ast.literal_eval(line)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse line as JSON or Python dict: {e}")


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

    # Per-scenario statistics
    scenario_spearman_min = []
    scenario_spearman_median = []
    scenario_spearman_max = []
    scenario_cosine_min = []
    scenario_cosine_median = []
    scenario_cosine_max = []

    # Read and process the dataset
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                scenario = parse_line(line)
                total_scenarios += 1

                # Check accuracy using the same method as collection phase
                correct_choice = extract_choice(scenario.get("correct_label", ""))
                decision_choice = extract_choice(scenario.get("decision_output", ""))
                if correct_choice and decision_choice and correct_choice == decision_choice:
                    correct_decisions += 1

                # Process explanation attributions
                if "explanation_attributions" in scenario and "decision_attributions" in scenario:
                    decision_attr = scenario["decision_attributions"]
                    explanation_attrs = scenario["explanation_attributions"]

                    # Calculate scores for each explanation
                    scenario_spearman = []
                    scenario_cosine = []

                    for expl_attr in explanation_attrs:
                        # Calculate Spearman correlation
                        from src.collect_data.comp_similarity_scores import (
                            calculate_spearman_correlation,
                        )

                        spearman_score = calculate_spearman_correlation(decision_attr, expl_attr)
                        if spearman_score is not None:
                            scenario_spearman.append(spearman_score)
                            spearman_scores.append(spearman_score)

                        # Calculate Cosine similarity
                        from src.collect_data.comp_similarity_scores import (
                            calculate_cosine_similarity,
                        )

                        cosine_score = calculate_cosine_similarity(decision_attr, expl_attr)
                        if cosine_score is not None:
                            scenario_cosine.append(cosine_score)
                            cosine_scores.append(cosine_score)

                    # Add per-scenario statistics
                    if scenario_spearman:
                        scenario_spearman_min.append(min(scenario_spearman))
                        scenario_spearman_median.append(np.median(scenario_spearman))
                        scenario_spearman_max.append(max(scenario_spearman))

                    if scenario_cosine:
                        scenario_cosine_min.append(min(scenario_cosine))
                        scenario_cosine_median.append(np.median(scenario_cosine))
                        scenario_cosine_max.append(max(scenario_cosine))

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
        # Spearman correlation statistics
        "spearman": {
            "worst": {
                "mean": np.mean(scenario_spearman_min) if scenario_spearman_min else 0,
                "median": np.median(scenario_spearman_min) if scenario_spearman_min else 0,
                "min": min(scenario_spearman_min) if scenario_spearman_min else 0,
                "max": max(scenario_spearman_min) if scenario_spearman_min else 0,
            },
            "median": {
                "mean": np.mean(scenario_spearman_median) if scenario_spearman_median else 0,
                "median": np.median(scenario_spearman_median) if scenario_spearman_median else 0,
                "min": min(scenario_spearman_median) if scenario_spearman_median else 0,
                "max": max(scenario_spearman_median) if scenario_spearman_median else 0,
            },
            "best": {
                "mean": np.mean(scenario_spearman_max) if scenario_spearman_max else 0,
                "median": np.median(scenario_spearman_max) if scenario_spearman_max else 0,
                "min": min(scenario_spearman_max) if scenario_spearman_max else 0,
                "max": max(scenario_spearman_max) if scenario_spearman_max else 0,
            },
        },
        # Cosine similarity statistics
        "cosine": {
            "worst": {
                "mean": np.mean(scenario_cosine_min) if scenario_cosine_min else 0,
                "median": np.median(scenario_cosine_min) if scenario_cosine_min else 0,
                "min": min(scenario_cosine_min) if scenario_cosine_min else 0,
                "max": max(scenario_cosine_min) if scenario_cosine_min else 0,
            },
            "median": {
                "mean": np.mean(scenario_cosine_median) if scenario_cosine_median else 0,
                "median": np.median(scenario_cosine_median) if scenario_cosine_median else 0,
                "min": min(scenario_cosine_median) if scenario_cosine_median else 0,
                "max": max(scenario_cosine_median) if scenario_cosine_median else 0,
            },
            "best": {
                "mean": np.mean(scenario_cosine_max) if scenario_cosine_max else 0,
                "median": np.median(scenario_cosine_max) if scenario_cosine_max else 0,
                "min": min(scenario_cosine_max) if scenario_cosine_max else 0,
                "max": max(scenario_cosine_max) if scenario_cosine_max else 0,
            },
        },
    }

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print the metrics in a readable format."""
    print("\n=== Dataset Analysis Results ===")
    print(f"Total scenarios analyzed: {metrics['total_scenarios']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")

    print("\n=== Spearman Correlation Statistics ===")
    for category in ["worst", "median", "best"]:
        print(f"\n{category.title()} values:")
        stats = metrics["spearman"][category]
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n=== Cosine Similarity Statistics ===")
    for category in ["worst", "median", "best"]:
        print(f"\n{category.title()} values:")
        stats = metrics["cosine"][category]
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")


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

                    # Print explanations and their scores
                    if "explanation_attributions" in scenario and "decision_attributions" in scenario:
                        decision_attr = scenario["decision_attributions"]
                        explanation_attrs = scenario["explanation_attributions"]

                        # Calculate scores for all explanations first
                        explanation_scores = []
                        for j, expl_attr in enumerate(explanation_attrs):
                            spearman_score = calculate_spearman_correlation(decision_attr, expl_attr)
                            cosine_score = calculate_cosine_similarity(decision_attr, expl_attr)

                            explanation_text = (
                                scenario.get("explanation_outputs", ["N/A"])[j]
                                if "explanation_outputs" in scenario
                                else scenario.get("explanation", "N/A")
                            )

                            explanation_scores.append(
                                {
                                    "index": j,
                                    "text": explanation_text,
                                    "spearman": spearman_score,
                                    "cosine": cosine_score,
                                }
                            )

                        # Sort explanations by Spearman correlation score
                        explanation_scores.sort(
                            key=lambda x: x["spearman"] if x["spearman"] is not None else float("-inf"), reverse=True
                        )

                        print("\nExplanations and Scores (sorted by Spearman correlation):")
                        for j, score_info in enumerate(explanation_scores):
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
    # import argparse

    # parser = argparse.ArgumentParser(description="Analyze a dataset file and compute metrics")
    # parser.add_argument("file_path", type=str, help="Path to the dataset file (JSONL or Python dict format)")
    # args = parser.parse_args()

    file_path = "data/collection_data/ecqa/unsloth_Qwen2.5-7B-Instruct/ecqa_20250405_155841_LIME_Qwen2.5/ecqa_20250405_155841_LIME_Qwen2.5_fixed.jsonl"

    # Print overall metrics
    metrics = analyze_dataset(file_path)
    print_metrics(metrics)

    # Print detailed scenario information to console and file
    print_scenario_details(file_path, num_scenarios=20)  # Print to console
    print_scenario_details(file_path, num_scenarios=20, output_file="outputs/scenario_details.txt")  # Print to file
