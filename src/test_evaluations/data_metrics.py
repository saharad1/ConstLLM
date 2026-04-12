import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from src.collect_data.comp_similarity_scores import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
)


@dataclass
class ScenarioData:
    """Class to store data for a single scenario."""

    scenario_id: int
    decision_prompt: str = ""
    decision_output: str = ""
    correct_label: str = ""
    explanation_prompt: str = ""
    explanation_outputs: List[str] = field(default_factory=list)
    spearman_scores: List[float] = field(default_factory=list)
    cosine_scores: List[float] = field(default_factory=list)

    @property
    def avg_spearman(self) -> float:
        """Average Spearman correlation across all explanations."""
        if not self.spearman_scores:
            return 0.0
        return np.mean(self.spearman_scores)

    @property
    def avg_cosine(self) -> float:
        """Average cosine similarity across all explanations."""
        if not self.cosine_scores:
            return 0.0
        return np.mean(self.cosine_scores)


def create_scenarios_data(dataset_path):
    """
    Analyze both Spearman and cosine similarity values in a dataset.

    Args:
        dataset_path: Path to the JSONL file containing the dataset

    Returns:
        List of ScenarioData objects containing similarity metrics
    """
    scenario_data_list = []

    # Process the JSONL file line by line
    with open(dataset_path, "r") as f:
        for line_idx, line in enumerate(f):
            try:
                # Skip empty lines
                if not line.strip():
                    continue

                # Parse the JSON object
                item = json.loads(line.strip())

                # Get scenario_id or use line index as fallback
                scenario_id = item.get("scenario_id", line_idx)
                if isinstance(scenario_id, str):
                    try:
                        scenario_id = int(scenario_id)
                    except ValueError:
                        scenario_id = line_idx

                # Create a ScenarioData object
                scenario = ScenarioData(
                    scenario_id=scenario_id,
                    decision_prompt=item.get("decision_prompt", ""),
                    decision_output=item.get("decision_output", ""),
                    correct_label=item.get("correct_label", ""),
                    explanation_prompt=item.get("explanation_prompt", ""),
                    explanation_outputs=item.get("explanation_outputs", []),
                )

                # Process attributions if they exist
                if "decision_attributions" in item and "explanation_attributions" in item:
                    decision_attr = item["decision_attributions"]
                    explanation_attrs = item["explanation_attributions"]

                    # Ensure explanation_attributions is a list
                    if not isinstance(explanation_attrs, list):
                        explanation_attrs = [explanation_attrs]

                    # Process each explanation
                    for expl_idx, explanation_attr in enumerate(explanation_attrs):
                        try:
                            # Calculate Spearman correlation
                            spearman_score = calculate_spearman_correlation(decision_attr, explanation_attr)
                            if spearman_score is not None:
                                scenario.spearman_scores.append(spearman_score)

                            # Calculate Cosine similarity
                            cosine_score = calculate_cosine_similarity(decision_attr, explanation_attr)
                            if cosine_score is not None:
                                scenario.cosine_scores.append(cosine_score)
                        except Exception as e:
                            print(f"Error calculating similarity for scenario {scenario_id}, explanation {expl_idx}: {str(e)}")

                # Add scenario to the list
                scenario_data_list.append(scenario)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_idx+1}: {str(e)}")
            except Exception as e:
                print(f"Error processing line {line_idx+1}: {str(e)}")

    print(f"Successfully processed {len(scenario_data_list)} scenarios from {dataset_path}")
    return scenario_data_list


def compute_and_display_statistics(scenario_data_list):
    """
    Compute and display statistics for similarity scores across the dataset.

    Args:
        scenario_data_list: List of ScenarioData objects

    Returns:
        Tuple of (spearman_array, cosine_array) containing all scores
    """
    if not scenario_data_list:
        print("No scenarios found in the dataset.")
        return None, None

    print(f"Analyzing {len(scenario_data_list)} scenarios from the dataset.")

    # Extract all scores for statistics and plotting
    all_spearman_scores = []
    all_cosine_scores = []

    # Per-scenario min, median, and max values
    scenario_min_spearman = []
    scenario_median_spearman = []
    scenario_max_spearman = []
    scenario_avg_spearman = []

    scenario_min_cosine = []
    scenario_median_cosine = []
    scenario_max_cosine = []
    scenario_avg_cosine = []

    for scenario in scenario_data_list:
        # Add all scores to global lists
        all_spearman_scores.extend(scenario.spearman_scores)
        all_cosine_scores.extend(scenario.cosine_scores)

        # Calculate per-scenario statistics if there are scores
        if scenario.spearman_scores:
            scenario_min_spearman.append(min(scenario.spearman_scores))
            scenario_median_spearman.append(np.median(scenario.spearman_scores))
            scenario_max_spearman.append(max(scenario.spearman_scores))
            scenario_avg_spearman.append(scenario.avg_spearman)

        if scenario.cosine_scores:
            scenario_min_cosine.append(min(scenario.cosine_scores))
            scenario_median_cosine.append(np.median(scenario.cosine_scores))
            scenario_max_cosine.append(max(scenario.cosine_scores))
            scenario_avg_cosine.append(scenario.avg_cosine)

    # Convert to numpy arrays
    spearman_array = np.array(all_spearman_scores)
    cosine_array = np.array(all_cosine_scores)

    # Calculate and print Spearman statistics
    print("\n=== SPEARMAN SIMILARITY ANALYSIS ===")
    print(f"Number of samples: {len(all_spearman_scores)}")
    print(f"Min similarity: {np.min(spearman_array):.4f}")
    print(f"Median similarity: {np.median(spearman_array):.4f}")
    print(f"Max similarity: {np.max(spearman_array):.4f}")
    print(f"Mean similarity: {np.mean(spearman_array):.4f}")
    print(f"Standard deviation: {np.std(spearman_array):.4f}")

    # Calculate and print Cosine statistics
    print("\n=== COSINE SIMILARITY ANALYSIS ===")
    print(f"Number of samples: {len(all_cosine_scores)}")
    print(f"Min similarity: {np.min(cosine_array):.4f}")
    print(f"Median similarity: {np.median(cosine_array):.4f}")
    print(f"Max similarity: {np.max(cosine_array):.4f}")
    print(f"Mean similarity: {np.mean(cosine_array):.4f}")
    print(f"Standard deviation: {np.std(cosine_array):.4f}")

    # Print per-scenario statistics
    print("\n=== PER-SCENARIO STATISTICS ===")

    print("\nSpearman correlation:")
    print("  Worst values across scenarios:")
    print(f"    Mean of worst: {np.mean(scenario_min_spearman):.4f}")

    print("  Median values across scenarios:")
    print(f"    Mean of medians: {np.mean(scenario_median_spearman):.4f}")
    # print(f"    Max of medians: {np.max(scenario_median_spearman):.4f}")

    print("  Best values across scenarios:")
    print(f"    Mean of best: {np.mean(scenario_max_spearman):.4f}")
    # print(f"    Max of best: {np.max(scenario_max_spearman):.4f}")

    print("\nCosine similarity:")
    print("  Worst values across scenarios:")
    # print(f"    Min of mins: {np.min(scenario_min_cosine):.4f}")
    # print(f"    Median of mins: {np.median(scenario_min_cosine):.4f}")
    print(f"    Mean of worst: {np.mean(scenario_min_cosine):.4f}")
    # print(f"    Max of mins: {np.max(scenario_min_cosine):.4f}")

    print("  Median values across scenarios:")
    print(f"    Mean of medians: {np.mean(scenario_median_cosine):.4f}")
    # print(f"    Max of medians: {np.max(scenario_median_cosine):.4f}")

    print("  Best values across scenarios:")
    # print(f"    Min of maxes: {np.min(scenario_max_cosine):.4f}")
    # print(f"    Median of maxes: {np.median(scenario_max_cosine):.4f}")
    print(f"    Mean of best: {np.mean(scenario_max_cosine):.4f}")
    # print(f"    Max of maxes: {np.max(scenario_max_cosine):.4f}")

    # print("  Average values across scenarios:")
    # print(f"    Min average: {np.min(scenario_avg_cosine):.4f}")
    # print(f"    Median average: {np.median(scenario_avg_cosine):.4f}")
    # print(f"    Mean of averages: {np.mean(scenario_avg_cosine):.4f}")
    # print(f"    Max average: {np.max(scenario_avg_cosine):.4f}")

    return spearman_array, cosine_array


if __name__ == "__main__":
    # Path to the dataset
    dataset_path = Path("dpo_datasets/cleaned_ecqa_dpo_datasets/cleaned_ecqa_250221_181714_LIME/test_1089.jsonl")

    # Analyze both similarity metrics
    scenario_data_list = create_scenarios_data(dataset_path)
    compute_and_display_statistics(scenario_data_list)
