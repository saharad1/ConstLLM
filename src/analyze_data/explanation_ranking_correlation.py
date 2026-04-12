"""
Module for analyzing correlation between explanation rankings from two datasets.

This module provides functionality to:
1. Load and parse datasets
2. Compute explanation rankings based on Spearman correlation scores
3. Calculate correlation between rankings from two datasets
4. Test functionality using the same dataset to check per-scenario correlation
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr

from src.analyze_data.analysis_utils import compute_explanation_ranks, parse_line
from src.collect_data.comp_similarity_scores import calculate_spearman_correlation


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from a JSONL file.

    Args:
        file_path: Path to the dataset file

    Returns:
        List of scenario dictionaries
    """
    scenarios = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                scenario = parse_line(line)
                scenarios.append(scenario)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    return scenarios


def compute_explanation_ranking(scenario: Dict[str, Any], metric_type: str = "spearman") -> List[int]:
    """
    Compute explanation ranking for a single scenario.

    Args:
        scenario: Scenario dictionary containing explanation data
        metric_type: Metric to use for ranking ("spearman", "cosine", "lma")

    Returns:
        List of explanation indices sorted by ranking (best to worst)
    """
    explanation_ranks = compute_explanation_ranks(scenario, metric_type)
    return [rank_info["index"] for rank_info in explanation_ranks]


def compute_dataset_rankings(dataset: List[Dict[str, Any]], metric_type: str = "spearman") -> List[List[int]]:
    """
    Compute explanation rankings for all scenarios in a dataset.

    Args:
        dataset: List of scenario dictionaries
        metric_type: Metric to use for ranking ("spearman", "cosine", "lma")

    Returns:
        List of rankings, where each ranking is a list of explanation indices
    """
    rankings = []
    for scenario in dataset:
        ranking = compute_explanation_ranking(scenario, metric_type)
        if ranking:  # Only include scenarios with valid rankings
            rankings.append(ranking)
    return rankings


def calculate_ranking_correlation(ranking1: List[int], ranking2: List[int]) -> Tuple[float, float]:
    """
    Calculate Spearman correlation between two explanation rankings.

    Args:
        ranking1: First ranking (list of explanation indices)
        ranking2: Second ranking (list of explanation indices)

    Returns:
        Tuple of (correlation_coefficient, p_value)
    """
    if len(ranking1) != len(ranking2):
        raise ValueError("Rankings must have the same length")

    # Convert rankings to rank positions (1-based)
    rank1_positions = [ranking1.index(i) + 1 for i in range(len(ranking1))]
    rank2_positions = [ranking2.index(i) + 1 for i in range(len(ranking2))]

    correlation, p_value = spearmanr(rank1_positions, rank2_positions)
    return correlation, p_value


def calculate_multiple_ranking_correlations(ranking1: List[int], ranking2: List[int]) -> Dict[str, Any]:
    """
    Calculate multiple correlation metrics between two explanation rankings.

    Args:
        ranking1: First ranking (list of explanation indices)
        ranking2: Second ranking (list of explanation indices)

    Returns:
        Dictionary containing various correlation metrics and their p-values
    """
    if len(ranking1) != len(ranking2):
        raise ValueError("Rankings must have the same length")

    # Convert rankings to rank positions (1-based)
    rank1_positions = [ranking1.index(i) + 1 for i in range(len(ranking1))]
    rank2_positions = [ranking2.index(i) + 1 for i in range(len(ranking2))]

    # Convert to numpy arrays for easier computation
    rank1_array = np.array(rank1_positions)
    rank2_array = np.array(rank2_positions)

    results = {}

    # 1. Spearman correlation (rank-based)
    spearman_corr, spearman_p = spearmanr(rank1_array, rank2_array)
    results["spearman"] = {"correlation": spearman_corr, "p_value": spearman_p}

    # 2. Pearson correlation (linear relationship)
    pearson_corr, pearson_p = pearsonr(rank1_array, rank2_array)
    results["pearson"] = {"correlation": pearson_corr, "p_value": pearson_p}

    # 3. Kendall's tau (rank-based, more robust to ties)
    kendall_corr, kendall_p = kendalltau(rank1_array, rank2_array)
    results["kendall"] = {"correlation": kendall_corr, "p_value": kendall_p}

    # 4. Rank agreement (proportion of items in same relative position)
    # This measures how many pairs of explanations maintain the same relative ordering
    # between the two rankings. For example, if explanation A is ranked higher than
    # explanation B in both rankings, that's a concordant pair.
    #
    # Example: Ranking1 = [A, B, C], Ranking2 = [A, C, B]
    # Pairs: (A,B), (A,C), (B,C)
    # - (A,B): A < B in both rankings → concordant
    # - (A,C): A < C in both rankings → concordant
    # - (B,C): B < C in ranking1, but C < B in ranking2 → discordant
    # Rank agreement = 2/3 = 0.67
    n = len(rank1_array)
    concordant_pairs = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Check if relative ordering is the same
            rank1_order = rank1_array[i] < rank1_array[j]
            rank2_order = rank2_array[i] < rank2_array[j]
            if rank1_order == rank2_order:
                concordant_pairs += 1
            total_pairs += 1

    rank_agreement = concordant_pairs / total_pairs if total_pairs > 0 else 0
    results["concordant_pairs_ratio"] = {
        "ratio": rank_agreement,
        "concordant_pairs": concordant_pairs,
        "total_pairs": total_pairs,
    }

    # 5. Normalized rank distance (1 - normalized distance)
    # Convert to 0-1 scale and calculate normalized distance
    rank1_norm = (rank1_array - 1) / (n - 1)  # Normalize to 0-1
    rank2_norm = (rank2_array - 1) / (n - 1)  # Normalize to 0-1

    rank_distance = np.mean(np.abs(rank1_norm - rank2_norm))
    normalized_rank_similarity = 1 - rank_distance  # Convert distance to similarity
    results["normalized_rank_similarity"] = {"similarity": normalized_rank_similarity, "distance": rank_distance}

    # 6. Top-k overlap (for k=1, 2, 3, 5)
    for k in [1, 2, 3, 5]:
        if k <= n:
            top_k_1 = set(ranking1[:k])  # Top k from ranking 1
            top_k_2 = set(ranking2[:k])  # Top k from ranking 2
            overlap = len(top_k_1 & top_k_2) / k
            results[f"top_{k}_overlap"] = {"overlap": overlap, "intersection": top_k_1 & top_k_2}

    return results


def verify_scenario_matching(dataset1_dict: Dict[str, Any], dataset2_dict: Dict[str, Any], common_ids: set) -> None:
    """
    Verify that scenario matching is done correctly by checking scenario IDs.

    Args:
        dataset1_dict: Dictionary mapping scenario_id to scenario data from dataset 1
        dataset2_dict: Dictionary mapping scenario_id to scenario data from dataset 2
        common_ids: Set of common scenario IDs between datasets
    """
    print(f"Verifying scenario matching for {len(common_ids)} common scenarios...")

    matched_scenarios1 = [dataset1_dict[scenario_id] for scenario_id in sorted(common_ids)]
    matched_scenarios2 = [dataset2_dict[scenario_id] for scenario_id in sorted(common_ids)]

    mismatches = 0
    for i, (scenario_id, s1, s2) in enumerate(zip(sorted(common_ids), matched_scenarios1, matched_scenarios2)):
        if s1.get("scenario_id") != s2.get("scenario_id"):
            print(f"ERROR: Mismatch at index {i}: {s1.get('scenario_id')} != {s2.get('scenario_id')}")
            mismatches += 1
        elif i < 3:  # Show first 3 for verification
            print(f"  Scenario {i+1}: ID {scenario_id} - Both datasets have same scenario")

    if mismatches == 0:
        print(f"✅ All {len(common_ids)} scenarios are properly matched by ID")
    else:
        print(f"❌ Found {mismatches} mismatches in scenario matching")


def compare_dataset_rankings_multiple_metrics(
    dataset1_path: str, dataset2_path: str, metric_type: str = "spearman"
) -> Dict[str, Any]:
    """
    Compare explanation rankings between two datasets using multiple correlation metrics.

    Args:
        dataset1_path: Path to first dataset
        dataset2_path: Path to second dataset
        metric_type: Metric to use for ranking ("spearman", "cosine", "lma")

    Returns:
        Dictionary containing correlation analysis results with multiple metrics
    """
    # Load datasets
    dataset1 = load_dataset(dataset1_path)
    dataset2 = load_dataset(dataset2_path)

    print(f"Loaded {len(dataset1)} scenarios from dataset 1")
    print(f"Loaded {len(dataset2)} scenarios from dataset 2")

    # Create dictionaries mapping scenario_id to scenario data
    dataset1_dict = {scenario.get("scenario_id"): scenario for scenario in dataset1 if "scenario_id" in scenario}
    dataset2_dict = {scenario.get("scenario_id"): scenario for scenario in dataset2 if "scenario_id" in scenario}

    # Find common scenario IDs
    common_ids = set(dataset1_dict.keys()) & set(dataset2_dict.keys())
    print(f"Found {len(common_ids)} common scenario IDs between datasets")

    # Verify scenario matching
    verify_scenario_matching(dataset1_dict, dataset2_dict, common_ids)

    if len(common_ids) == 0:
        print("Warning: No common scenario IDs found between datasets!")
        return {
            "dataset1_path": dataset1_path,
            "dataset2_path": dataset2_path,
            "metric_type": metric_type,
            "overall_stats": {},
            "scenario_correlations": [],
            "valid_correlations": {},
        }

    # Compute rankings for matched scenarios
    matched_scenarios1 = [dataset1_dict[scenario_id] for scenario_id in sorted(common_ids)]
    matched_scenarios2 = [dataset2_dict[scenario_id] for scenario_id in sorted(common_ids)]

    rankings1 = compute_dataset_rankings(matched_scenarios1, metric_type)
    rankings2 = compute_dataset_rankings(matched_scenarios2, metric_type)

    print(f"Computed {len(rankings1)} valid rankings from dataset 1")
    print(f"Computed {len(rankings2)} valid rankings from dataset 2")

    # Calculate correlations for each matched scenario
    scenario_correlations = []
    valid_correlations = {
        "spearman": [],
        "pearson": [],
        "kendall": [],
        "concordant_pairs_ratio": [],
        "normalized_rank_similarity": [],
        "top_1_overlap": [],
        "top_2_overlap": [],
        "top_3_overlap": [],
        "top_5_overlap": [],
    }
    sorted_common_ids = sorted(common_ids)

    for i, (scenario_id, rank1, rank2) in enumerate(zip(sorted_common_ids, rankings1, rankings2)):
        try:
            correlations = calculate_multiple_ranking_correlations(rank1, rank2)
            scenario_correlations.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_index": i,
                    "correlations": correlations,
                    "ranking1": rank1,
                    "ranking2": rank2,
                }
            )

            # Collect valid correlations for each metric
            for metric_name, metric_data in correlations.items():
                if metric_name in ["spearman", "pearson", "kendall"]:
                    if not np.isnan(metric_data["correlation"]):
                        valid_correlations[metric_name].append(metric_data["correlation"])
                elif metric_name in ["concordant_pairs_ratio", "normalized_rank_similarity"]:
                    if not np.isnan(metric_data["ratio"] if "ratio" in metric_data else metric_data["similarity"]):
                        key = "ratio" if "ratio" in metric_data else "similarity"
                        valid_correlations[metric_name].append(metric_data[key])
                elif metric_name.startswith("top_"):
                    if not np.isnan(metric_data["overlap"]):
                        valid_correlations[metric_name].append(metric_data["overlap"])

        except Exception as e:
            print(f"Error calculating correlations for scenario ID {scenario_id}: {e}")
            continue

    # Calculate overall statistics for each metric
    overall_stats = {}
    for metric_name, values in valid_correlations.items():
        if values:
            overall_stats[metric_name] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "num_valid": len(values),
            }
        else:
            overall_stats[metric_name] = {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "num_valid": 0,
            }

    return {
        "dataset1_path": dataset1_path,
        "dataset2_path": dataset2_path,
        "metric_type": metric_type,
        "overall_stats": overall_stats,
        "scenario_correlations": scenario_correlations,
        "valid_correlations": valid_correlations,
    }


def compare_dataset_rankings(dataset1_path: str, dataset2_path: str, metric_type: str = "spearman") -> Dict[str, Any]:
    """
    Compare explanation rankings between two datasets, matching scenarios by ID.

    Args:
        dataset1_path: Path to first dataset
        dataset2_path: Path to second dataset
        metric_type: Metric to use for ranking ("spearman", "cosine", "lma")

    Returns:
        Dictionary containing correlation analysis results
    """
    # Load datasets
    dataset1 = load_dataset(dataset1_path)
    dataset2 = load_dataset(dataset2_path)

    print(f"Loaded {len(dataset1)} scenarios from dataset 1")
    print(f"Loaded {len(dataset2)} scenarios from dataset 2")

    # Create dictionaries mapping scenario_id to scenario data
    dataset1_dict = {scenario.get("scenario_id"): scenario for scenario in dataset1 if "scenario_id" in scenario}
    dataset2_dict = {scenario.get("scenario_id"): scenario for scenario in dataset2 if "scenario_id" in scenario}

    # Find common scenario IDs
    common_ids = set(dataset1_dict.keys()) & set(dataset2_dict.keys())
    print(f"Found {len(common_ids)} common scenario IDs between datasets")

    if len(common_ids) == 0:
        print("Warning: No common scenario IDs found between datasets!")
        return {
            "dataset1_path": dataset1_path,
            "dataset2_path": dataset2_path,
            "metric_type": metric_type,
            "overall_stats": {
                "mean_correlation": None,
                "median_correlation": None,
                "std_correlation": None,
                "min_correlation": None,
                "max_correlation": None,
                "num_valid_correlations": 0,
                "num_total_scenarios": 0,
            },
            "scenario_correlations": [],
            "valid_correlations": [],
        }

    # Compute rankings for matched scenarios
    matched_scenarios1 = [dataset1_dict[scenario_id] for scenario_id in sorted(common_ids)]
    matched_scenarios2 = [dataset2_dict[scenario_id] for scenario_id in sorted(common_ids)]

    rankings1 = compute_dataset_rankings(matched_scenarios1, metric_type)
    rankings2 = compute_dataset_rankings(matched_scenarios2, metric_type)

    print(f"Computed {len(rankings1)} valid rankings from dataset 1")
    print(f"Computed {len(rankings2)} valid rankings from dataset 2")

    # Calculate correlations for each matched scenario
    scenario_correlations = []
    valid_correlations = []
    sorted_common_ids = sorted(common_ids)

    for i, (scenario_id, rank1, rank2) in enumerate(zip(sorted_common_ids, rankings1, rankings2)):
        try:
            correlation, p_value = calculate_ranking_correlation(rank1, rank2)
            scenario_correlations.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_index": i,
                    "correlation": correlation,
                    "p_value": p_value,
                    "ranking1": rank1,
                    "ranking2": rank2,
                }
            )
            if not np.isnan(correlation):
                valid_correlations.append(correlation)
        except Exception as e:
            print(f"Error calculating correlation for scenario ID {scenario_id}: {e}")
            continue

    # Calculate overall statistics
    if valid_correlations:
        overall_stats = {
            "mean_correlation": np.mean(valid_correlations),
            "median_correlation": np.median(valid_correlations),
            "std_correlation": np.std(valid_correlations),
            "min_correlation": np.min(valid_correlations),
            "max_correlation": np.max(valid_correlations),
            "num_valid_correlations": len(valid_correlations),
            "num_total_scenarios": len(scenario_correlations),
        }
    else:
        overall_stats = {
            "mean_correlation": None,
            "median_correlation": None,
            "std_correlation": None,
            "min_correlation": None,
            "max_correlation": None,
            "num_valid_correlations": 0,
            "num_total_scenarios": len(scenario_correlations),
        }

    return {
        "dataset1_path": dataset1_path,
        "dataset2_path": dataset2_path,
        "metric_type": metric_type,
        "overall_stats": overall_stats,
        "scenario_correlations": scenario_correlations,
        "valid_correlations": valid_correlations,
    }


def test_same_dataset_correlation(
    dataset_path: str, metric_type: str = "spearman", num_scenarios: int = None
) -> Dict[str, Any]:
    """
    Test correlation analysis using the same dataset (for validation).
    This simulates comparing a dataset with itself to check per-scenario correlation.

    Args:
        dataset_path: Path to the dataset
        metric_type: Metric to use for ranking ("spearman", "cosine", "lma")
        num_scenarios: Number of scenarios to analyze (None for all)

    Returns:
        Dictionary containing correlation analysis results
    """
    # Load dataset
    dataset = load_dataset(dataset_path)

    if num_scenarios is not None:
        dataset = dataset[:num_scenarios]

    print(f"Testing correlation with same dataset: {len(dataset)} scenarios")

    # Compute rankings
    rankings = compute_dataset_rankings(dataset, metric_type)
    print(f"Computed {len(rankings)} valid rankings")

    # Calculate correlations between consecutive scenarios
    scenario_correlations = []
    valid_correlations = []

    for i in range(len(rankings) - 1):
        try:
            rank1 = rankings[i]
            rank2 = rankings[i + 1]
            correlation, p_value = calculate_ranking_correlation(rank1, rank2)
            scenario_correlations.append(
                {
                    "scenario_pair": (i, i + 1),
                    "correlation": correlation,
                    "p_value": p_value,
                    "ranking1": rank1,
                    "ranking2": rank2,
                }
            )
            if not np.isnan(correlation):
                valid_correlations.append(correlation)
        except Exception as e:
            print(f"Error calculating correlation for scenario pair {i}-{i+1}: {e}")
            continue

    # Calculate overall statistics
    if valid_correlations:
        overall_stats = {
            "mean_correlation": np.mean(valid_correlations),
            "median_correlation": np.median(valid_correlations),
            "std_correlation": np.std(valid_correlations),
            "min_correlation": np.min(valid_correlations),
            "max_correlation": np.max(valid_correlations),
            "num_valid_correlations": len(valid_correlations),
            "num_total_pairs": len(scenario_correlations),
        }
    else:
        overall_stats = {
            "mean_correlation": None,
            "median_correlation": None,
            "std_correlation": None,
            "min_correlation": None,
            "max_correlation": None,
            "num_valid_correlations": 0,
            "num_total_pairs": len(scenario_correlations),
        }

    return {
        "dataset_path": dataset_path,
        "metric_type": metric_type,
        "test_type": "same_dataset_consecutive_pairs",
        "overall_stats": overall_stats,
        "scenario_correlations": scenario_correlations,
        "valid_correlations": valid_correlations,
    }


def print_correlation_results(results: Dict[str, Any]) -> None:
    """
    Print correlation analysis results in a readable format.

    Args:
        results: Results dictionary from compare_dataset_rankings or test_same_dataset_correlation
    """
    print("\n" + "=" * 60)
    print("EXPLANATION RANKING CORRELATION ANALYSIS")
    print("=" * 60)

    if "dataset1_path" in results:
        print(f"Dataset 1: {results['dataset1_path']}")
        print(f"Dataset 2: {results['dataset2_path']}")
    else:
        print(f"Dataset: {results['dataset_path']}")
        print(f"Test Type: {results['test_type']}")

    print(f"Metric Type: {results['metric_type']}")

    stats = results["overall_stats"]
    print(f"\nOverall Statistics:")
    print(f"  Number of valid correlations: {stats['num_valid_correlations']}")
    print(f"  Total scenarios/pairs: {stats.get('num_total_scenarios', stats.get('num_total_pairs', 'N/A'))}")

    if stats["mean_correlation"] is not None:
        print(f"  Mean correlation: {stats['mean_correlation']:.4f}")
        print(f"  Median correlation: {stats['median_correlation']:.4f}")
        print(f"  Standard deviation: {stats['std_correlation']:.4f}")
        print(f"  Min correlation: {stats['min_correlation']:.4f}")
        print(f"  Max correlation: {stats['max_correlation']:.4f}")
    else:
        print("  No valid correlations found")

    # Show some example correlations
    if results["scenario_correlations"]:
        print(f"\nExample correlations (first 5):")
        for i, corr_info in enumerate(results["scenario_correlations"][:5]):
            if "scenario_pair" in corr_info:
                print(
                    f"  Pair {corr_info['scenario_pair']}: {corr_info['correlation']:.4f} (p={corr_info['p_value']:.4f})"
                )
            elif "scenario_id" in corr_info:
                print(
                    f"  Scenario ID {corr_info['scenario_id']}: {corr_info['correlation']:.4f} (p={corr_info['p_value']:.4f})"
                )
            else:
                print(
                    f"  Scenario {corr_info['scenario_index']}: {corr_info['correlation']:.4f} (p={corr_info['p_value']:.4f})"
                )


def print_multiple_metrics_results(results: Dict[str, Any]) -> None:
    """
    Print correlation analysis results with multiple metrics in a readable format.

    Args:
        results: Results dictionary from compare_dataset_rankings_multiple_metrics
    """
    print("\n" + "=" * 80)
    print("EXPLANATION RANKING CORRELATION ANALYSIS - MULTIPLE METRICS")
    print("=" * 80)

    if "dataset1_path" in results:
        print(f"Dataset 1: {results['dataset1_path']}")
        print(f"Dataset 2: {results['dataset2_path']}")
    else:
        print(f"Dataset: {results['dataset_path']}")
        print(f"Test Type: {results['test_type']}")

    print(f"Metric Type: {results['metric_type']}")

    stats = results["overall_stats"]
    print(f"\nOverall Statistics:")

    # Print statistics for each metric
    metric_descriptions = {
        "spearman": "Spearman Correlation (rank-based)",
        "pearson": "Pearson Correlation (linear relationship)",
        "kendall": "Kendall's Tau (rank-based, robust to ties)",
        "concordant_pairs_ratio": "Concordant Pairs Ratio (proportion of pairs with same relative order)",
        "normalized_rank_similarity": "Normalized Rank Similarity (1 - distance)",
        "top_1_overlap": "Top-1 Overlap (best explanation match)",
        "top_2_overlap": "Top-2 Overlap (top 2 explanations match)",
        "top_3_overlap": "Top-3 Overlap (top 3 explanations match)",
        "top_5_overlap": "Top-5 Overlap (top 5 explanations match)",
    }

    for metric_name, description in metric_descriptions.items():
        if metric_name in stats and stats[metric_name]["num_valid"] > 0:
            metric_stats = stats[metric_name]
            print(f"\n{description}:")
            print(f"  Mean: {metric_stats['mean']:.4f}")
            print(f"  Median: {metric_stats['median']:.4f}")
            print(f"  Std Dev: {metric_stats['std']:.4f}")
            print(f"  Range: [{metric_stats['min']:.4f}, {metric_stats['max']:.4f}]")
            print(f"  Valid samples: {metric_stats['num_valid']}")
        else:
            print(f"\n{description}: No valid data")

    # Show some example correlations
    if results["scenario_correlations"]:
        print(f"\nExample correlations (first 3 scenarios):")
        for i, corr_info in enumerate(results["scenario_correlations"][:3]):
            scenario_id = corr_info.get("scenario_id", f"Scenario {i}")
            correlations = corr_info["correlations"]
            print(f"\n  Scenario ID {scenario_id}:")
            for metric_name, metric_data in correlations.items():
                if metric_name in ["spearman", "pearson", "kendall"]:
                    print(f"    {metric_name}: {metric_data['correlation']:.4f} (p={metric_data['p_value']:.4f})")
                elif metric_name in ["concordant_pairs_ratio", "normalized_rank_similarity"]:
                    key = "ratio" if "ratio" in metric_data else "similarity"
                    print(f"    {metric_name}: {metric_data[key]:.4f}")
                elif metric_name.startswith("top_"):
                    print(f"    {metric_name}: {metric_data['overlap']:.4f}")


def plot_correlation_distribution(results: Dict[str, Any], save_path: str = None) -> None:
    """
    Plot the distribution of correlation coefficients.

    Args:
        results: Results dictionary from compare_dataset_rankings or test_same_dataset_correlation
        save_path: Optional path to save the plot
    """
    if not results["valid_correlations"]:
        print("No valid correlations to plot")
        return

    plt.figure(figsize=(10, 6))

    # Create histogram
    plt.subplot(1, 2, 1)
    plt.hist(results["valid_correlations"], bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.title("Distribution of Correlation Coefficients")
    plt.grid(True, alpha=0.3)

    # Create box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(results["valid_correlations"])
    plt.ylabel("Correlation Coefficient")
    plt.title("Box Plot of Correlation Coefficients")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_multiple_metrics_distribution(results: Dict[str, Any], save_path: str = None) -> None:
    """
    Plot the distribution of multiple correlation metrics.

    Args:
        results: Results dictionary from compare_dataset_rankings_multiple_metrics
        save_path: Optional path to save the plot
    """
    valid_correlations = results["valid_correlations"]

    if not any(valid_correlations.values()):
        print("No valid correlations to plot")
        return

    # Create a large figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # Plot each metric
    metric_names = [
        "spearman",
        "pearson",
        "kendall",
        "concordant_pairs_ratio",
        "normalized_rank_similarity",
        "top_1_overlap",
        "top_2_overlap",
        "top_3_overlap",
        "top_5_overlap",
    ]

    for i, metric_name in enumerate(metric_names):
        if i >= len(axes):
            break

        if metric_name in valid_correlations and valid_correlations[metric_name]:
            values = valid_correlations[metric_name]

            # Create histogram
            axes[i].hist(values, bins=20, alpha=0.7, edgecolor="black")
            axes[i].set_xlabel(f"{metric_name.replace('_', ' ').title()}")
            axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"{metric_name.replace('_', ' ').title()}\n(Mean: {np.mean(values):.3f})")
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f"No data for\n{metric_name}", ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_title(f"{metric_name.replace('_', ' ').title()}")

    # Hide unused subplots
    for i in range(len(metric_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage - test with the same dataset
    dataset_path = "./data/collection_data/arc_challenge/unsloth_Llama-3.2-3B-Instruct/arc_challenge_20250421_094925_LIME_llama3.2/test_255.jsonl"

    print("Testing explanation ranking correlation analysis...")

    # Test with same dataset (consecutive scenario pairs)
    results = test_same_dataset_correlation(dataset_path, metric_type="spearman", num_scenarios=20)
    print_correlation_results(results)

    # Plot the results
    plot_correlation_distribution(results, save_path="explanation_ranking_correlation_test.png")
