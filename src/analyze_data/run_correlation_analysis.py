#!/usr/bin/env python3
"""
Standalone script to run explanation ranking correlation analysis.
This script is called by the bash wrapper script.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from src.analyze_data.explanation_ranking_correlation import (
    compare_dataset_rankings,
    compare_dataset_rankings_multiple_metrics,
    plot_correlation_distribution,
    plot_multiple_metrics_distribution,
    print_correlation_results,
    print_multiple_metrics_results,
)


def main():
    parser = argparse.ArgumentParser(description="Calculate explanation ranking correlation between two datasets")
    parser.add_argument("dataset1_path", help="Path to the first dataset (JSONL file)")
    parser.add_argument("dataset2_path", help="Path to the second dataset (JSONL file)")
    parser.add_argument("metric_type", default="spearman", help="Ranking metric to use (spearman, cosine, lma)")
    parser.add_argument("output_prefix", help="Output file prefix for results")

    args = parser.parse_args()

    dataset1_path = args.dataset1_path
    dataset2_path = args.dataset2_path
    metric_type = args.metric_type
    output_prefix = args.output_prefix

    print(f"Starting correlation analysis...")
    print(f"Dataset 1: {dataset1_path}")
    print(f"Dataset 2: {dataset2_path}")
    print(f"Metric: {metric_type}")

    try:
        # Perform the correlation analysis with multiple metrics
        results = compare_dataset_rankings_multiple_metrics(dataset1_path, dataset2_path, metric_type)

        # Print results to console
        print_multiple_metrics_results(results)

        # Save detailed results to text file
        text_output = f"{output_prefix}_analysis.txt"
        with open(text_output, "w") as f:
            # Redirect stdout to file
            original_stdout = sys.stdout
            sys.stdout = f
            print_multiple_metrics_results(results)
            sys.stdout = original_stdout

        # Save summary to JSON file
        json_output = f"{output_prefix}_summary.json"
        summary = {
            "dataset1_path": results["dataset1_path"],
            "dataset2_path": results["dataset2_path"],
            "metric_type": results["metric_type"],
            "overall_stats": results["overall_stats"],
            "num_scenarios": len(results["scenario_correlations"]),
        }

        with open(json_output, "w") as f:
            json.dump(summary, f, indent=2)

        # Generate and save plot
        plot_output = f"{output_prefix}_plot.png"
        plot_multiple_metrics_distribution(results, save_path=plot_output)

        print(f"\nResults saved to:")
        print(f"  - Text analysis: {text_output}")
        print(f"  - JSON summary: {json_output}")
        print(f"  - Plot: {plot_output}")

        # Print summary statistics
        stats = results["overall_stats"]
        print(f"\nSummary:")
        print(f"  Number of scenarios: {len(results['scenario_correlations'])}")

        # Print key metrics summary
        key_metrics = [
            "spearman",
            "pearson",
            "kendall",
            "concordant_pairs_ratio",
            "top_1_overlap",
            "top_2_overlap",
            "top_3_overlap",
            "top_5_overlap",
        ]
        for metric in key_metrics:
            if metric in stats and stats[metric]["num_valid"] > 0:
                metric_stats = stats[metric]
                print(f"  {metric.capitalize()} - Mean: {metric_stats['mean']:.4f}, Std: {metric_stats['std']:.4f}")
            else:
                print(f"  {metric.capitalize()} - No valid data")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
