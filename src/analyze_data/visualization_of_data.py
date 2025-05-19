import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analyze_data.analysis_utils import compute_explanation_ranks


def analyze_metric_differences(jsonl_file, metric_type="cosine", num_bins=8, dataset_name="", output_dir=None):
    """
    Analyze and visualize the differences between best and worst scores for either cosine similarity
    or Spearman correlation from a JSONL file containing scenario data.

    Parameters:
    -----------
    jsonl_file : str
        Path to the JSONL file containing scenario data
    metric_type : str, optional
        Type of metric to analyze ('cosine' or 'spearman', default: 'cosine')
    num_bins : int, optional
        Number of bins for the histogram (default: 8)
    dataset_name : str, optional
        Name of the dataset for the plot title
    output_dir : str, optional
        Directory to save the output figure (default: same directory as input)

    Returns:
    --------
    dict
        Dictionary containing the analysis results
    """
    # Load data from JSONL file
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    scenario_data = json.loads(line)
                    data.append(scenario_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")

    print(f"Parsed {len(data)} scenario objects")

    # Extract scores and calculate differences
    differences = []
    best_scores = []
    worst_scores = []

    metric_key = f"{metric_type}_best"
    worst_key = f"{metric_type}_worst"
    score_key = f"{metric_type}_score"

    for scenario in data:
        best_score = scenario[metric_key][score_key]
        worst_score = scenario[worst_key][score_key]
        difference = best_score - worst_score

        best_scores.append(best_score)
        worst_scores.append(worst_score)
        differences.append(difference)

    # Calculate statistics
    mean_diff = np.mean(differences)
    median_diff = np.median(differences)
    std_diff = np.std(differences)

    # Print summary statistics
    print(f"Number of differences calculated: {len(differences)}")
    print(f"Range of differences: {min(differences):.4f} to {max(differences):.4f}")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Best scores range: {min(best_scores):.4f} to {max(best_scores):.4f}")
    print(f"Worst scores range: {min(worst_scores):.4f} to {max(worst_scores):.4f}")
    print("\nSummary statistics for differences:")
    print(f"Mean: {mean_diff:.4f}")
    print(f"Median: {median_diff:.4f}")
    print(f"Std Dev: {std_diff:.4f}")

    # Create the visualization
    plt.figure(figsize=(10, 6))

    # Create histogram with KDE
    ax = sns.histplot(data=differences, bins=num_bins, alpha=0.7, color="skyblue", edgecolor="black", kde=True)

    # Get the KDE line and modify its properties
    for line in ax.lines:
        line.set_color("#d95f02")
        line.set_linestyle("--")
        line.set_linewidth(1.5)
        line.set_label("Smoothed Density")

    # Add labels and title
    metric_name = "Cosine Similarity" if metric_type == "cosine" else "Spearman Correlation"
    plt.xlabel(f"Difference between Best and Worst {metric_name} Scores ($\\Delta \\rho$)")
    plt.ylabel("Number of Scenarios")
    plt.title(f"{dataset_name}: Distribution of Differences Between Best and Worst {metric_name} Scores")
    plt.grid(axis="y", alpha=0.3)

    # Add vertical lines for mean and median with refined styles
    plt.axvline(x=mean_diff, color="#0072B2", linestyle="--", label=f"Mean = {mean_diff:.3f}")
    plt.axvline(x=median_diff, color="green", linestyle="--", label=f"Median = {median_diff:.3f}")
    plt.legend()

    # Show counts above bars
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:  # Only show for non-empty bins
            plt.text(patch.get_x() + patch.get_width() / 2, height + 0.1, str(int(height)), ha="center", fontsize=8)

    # Display bin information
    print("\nBin information:")
    bin_edges = ax.patches[0].get_bbox().get_points()
    bin_width = bin_edges[1][0] - bin_edges[0][0]
    for i, patch in enumerate(ax.patches):
        bin_start = patch.get_x()
        bin_end = bin_start + patch.get_width()
        count = int(patch.get_height())
        print(f"Bin {i+1}: {bin_start:.4f} to {bin_end:.4f} - Count: {count}")

    plt.tight_layout()

    # Save the figure
    if output_dir is None:
        output_dir = os.path.dirname(jsonl_file) or "."

    output_path = os.path.join(output_dir, f"{metric_type}_score_differences.png")
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to: {output_path}")

    plt.show()

    # Return results as a dictionary
    return {
        "differences": differences,
        "best_scores": best_scores,
        "worst_scores": worst_scores,
        "mean": mean_diff,
        "median": median_diff,
        "std_dev": std_diff,
        "bin_edges": bin_edges.tolist(),
        "counts": [int(patch.get_height()) for patch in ax.patches],
    }


# --- New function: plot_ranked_kde ---
def plot_ranked_kde(jsonl_file, metric_type="cosine", dataset_name="", output_dir=None):
    """
    Plot KDEs of explanation scores by rank for each scenario in the dataset.
    """
    # Load data from JSONL file
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                try:
                    scenario_data = json.loads(line)
                    data.append(scenario_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")

    print(f"Parsed {len(data)} scenario objects for ranked KDE")

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Process and plot both metrics
    for ax, current_metric in [(ax1, "cosine"), (ax2, "spearman")]:
        # Extract scores by rank with updated labels
        rank_data = {"Rank 1 (Best)": [], "Rank 2": [], "Rank 3": [], "Rank 4": [], "Rank 5 (Worst)": []}
        for scenario in data:
            ranked_explanations = compute_explanation_ranks(scenario, metric_type=current_metric)
            if len(ranked_explanations) == 5:
                for i, score_info in enumerate(ranked_explanations):
                    rank_label = "Rank 1 (Best)" if i == 0 else "Rank 5 (Worst)" if i == 4 else f"Rank {i+1}"
                    rank_data[rank_label].append(score_info[current_metric])

        # Calculate mean of all scores to center the plot
        all_scores = []
        for scores in rank_data.values():
            all_scores.extend(scores)
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)

        # Set x-axis limits centered around the mean with a reasonable range
        x_min = mean_score - 3 * std_score
        x_max = mean_score + 3 * std_score

        # Ensure we don't exceed the valid range for each metric
        if current_metric in ["cosine", "spearman"]:
            x_min = max(x_min, -1)
            x_max = min(x_max, 1)
        else:  # lma
            x_min = max(x_min, 0)
            x_max = min(x_max, 1)

        ax.set_xlim(x_min, x_max)

        # Define colors to match the reference image
        rank_colors = {
            "Rank 1 (Best)": "blue",
            "Rank 2": "orange",
            "Rank 3": "green",
            "Rank 4": "red",
            "Rank 5 (Worst)": "purple",
        }

        for rank_label, scores in rank_data.items():
            if scores:
                sns.kdeplot(
                    scores,
                    label=rank_label,
                    fill=True,  # Restore the fill
                    common_norm=False,
                    color=rank_colors[rank_label],
                    alpha=0.3,  # Restore original alpha
                    linewidth=2,
                    bw_adjust=1,
                    ax=ax,
                )

        metric_name = "Cosine Similarity" if current_metric == "cosine" else "Spearman Correlation"
        ax.set_xlabel(metric_name, fontsize=26)
        # Only set ylabel for the first subplot (cosine)
        if current_metric == "cosine":
            ax.set_ylabel("Density", fontsize=26)
        else:
            ax.set_ylabel("")  # Remove ylabel for spearman
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=26)

        # Create legend for this subplot
        ax.legend(title="Explanation Rank", loc="upper right", fontsize=26, title_fontsize=26)

    # Create a single legend at the top using handles from ax1
    handles, labels = ax1.get_legend_handles_labels()
    # Reverse the order of handles and labels
    handles = handles[::-1]
    labels = labels[::-1]
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.85),  # More reasonable legend position
        ncol=5,
        fontsize=26,
    )

    # Now we can safely remove the individual legends
    ax1.get_legend().remove()
    ax2.get_legend().remove()

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.15)  # Restore original top margin

    if output_dir is None:
        output_dir = os.path.dirname(jsonl_file) or "."

    output_path = os.path.join(output_dir, "ranked_kde_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Combined ranked KDE figure saved to: {output_path}")
    plt.show()


# Example usage:
if __name__ == "__main__":
    file_path = Path(
        "data/collection_data/ecqa/unsloth_Meta-Llama-3.1-8B-Instruct/ecqa_20250404_120218_LIME_llama3.1/ecqa_20250404_120218_LIME_llama3.1_fixed.jsonl"
    )

    dataset_name = "ECQA-LIME"
    # # Analyze the JSONL file with cosine similarity
    # results_cosine = analyze_metric_differences(
    #     str(file_path), metric_type="cosine", num_bins=50, dataset_name=dataset_name, output_dir=None
    # )
    # # Analyze the JSONL file with Spearman correlation
    # results_spearman = analyze_metric_differences(
    #     str(file_path), metric_type="spearman", num_bins=50, dataset_name=dataset_name, output_dir=None
    # )

    plot_ranked_kde(str(file_path), metric_type="spearman", dataset_name=dataset_name, output_dir=None)
    plot_ranked_kde(str(file_path), metric_type="cosine", dataset_name=dataset_name, output_dir=None)
    plot_ranked_kde(str(file_path), metric_type="lma", dataset_name=dataset_name, output_dir=None)
