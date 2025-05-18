import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    plt.xlabel(f"Difference between Best and Worst {metric_name} Scores ($\\Delta \\rho$)", fontsize=16)
    plt.ylabel("Number of Scenarios", fontsize=16)
    plt.title(f"{dataset_name}: Distribution of Differences Between Best and Worst {metric_name} Scores", fontsize=18)
    plt.grid(axis="y", alpha=0.3)

    # Add vertical lines for mean and median with refined styles
    plt.axvline(x=mean_diff, color="#0072B2", linestyle="--", label=f"Mean = {mean_diff:.3f}")
    plt.axvline(x=median_diff, color="green", linestyle="--", label=f"Median = {median_diff:.3f}")
    plt.legend(fontsize=14)

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

    Parameters:
    -----------
    jsonl_file : str
        Path to the JSONL file containing scenario data
    metric_type : str, optional
        Type of metric to analyze ('cosine' or 'spearman', default: 'cosine')
    dataset_name : str, optional
        Name of the dataset for the plot title
    output_dir : str, optional
        Directory to save the output figure (default: same directory as input)
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

    print(f"Parsed {len(data)} scenario objects for ranked KDE")

    # Extract scores by rank robustly
    rank_data = {f"Rank {i+1}": [] for i in range(5)}
    score_key = f"{metric_type}_scores"

    for scenario in data:
        # Try to extract the list of scores for this metric
        scores = scenario.get(score_key, None)
        if scores is None:
            # Try to extract from explanation details if present
            explanation_details = scenario.get("explanation_details", [])
            scores = []
            for detail in explanation_details:
                score = detail.get(f"{metric_type}_score", None)
                if score is not None:
                    scores.append(score)
        # Only use if we have 5 scores
        if isinstance(scores, list) and len(scores) == 5:
            for i, score in enumerate(scores):
                rank_data[f"Rank {i+1}"].append(score)

    # Print how many scores per rank for debugging
    print("Rank data counts:")
    for rank_label, scores in rank_data.items():
        print(rank_label, len(scores))

    # Define colors to match the reference image
    rank_colors = {"Rank 1": "blue", "Rank 2": "orange", "Rank 3": "green", "Rank 4": "red", "Rank 5": "purple"}

    # Plot
    plt.figure(figsize=(10, 6))
    for rank_label, scores in rank_data.items():
        if scores:
            sns.kdeplot(
                scores,
                label=rank_label,
                fill=True,
                common_norm=False,
                color=rank_colors[rank_label],
                alpha=0.4,
                linewidth=2,
            )

    metric_name = "Cosine Similarity" if metric_type == "cosine" else "Spearman Correlation"
    # plt.title(f"Ranked KDEs of Self-Consistency Scores (Manual)", fontsize=18)
    plt.xlabel(f"{metric_name}", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.legend(title="Explanation Rank", loc="best", fontsize=16, title_fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir is None:
        output_dir = os.path.dirname(jsonl_file) or "."

    output_path = os.path.join(output_dir, f"{metric_type}_ranked_kde.png")
    plt.savefig(output_path, dpi=300)
    print(f"Ranked KDE figure saved to: {output_path}")
    plt.show()


# Example usage:
if __name__ == "__main__":
    file_path = Path(
        "data/collection_data/ecqa/unsloth_Meta-Llama-3.1-8B-Instruct/ecqa_20250404_120218_LIME_llama3.1/ecqa_20250404_120218_LIME_llama3.1_fixed.jsonl"
    )

    dataset_name = "ECQA-LIME"
    # Analyze the JSONL file with cosine similarity
    # results_cosine = analyze_metric_differences(
    #     str(file_path), metric_type="cosine", num_bins=50, dataset_name=dataset_name, output_dir=None
    # )
    # # Analyze the JSONL file with Spearman correlation
    # results_spearman = analyze_metric_differences(
    #     str(file_path), metric_type="spearman", num_bins=50, dataset_name=dataset_name, output_dir=None
    # )

    plot_ranked_kde(str(file_path), metric_type="spearman", dataset_name=dataset_name, output_dir=None)
