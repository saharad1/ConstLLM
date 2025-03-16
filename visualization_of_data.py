import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_spearman_differences(jsonl_file, num_bins=8, dataset_name="", output_dir=None):
    """
    Analyze and visualize the differences between best and worst Spearman scores
    from a JSONL file containing scenario data.

    Parameters:
    -----------
    jsonl_file : str
        Path to the JSONL file containing scenario data
    num_bins : int, optional
        Number of bins for the histogram (default: 8)
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

    for scenario in data:
        best_score = scenario["explanation_best"]["spearman_score"]
        worst_score = scenario["explanation_worst"]["spearman_score"]
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

    # Create histogram
    counts, bin_edges, _ = plt.hist(differences, bins=num_bins, alpha=0.7, color="skyblue", edgecolor="black")

    # Add labels and title
    plt.xlabel("Difference between Best and Worst Spearman Scores")
    plt.ylabel("Number of Scenarios")
    plt.title(f"{dataset_name}: Distribution of Differences Between Best and Worst Spearman Scores")
    plt.grid(axis="y", alpha=0.3)

    # Add vertical lines for mean and median
    plt.axvline(x=mean_diff, color="red", linestyle="--", label=f"Mean = {mean_diff:.3f}")
    plt.axvline(x=median_diff, color="green", linestyle="-.", label=f"Median = {median_diff:.3f}")
    plt.legend()

    # Show counts above bars
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for count, x in zip(counts, bin_centers):
        if count > 0:  # Only show for non-empty bins
            plt.text(x, count + 0.1, str(int(count)), ha="center")

    # Display bin information
    print("\nBin information:")
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        count = len([d for d in differences if bin_start <= d < bin_end])
        print(f"Bin {i+1}: {bin_start:.4f} to {bin_end:.4f} - Count: {count}")

    plt.tight_layout()

    # Save the figure
    if output_dir is None:
        output_dir = os.path.dirname(jsonl_file) or "."

    output_path = os.path.join(output_dir, "spearman_score_differences.png")
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
        "counts": counts.tolist(),
    }


# Example usage:
if __name__ == "__main__":
    file_path = Path("dpo_datasets/codah_dpo_datasets/codah_250219_165846_LIME.jsonl")
    dataset_name = "CodaH-LIME"
    # Analyze the JSONL file
    results = analyze_spearman_differences(str(file_path), num_bins=50, dataset_name=dataset_name, output_dir=None)
