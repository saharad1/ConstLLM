import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.analyze_data.analysis_utils import extract_choice, parse_line
from src.collect_data.comp_similarity_scores import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
)


def clean_token(token: str) -> str:
    """
    Clean a token by removing special characters and properly formatting it.

    Args:
        token: The raw token string

    Returns:
        Cleaned token string
    """
    # Remove special characters
    token = token.replace("Ċ", "")  # Remove newline character
    token = token.replace("Ġ", " ")  # Replace space marker with actual space

    # Clean up any double spaces
    token = " ".join(token.split())

    return token


def is_system_prompt_token(token: str, prev_token: str = None, token_index: int = 0) -> bool:
    """
    Check if a token is part of the system prompt by looking for specific patterns at the beginning.

    Args:
        token: The token to check
        prev_token: The previous token (if any) to check for patterns
        token_index: The index of the current token in the sequence

    Returns:
        True if the token is part of the system prompt, False otherwise
    """
    # Only check for system prompt in the first 20 tokens
    if token_index > 20:
        return False

    # Clean the tokens
    clean_token_text = clean_token(token).lower()
    clean_prev_token = clean_token(prev_token).lower() if prev_token else None

    # System prompt patterns to check
    system_patterns = [
        # "You are Qwen" pattern
        (clean_prev_token == "you" and clean_token_text == "are"),
        (clean_prev_token == "are" and clean_token_text == "qwen"),
        # "created by Alibaba Cloud" pattern
        (clean_prev_token == "qwen" and clean_token_text == "created"),
        (clean_prev_token == "created" and clean_token_text == "by"),
        (clean_prev_token == "by" and clean_token_text == "alibaba"),
        (clean_prev_token == "alibaba" and clean_token_text == "cloud"),
        # "You are a helpful assistant" pattern
        (clean_prev_token == "cloud" and clean_token_text == "you"),
        (clean_prev_token == "you" and clean_token_text == "are"),
        (clean_prev_token == "are" and clean_token_text == "a"),
        (clean_prev_token == "a" and clean_token_text == "helpful"),
        (clean_prev_token == "helpful" and clean_token_text == "assistant"),
    ]

    return any(system_patterns)


def is_qwen_model(file_path: str) -> bool:
    """
    Check if the dataset is from a Qwen model based on the file path.

    Args:
        file_path: Path to the dataset file

    Returns:
        True if the dataset is from a Qwen model, False otherwise
    """
    return "qwen" in file_path.lower()


def is_llama_model(file_path: str) -> bool:
    """
    Check if the dataset is from a Llama model based on the file path.

    Args:
        file_path: Path to the dataset file

    Returns:
        True if the dataset is from a Llama model, False otherwise
    """
    return "llama" in file_path.lower()


def extract_model_version(part: str) -> str:
    """
    Extract the model version from a path part.

    Args:
        part: A part of the file path

    Returns:
        The extracted model version or the original part if no version is found
    """
    part_lower = part.lower()
    print(f"Checking part for model version: {part}")  # Debug print

    # Check for specific LLaMA versions with their exact path format
    if "meta-llama-3.1-8b-instruct" in part_lower:
        return "LLaMA3.1-8B"
    elif "llama-3.2-3b-instruct" in part_lower:
        return "LLaMA3.2-3B"

    # If no version found, return the original part
    print(f"No version found in part: {part}")  # Debug print
    return part


def create_comparison_heatmap(
    file_path1: str,
    file_path2: str,
    num_scenarios: int = 20,
    output_dir: str = "outputs/heatmaps/comparison",
) -> None:
    """
    Create visualizations comparing feature attributions between two models for the same scenarios.

    Args:
        file_path1: Path to the first model's dataset file
        file_path2: Path to the second model's dataset file
        num_scenarios: Number of scenarios to visualize (default: 20)
        output_dir: Directory to save the visualization images (default: 'outputs/heatmaps/comparison')
    """
    print(f"\nProcessing files:\n1. {file_path1}\n2. {file_path2}")

    # Extract dataset names and model names
    dataset_name1 = None
    dataset_name2 = None
    model_name1 = None
    model_name2 = None

    # Process first file
    if "ecqa" in file_path1.lower():
        dataset_name1 = "ecqa"
    elif "arc_easy" in file_path1.lower():
        dataset_name1 = "arc_easy"
    else:
        path_parts = file_path1.split("/")
        for part in path_parts:
            if part not in ["data", "collection_data", "outputs", "heatmaps"]:
                dataset_name1 = part
                break

    # Process second file
    if "ecqa" in file_path2.lower():
        dataset_name2 = "ecqa"
    elif "arc_easy" in file_path2.lower():
        dataset_name2 = "arc_easy"
    else:
        path_parts = file_path2.split("/")
        for part in path_parts:
            if part not in ["data", "collection_data", "outputs", "heatmaps"]:
                dataset_name2 = part
                break

    # Extract model names
    for file_path, model_name_var in [(file_path1, "model_name1"), (file_path2, "model_name2")]:
        path_parts = file_path.split("/")
        for part in path_parts:
            if "llama" in part.lower():
                if model_name_var == "model_name1":
                    model_name1 = extract_model_version(part)
                else:
                    model_name2 = extract_model_version(part)
                break
            elif "qwen" in part.lower():
                if model_name_var == "model_name1":
                    model_name1 = part
                else:
                    model_name2 = part
                break

    if dataset_name1 != dataset_name2:
        raise ValueError("Both files must be from the same dataset")

    # Create output directory with a more specific name
    comparison_dir = f"{output_dir}/{dataset_name1}/{model_name1}_vs_{model_name2}/top_100_spearman_diff"
    Path(comparison_dir).mkdir(parents=True, exist_ok=True)

    # Custom colormap for better visualization
    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging", ["#66b3ff", "#b3e0ff", "#ffffff", "#ffd3b6", "#ffa07a"]
    )

    # Store scenarios with their Spearman differences
    scenarios_with_scores = []

    # Read both files
    with open(file_path1, "r") as f1, open(file_path2, "r") as f2:
        for i, (line1, line2) in enumerate(zip(f1, f2)):
            try:
                scenario1 = parse_line(line1)
                scenario2 = parse_line(line2)

                if (
                    "decision_attributions" in scenario1
                    and "decision_prompt" in scenario1
                    and "decision_attributions" in scenario2
                    and "decision_prompt" in scenario2
                ):
                    # Get the attributions and prompts
                    attributions1 = scenario1["decision_attributions"]
                    attributions2 = scenario2["decision_attributions"]
                    prompt1 = scenario1["decision_prompt"]
                    prompt2 = scenario2["decision_prompt"]

                    # Get models' decisions
                    decision_output1 = scenario1.get("decision_output", "")
                    decision_output2 = scenario2.get("decision_output", "")
                    decision_choice1 = extract_choice(decision_output1)
                    decision_choice2 = extract_choice(decision_output2)

                    # Find worst explanations for both models
                    worst_explanation1 = None
                    worst_explanation2 = None
                    worst_spearman1 = float("inf")
                    worst_spearman2 = float("inf")

                    if (
                        "explanation_attributions" in scenario1
                        and "explanation_outputs" in scenario1
                        and "explanation_attributions" in scenario2
                        and "explanation_outputs" in scenario2
                    ):
                        # Process first model's explanations
                        for expl_attr, expl_text in zip(
                            scenario1["explanation_attributions"], scenario1["explanation_outputs"]
                        ):
                            spearman_score = calculate_spearman_correlation(attributions1, expl_attr)
                            if spearman_score is not None and spearman_score < worst_spearman1:
                                worst_spearman1 = spearman_score
                                worst_explanation1 = (expl_attr, expl_text)

                        # Process second model's explanations
                        for expl_attr, expl_text in zip(
                            scenario2["explanation_attributions"], scenario2["explanation_outputs"]
                        ):
                            spearman_score = calculate_spearman_correlation(attributions2, expl_attr)
                            if spearman_score is not None and spearman_score < worst_spearman2:
                                worst_spearman2 = spearman_score
                                worst_explanation2 = (expl_attr, expl_text)

                        # Calculate the difference in Spearman correlations
                        if worst_spearman1 != float("inf") and worst_spearman2 != float("inf"):
                            spearman_diff = abs(worst_spearman1 - worst_spearman2)
                            scenarios_with_scores.append(
                                (i, scenario1, scenario2, worst_spearman1, worst_spearman2, spearman_diff)
                            )

            except Exception as e:
                print(f"Error processing scenario {i+1}: {e}")
                continue

    # Sort scenarios by Spearman difference and take top 100
    scenarios_with_scores.sort(key=lambda x: x[5], reverse=True)
    top_scenarios = scenarios_with_scores[:100]

    # Process top 100 scenarios
    for i, (scenario_idx, scenario1, scenario2, worst_spearman1, worst_spearman2, spearman_diff) in enumerate(
        top_scenarios
    ):
        try:
            # Get the attributions and prompts
            attributions1 = scenario1["decision_attributions"]
            attributions2 = scenario2["decision_attributions"]
            prompt1 = scenario1["decision_prompt"]
            prompt2 = scenario2["decision_prompt"]

            # Get models' decisions
            decision_output1 = scenario1.get("decision_output", "")
            decision_output2 = scenario2.get("decision_output", "")
            decision_choice1 = extract_choice(decision_output1)
            decision_choice2 = extract_choice(decision_output2)

            # Find worst explanations for both models
            worst_explanation1 = None
            worst_explanation2 = None

            if (
                "explanation_attributions" in scenario1
                and "explanation_outputs" in scenario1
                and "explanation_attributions" in scenario2
                and "explanation_outputs" in scenario2
            ):
                # Process first model's explanations
                for expl_attr, expl_text in zip(
                    scenario1["explanation_attributions"], scenario1["explanation_outputs"]
                ):
                    spearman_score = calculate_spearman_correlation(attributions1, expl_attr)
                    if spearman_score is not None and spearman_score == worst_spearman1:
                        worst_explanation1 = (expl_attr, expl_text)
                        break

                # Process second model's explanations
                for expl_attr, expl_text in zip(
                    scenario2["explanation_attributions"], scenario2["explanation_outputs"]
                ):
                    spearman_score = calculate_spearman_correlation(attributions2, expl_attr)
                    if spearman_score is not None and spearman_score == worst_spearman2:
                        worst_explanation2 = (expl_attr, expl_text)
                        break

            # Create HTML content for comparison
            html_content = []
            html_content.append(
                f"""
            <html>
            <head>
                <style>
                    .figure-container {{
                        font-family: Arial, sans-serif;
                        font-size: 13px;
                        max-width: 900px;
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    .decision-section {{
                        width: 100%;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 6px 15px;
                        background: #fafbfc;
                        margin-bottom: 5px;
                        box-sizing: border-box;
                    }}
                    .model-header {{
                        font-weight: bold;
                        font-size: 16px;
                        margin-bottom: 4px;
                        color: #333;
                        border-bottom: 2px solid #ddd;
                        padding-bottom: 2px;
                    }}
                    .explanation-section {{
                        width: 100%;
                        padding: 6px 15px;
                        background: #f8f9fa;
                        border-radius: 8px;
                        border: 1px solid #ddd;
                        margin-bottom: 5px;
                        box-sizing: border-box;
                    }}
                    .explanation-title {{
                        font-weight: bold;
                        margin-bottom: 4px;
                    }}
                    .spearman-score {{
                        font-weight: bold;
                        color: green;
                    }}
                    .explanation-text {{
                        margin-top: 4px;
                        font-size: 14px;
                        color: #333;
                        line-height: 1.2;
                    }}
                    .heatmap-tokens {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 1px;
                        margin-bottom: 4px;
                    }}
                    .word {{
                        padding: 0px 4px;
                        border-radius: 3px;
                        margin: 1px;
                    }}
                    .color-scale-container {{
                        width: 100%;
                        margin-top: 5px;
                        display: flex;
                        flex-direction: column;
                        align-items: stretch;
                        box-sizing: border-box;
                    }}
                    .color-scale-horizontal {{
                        width: 100%;
                        height: 8px;
                        background: linear-gradient(to right, #66b3ff 0%, #b3e0ff 25%, #ffffff 50%, #ffd3b6 75%, #ffa07a 100%);
                        border: 1px solid #ccc;
                        border-radius: 2px;
                        margin: 1px 0;
                        box-sizing: border-box;
                    }}
                    .scale-labels-horizontal {{
                        display: flex;
                        justify-content: space-between;
                        font-size: 12px;
                        color: #666;
                        width: 100%;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    .spearman-diff {{
                        font-weight: bold;
                        color: #e74c3c;
                        margin-left: 10px;
                    }}
                </style>
            </head>
            <body>
            <div class="figure-container">
            """
            )

            # First, create the decision section for the first model only
            model_name = model_name1
            attributions = attributions1
            decision_choice = decision_choice1
            worst_explanation = worst_explanation1
            worst_spearman = worst_spearman1

            # Check if this is a Qwen or Llama model dataset
            is_qwen = is_qwen_model(file_path1)
            is_llama = is_llama_model(file_path1)

            # Decide how many tokens to skip
            if is_qwen:
                skip_tokens = 17
            elif is_llama:
                skip_tokens = 19
            else:
                skip_tokens = 0

            html_content.append('<div class="decision-section">')
            html_content.append(f'<div class="explanation-title">Model Decision: {decision_choice}</div>')
            html_content.append('<div class="heatmap-tokens">')

            # Calculate normalization parameters
            valid_scores = []
            for token_idx, (token, score) in enumerate(attributions):
                try:
                    if token_idx < skip_tokens:
                        continue
                    score = float(score)
                    valid_scores.append(score)
                except (ValueError, TypeError):
                    continue

            if valid_scores:
                scores_array = np.array(valid_scores)
                p95 = np.percentile(scores_array, 95)
                p5 = np.percentile(scores_array, 5)
            else:
                p5, p95 = -1, 1

            # Add heatmap tokens
            for token_idx, (token, score) in enumerate(attributions):
                try:
                    if token_idx < skip_tokens:
                        continue
                    score = float(score)
                    clipped_score = np.clip(score, p5, p95)
                    if p95 == p5:
                        normalized_score = 0
                    else:
                        normalized_score = (clipped_score - p5) / (p95 - p5) * 2 - 1
                        if clipped_score == p5:
                            normalized_score = -1
                        elif clipped_score == p95:
                            normalized_score = 1
                    normalized_score = max(-1, min(1, normalized_score * 1.2))
                    color = cmap((normalized_score + 1) / 2)
                    color_hex = "#{:02x}{:02x}{:02x}".format(
                        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                    )
                    clean_token_text = clean_token(token)
                    if clean_token_text:
                        html_content.append(
                            f'<span class="word" style="background-color: {color_hex}" title="Score: {score:.3f}">{clean_token_text}</span>'
                        )
                except (ValueError, TypeError):
                    clean_token_text = clean_token(token)
                    if clean_token_text:
                        html_content.append(f'<span class="word">{clean_token_text}</span>')
            html_content.append("</div>")
            html_content.append("</div>")  # Close decision-section

            # Now add the explanation sections for both models
            for model_idx, (
                model_name,
                attributions,
                decision_choice,
                worst_explanation,
                worst_spearman,
            ) in enumerate(
                [
                    (
                        model_name1 + " Vanilla",
                        attributions1,
                        decision_choice1,
                        worst_explanation1,
                        worst_spearman1,
                    ),
                    (
                        model_name2 + " Ours (DPO-Tuned)",
                        attributions2,
                        decision_choice2,
                        worst_explanation2,
                        worst_spearman2,
                    ),
                ]
            ):
                if worst_explanation is not None:
                    expl_attr, expl_text = worst_explanation
                    html_content.append('<div class="explanation-section">')
                    # Calculate cosine score
                    cosine_score = calculate_cosine_similarity(attributions, expl_attr)
                    html_content.append(
                        f'<div class="explanation-title">{model_name} - Worst Explanation (Spearman: <span class="spearman-score">{-worst_spearman:.4f}</span>, Cosine: <span class="spearman-score">{cosine_score:.4f}</span>)</div>'
                    )
                    html_content.append('<div class="heatmap-tokens">')

                    # Calculate normalization for explanation
                    valid_scores = []
                    # Remove trailing tokens
                    N_TRAILING = 11  # Number of words in the trailing prompt
                    if len(expl_attr) > N_TRAILING:
                        expl_attr = expl_attr[:-N_TRAILING]
                    for token_idx, (token, score) in enumerate(expl_attr):
                        try:
                            if token_idx < skip_tokens:
                                continue
                            score = float(score)
                            valid_scores.append(score)
                        except (ValueError, TypeError):
                            continue

                    if valid_scores:
                        scores_array = np.array(valid_scores)
                        p95 = np.percentile(scores_array, 95)
                        p5 = np.percentile(scores_array, 5)
                    else:
                        p5, p95 = -1, 1

                    for token_idx, (token, score) in enumerate(expl_attr):
                        try:
                            if token_idx < skip_tokens:
                                continue
                            score = float(score)
                            clipped_score = np.clip(score, p5, p95)
                            if p95 == p5:
                                normalized_score = 0
                            else:
                                normalized_score = (clipped_score - p5) / (p95 - p5) * 2 - 1
                                if clipped_score == p5:
                                    normalized_score = -1
                                elif clipped_score == p95:
                                    normalized_score = 1
                            normalized_score = max(-1, min(1, normalized_score * 1.2))
                            color = cmap((normalized_score + 1) / 2)
                            color_hex = "#{:02x}{:02x}{:02x}".format(
                                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                            )
                            clean_token_text = clean_token(token)
                            if clean_token_text:
                                html_content.append(
                                    f'<span class="word" style="background-color: {color_hex}" title="Score: {score:.3f}">{clean_token_text}</span>'
                                )
                        except (ValueError, TypeError):
                            clean_token_text = clean_token(token)
                            if clean_token_text:
                                html_content.append(f'<span class="word">{clean_token_text}</span>')
                    html_content.append("</div>")
                    html_content.append(f'<div class="explanation-text">{expl_text.strip()}</div>')
                    html_content.append("</div>")

            # Add color scale
            html_content.append(
                """
                <div class="color-scale-container">
                    <div class="color-scale-horizontal"></div>
                    <div class="scale-labels-horizontal">
                        <span>-1.0</span>
                        <span>0.0</span>
                        <span>1.0</span>
                    </div>
                </div>
            </div>
            </body>
            </html>
            """
            )

            # Save as HTML file
            output_file = f"{comparison_dir}/scenario_{scenario_idx+1}_comparison.html"
            with open(output_file, "w") as f:
                f.write("\n".join(html_content))

        except Exception as e:
            print(f"Error creating comparison visualization for scenario {scenario_idx+1}: {e}")
            continue

    print(f"\nGenerated {len(top_scenarios)} heatmaps with highest Spearman correlation differences")


if __name__ == "__main__":
    # Example usage
    file_path1 = "data/eval_results/ecqa/Llama-3.2-3B-Instruct/ecqa_250510_003719_lr2.67e-06_beta6.41/eval_250510_135741_test_1089_LIME/eval_250510_135741_test_1089_LIME_results.jsonl"
    file_path2 = "data/eval_results/ecqa/huggingface/Llama-3.2-3B-Instruct/eval_250505_103954_test_1089_LIME/eval_250505_103954_test_1089_LIME_results.jsonl"

    # Create comparison heatmaps
    create_comparison_heatmap(file_path1, file_path2, num_scenarios=1000)
