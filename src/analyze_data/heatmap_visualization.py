import ast
import json
from pathlib import Path
from typing import Any, Dict, List

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


def create_attribution_heatmap(file_path: str, num_scenarios: int = 20, output_dir: str = "outputs/heatmaps") -> None:
    """
    Create visualizations of feature attributions by coloring the words in the text based on their attribution scores.

    Process:
    1. First remove system prompt tokens (if Qwen or Llama model)
    2. Then collect valid scores from remaining tokens
    3. Calculate normalization parameters (percentiles) on valid scores
    4. Finally create visualization with normalized colors

    Args:
        file_path: Path to the dataset file
        num_scenarios: Number of scenarios to visualize (default: 20)
        output_dir: Directory to save the visualization images (default: 'outputs/heatmaps')
    """
    # Extract dataset name from file path
    dataset_name = None
    if "ecqa" in file_path.lower():
        dataset_name = "ecqa"
    elif "arc_easy" in file_path.lower():
        dataset_name = "arc_easy"
    else:
        # Extract dataset name from the path if it's not one of the known datasets
        path_parts = file_path.split("/")
        for part in path_parts:
            if part not in ["data", "collection_data", "outputs", "heatmaps"]:
                dataset_name = part
                break

    if dataset_name is None:
        dataset_name = "unknown_dataset"

    # Extract model name from file path
    model_name = None
    path_parts = file_path.split("/")
    for part in path_parts:
        if "llama" in part.lower():
            # Extract Llama version
            if "llama-3.1" in part.lower() or "llama3.1" in part.lower():
                model_name = "llama3.1"
            elif "llama-3.2" in part.lower() or "llama3.2" in part.lower():
                model_name = "llama3.2"
            else:
                model_name = part
            break
        elif "qwen" in part.lower():
            model_name = part
            break

    if model_name is None:
        model_name = "unknown_model"

    # Create dataset and model-specific output directory
    model_output_dir = f"{output_dir}/{dataset_name}/{model_name}"
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    # Check if this is a Qwen or Llama model dataset
    is_qwen = is_qwen_model(file_path)
    is_llama = is_llama_model(file_path)
    print(f"Dataset is from Qwen model: {is_qwen}")
    print(f"Dataset is from Llama model: {is_llama}")

    # Decide how many tokens to skip
    if is_qwen:
        skip_tokens = 17
    elif is_llama:
        skip_tokens = 19
    else:
        skip_tokens = 0

    # Custom colormap for better visualization - using brighter colors
    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging", ["#66b3ff", "#b3e0ff", "#ffffff", "#ffd3b6", "#ffa07a"]
    )

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_scenarios:
                break

            try:
                scenario = parse_line(line)

                if "decision_attributions" in scenario and "decision_prompt" in scenario:
                    # Get the attributions and prompt
                    attributions = scenario["decision_attributions"]
                    prompt = scenario["decision_prompt"]

                    # Get model's decision
                    decision_output = scenario.get("decision_output", "")
                    decision_choice = extract_choice(decision_output)

                    # Get best and worst explanations based on Spearman correlation
                    best_explanation = None
                    worst_explanation = None
                    best_spearman = float("-inf")
                    worst_spearman = float("inf")

                    if "explanation_attributions" in scenario and "explanation_outputs" in scenario:
                        decision_attr = scenario["decision_attributions"]
                        explanation_attrs = scenario["explanation_attributions"]
                        explanation_texts = scenario["explanation_outputs"]

                        for expl_attr, expl_text in zip(explanation_attrs, explanation_texts):
                            spearman_score = calculate_spearman_correlation(decision_attr, expl_attr)
                            if spearman_score is not None:
                                if spearman_score > best_spearman:
                                    best_spearman = spearman_score
                                    best_explanation = expl_text
                                if spearman_score < worst_spearman:
                                    worst_spearman = spearman_score
                                    worst_explanation = expl_text

                    # Step 1: First pass - collect valid scores (excluding system prompt tokens for Qwen/Llama)
                    valid_scores = []
                    print(f"\nScenario {i+1} - Raw attribution values:")
                    for token_idx, (token, score) in enumerate(attributions):
                        try:
                            score = float(score)
                            # Skip system prompt tokens for Qwen/Llama
                            if token_idx >= skip_tokens:
                                valid_scores.append(score)
                                print(f"Token: {token}, Score: {score:.4f}")
                        except (ValueError, TypeError):
                            print(f"Token: {token}, Score: Invalid")
                            continue

                    # Step 2: Calculate normalization parameters using only valid scores
                    if valid_scores:
                        # Calculate percentiles for robust scaling
                        scores_array = np.array(valid_scores)
                        p98 = np.percentile(scores_array, 98)
                        p2 = np.percentile(scores_array, 2)
                        p95, p5 = p98, p2

                        print(f"Score statistics (after removing system prompt):")
                        print(f"  5th percentile: {p5:.4f}")
                        print(f"  95th percentile: {p95:.4f}")
                        print(f"  Number of valid tokens: {len(valid_scores)}")
                    else:
                        p5, p95 = -1, 1

                    # Create HTML content with new structure
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
                            }}
                            .top-row {{
                                margin-bottom: 10px;
                            }}
                            .model-decision-col {{
                                border: 1px solid #eee;
                                border-radius: 4px;
                                padding: 6px;
                                background: #fafbfc;
                                margin-bottom: 4px;
                            }}
                            .model-decision-label {{
                                font-weight: bold;
                                margin-bottom: 2px;
                            }}
                            .spearman-score {{
                                font-weight: bold;
                                color: green;
                            }}
                            .heatmap-tokens {{
                                display: flex;
                                flex-wrap: wrap;
                                gap: 1px;
                                margin-bottom: 6px;
                            }}
                            .bottom-row {{
                                display: flex;
                                gap: 16px;
                                margin-bottom: 8px;
                            }}
                            .explanation-col {{
                                flex: 1;
                                min-width: 0;
                                border: 1px solid #eee;
                                border-radius: 4px;
                                padding: 6px;
                                background: #fafbfc;
                            }}
                            .explanation-title {{
                                font-weight: bold;
                                margin-bottom: 2px;
                            }}
                            .explanation-text {{
                                margin-top: 4px;
                                font-size: 12px;
                                color: #333;
                            }}
                            .color-scale-horizontal {{
                                width: 100%;
                                height: 12px;
                                background: linear-gradient(to right, #66b3ff 0%, #b3e0ff 25%, #ffffff 50%, #ffd3b6 75%, #ffa07a 100%);
                                border: 1px solid #ccc;
                                border-radius: 2px;
                                margin: 6px 0 2px 0;
                            }}
                            .scale-labels-horizontal {{
                                display: flex;
                                justify-content: space-between;
                                font-size: 11px;
                                color: #666;
                            }}
                        </style>
                    </head>
                    <body>
                    <div class="figure-container">
                        <div class="top-row">
                            <div class="model-decision-col">
                                <div class="model-decision-label"><b>Model Decision:</b> {decision_choice}</div>
                                <div class="heatmap-tokens">
                    """
                    )

                    # Decision heatmap tokens
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
                            # Make colors stronger for decision heatmap
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
                    html_content.append("</div></div>")

                    # Bottom row: best and worst explanations
                    html_content.append('<div class="bottom-row">')
                    if "explanation_attributions" in scenario and "explanation_outputs" in scenario:
                        explanation_attrs = scenario["explanation_attributions"]
                        explanation_texts = scenario["explanation_outputs"]
                        # Find indices of best and worst explanations
                        best_idx = None
                        worst_idx = None
                        best_spearman = float("-inf")
                        worst_spearman = float("inf")
                        for idx, expl_attr in enumerate(explanation_attrs):
                            spearman_score = calculate_spearman_correlation(
                                scenario["decision_attributions"], expl_attr
                            )
                            if spearman_score is not None:
                                if spearman_score > best_spearman:
                                    best_spearman = spearman_score
                                    best_idx = idx
                                if spearman_score < worst_spearman:
                                    worst_spearman = spearman_score
                                    worst_idx = idx
                        # Output only best and worst explanations
                        for label, idx, spearman in [
                            ("Best Explanation", best_idx, best_spearman),
                            ("Worst Explanation", worst_idx, worst_spearman),
                        ]:
                            if idx is not None:
                                expl_attr = explanation_attrs[idx]
                                expl_text = explanation_texts[idx]
                                cosine_score = calculate_cosine_similarity(scenario["decision_attributions"], expl_attr)
                                html_content.append(f'<div class="explanation-col">')
                                html_content.append(
                                    f'<div class="explanation-title">{label} ('
                                    f'Spearman: <span class="spearman-score">{spearman:.4f}</span>, '
                                    f'Cosine: <span class="spearman-score">{cosine_score:.4f}</span>)</div>'
                                )
                                html_content.append('<div class="heatmap-tokens">')
                                N_TRAILING = 11  # Number of words in the trailing prompt
                                if len(expl_attr) > N_TRAILING:
                                    expl_attr = expl_attr[:-N_TRAILING]
                                valid_scores = []
                                for token_idx, (token, score) in enumerate(expl_attr):
                                    if token_idx >= skip_tokens:
                                        try:
                                            valid_scores.append(float(score))
                                        except (ValueError, TypeError):
                                            continue
                                if valid_scores:
                                    scores_array = np.array(valid_scores)
                                    p95 = np.percentile(scores_array, 95)
                                    p5 = np.percentile(scores_array, 5)
                                else:
                                    p5, p95 = -1, 1
                                for token_idx, (token, score) in enumerate(expl_attr):
                                    if token_idx < skip_tokens:
                                        continue
                                    try:
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
                                        # Make colors stronger for decision heatmap
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
                                # For the explanation text, join the cleaned tokens after skip_tokens
                                html_content.append(f'<div class="explanation-text">{expl_text.strip()}</div>')
                                html_content.append("</div>")
                    html_content.append("</div>")  # Close bottom-row

                    # Color scale
                    html_content.append(
                        """
                        <div class="color-scale-horizontal"></div>
                        <div class="scale-labels-horizontal">
                            <span>-1.0</span>
                            <span>0.0</span>
                            <span>1.0</span>
                        </div>
                    </div>
                    </body>
                    </html>
                    """
                    )

                    # Save as HTML file
                    output_file = f"{model_output_dir}/{dataset_name}_scenario_{i+1}_heatmap.html"
                    with open(output_file, "w") as f:
                        f.write("\n".join(html_content))

            except Exception as e:
                print(f"Error creating visualization for scenario {i+1}: {e}")
                continue


if __name__ == "__main__":

    file_path = "data/collection_data/ecqa/unsloth_Llama-3.2-3B-Instruct/ecqa_20250415_184138_LIME_llama3.2/ecqa_20250415_184138_LIME_llama3.2.jsonl"

    # Create heatmaps for feature attributions
    create_attribution_heatmap(file_path, num_scenarios=300)
