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
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
        "custom_diverging", ["#ffa07a", "#ffd3b6", "#ffffff", "#b3e0ff", "#66b3ff"]
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
                        p95 = np.percentile(scores_array, 95)
                        p5 = np.percentile(scores_array, 5)
                        abs_max = max(abs(p95), abs(p5))

                        print(f"Score statistics (after removing system prompt):")
                        print(f"  5th percentile: {p5:.4f}")
                        print(f"  95th percentile: {p95:.4f}")
                        print(f"  Abs Max (95th): {abs_max:.4f}")
                        print(f"  Number of valid tokens: {len(valid_scores)}")
                    else:
                        p5, p95, abs_max = -1, 1, 1

                    # Create HTML content with colored text
                    html_content = []
                    html_content.append(
                        """
                    <html>
                    <head>
                        <style>
                            body { font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6; }
                            .container { display: flex; flex-direction: column; align-items: flex-start; gap: 0px; }
                            .text-content {
                                width: fit-content;
                                max-width: 100%;
                                margin: 0 auto;
                            }
                            .color-scale-horizontal {
                                width: fit-content;
                                min-width: 300px;
                                max-width: 100%;
                                height: 20px;
                                background: linear-gradient(to right, #66b3ff, #b3e0ff, #ffffff, #ffd3b6, #ffa07a);
                                position: relative;
                                border: 1px solid #ccc;
                                border-radius: 3px;
                                margin: 10px auto 0 auto;
                                display: flex;
                                align-items: center;
                            }
                            .scale-labels-horizontal {
                                width: fit-content;
                                min-width: 300px;
                                max-width: 100%;
                                display: flex;
                                flex-direction: row;
                                justify-content: space-between;
                                margin: 2px auto 0 auto;
                                font-size: 12px;
                                color: #666;
                            }
                            .info-box {
                                background-color: #f5f5f5;
                                border: 1px solid #ddd;
                                border-radius: 5px;
                                padding: 10px;
                                margin: 16px auto 0 auto;
                                width: fit-content;
                                max-width: 100%;
                            }
                            .info-box h3 {
                                margin-top: 0;
                                color: #333;
                                font-size: 14px;
                            }
                            .info-box p {
                                margin: 5px 0;
                                font-size: 13px;
                            }
                            .spearman-score {
                                font-weight: bold;
                                color: green;
                            }
                        </style>
                    </head>
                    <body>
                    <div class="container">
                        <div class="text-content">
                    """
                    )

                    # Step 3: Create visualization using normalized scores
                    for token_idx, (token, score) in enumerate(attributions):
                        try:
                            # Skip system prompt tokens for Qwen/Llama
                            if token_idx < skip_tokens:
                                continue

                            score = float(score)
                            # Clip score to percentile range
                            clipped_score = np.clip(score, p5, p95)
                            # Normalize score to -1 to 1 range
                            normalized_score = (clipped_score - p5) / (p95 - p5) * 2 - 1

                            # Get RGB color from colormap (convert -1 to 1 range to 0 to 1 range)
                            color = cmap((normalized_score + 1) / 2)
                            # Convert RGB to hex
                            color_hex = "#{:02x}{:02x}{:02x}".format(
                                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                            )

                            # Clean the token for display
                            clean_token_text = clean_token(token)
                            if clean_token_text:  # Only add non-empty tokens
                                # Add word with background color and score as tooltip
                                html_content.append(
                                    f'<span class="word" style="background-color: {color_hex}" title="Score: {score:.3f}">{clean_token_text}</span>'
                                )
                        except (ValueError, TypeError):
                            # If score can't be converted to float, use neutral color
                            clean_token_text = clean_token(token)
                            if clean_token_text:  # Only add non-empty tokens
                                html_content.append(f'<span class="word">{clean_token_text}</span>')

                    # Close text-content div and add horizontal color scale
                    html_content.append(
                        f"""
                        </div>
                        <div class="color-scale-horizontal"></div>
                        <div class="scale-labels-horizontal">
                            <span>-1.0</span>
                            <span>0.0</span>
                            <span>1.0</span>
                        </div>
                    </div>
                    """
                    )
                    # Now add the info box below the container, aligned left
                    html_content.append(
                        f"""
                    <div class="info-box">
                        <h3>Model Decision: {decision_choice}</h3>
                    """
                    )
                    if best_explanation is not None:
                        html_content.append(
                            f"""
                        <h3>Best Explanation (Spearman: <span class="spearman-score">{best_spearman:.4f}</span>)</h3>
                        <p>{best_explanation}</p>
                        """
                        )
                    if worst_explanation is not None:
                        html_content.append(
                            f"""
                        <h3>Worst Explanation (Spearman: <span class="spearman-score">{worst_spearman:.4f}</span>)</h3>
                        <p>{worst_explanation}</p>
                        """
                        )
                    html_content.append(
                        """
                    </div>
                    </body>
                    </html>
                    """
                    )

                    # Save as HTML file
                    output_file = f"{output_dir}/scenario_{i+1}_attributions.html"
                    with open(output_file, "w") as f:
                        f.write("\n".join(html_content))

            except Exception as e:
                print(f"Error creating visualization for scenario {i+1}: {e}")
                continue


if __name__ == "__main__":

    file_path = "data/collection_data/ecqa/unsloth_Meta-Llama-3.1-8B-Instruct/ecqa_20250404_120218_LIME_llama3.1/ecqa_20250404_120218_LIME_llama3.1_fixed.jsonl"

    # Create heatmaps for feature attributions
    create_attribution_heatmap(file_path, num_scenarios=10)
