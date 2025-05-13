import ast
import json
from typing import Any, Dict

from src.collect_data.comp_similarity_scores import (
    calculate_cosine_similarity,
    calculate_spearman_correlation,
)


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


def compute_explanation_ranks(scenario, metric_type="spearman"):
    """
    Given a scenario dict, compute and return a sorted list of explanation scores
    (best to worst by the specified metric_type: 'spearman' or 'cosine').
    """
    decision_attr = scenario.get("decision_attributions", None)
    explanation_attrs = scenario.get("explanation_attributions", [])
    explanation_texts = scenario.get("explanation_outputs", [])

    if decision_attr is None or not explanation_attrs:
        return []

    explanation_scores = []
    for j, expl_attr in enumerate(explanation_attrs):
        spearman_score = calculate_spearman_correlation(decision_attr, expl_attr)
        cosine_score = calculate_cosine_similarity(decision_attr, expl_attr)
        explanation_text = explanation_texts[j] if explanation_texts and j < len(explanation_texts) else "N/A"
        explanation_scores.append(
            {
                "index": j,
                "text": explanation_text,
                "spearman": spearman_score,
                "cosine": cosine_score,
            }
        )
    # Sort explanations by the chosen metric
    explanation_scores.sort(
        key=lambda x: x[metric_type] if x[metric_type] is not None else float("-inf"),
        reverse=True,
    )
    return explanation_scores
