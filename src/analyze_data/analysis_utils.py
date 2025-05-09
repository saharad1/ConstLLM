import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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
