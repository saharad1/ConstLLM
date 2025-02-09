from typing import Any, List, Tuple, Union

import numpy as np
from scipy.special import rel_entr
from scipy.stats import entropy, spearmanr


def compute_spearman_score(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
):
    return _calculate_spearman(decision_attributions, explanation_attributions)[
        "correlation"
    ]


def compute_kl_divergence(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
):
    return _calculate_kl_divergence(decision_attributions, explanation_attributions)[
        "kl_divergence"
    ]


def _align_tokens(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
) -> Tuple[List[float], List[float]]:
    """
    Align tokens and scores.
    Args:
        decision_attributions: List of (token, score) tuples for decision.
        explanation_attributions: List of (token, score) tuples for explanation.
    Returns:
        Aligned decision and explanation scores.
    """
    aligned_decision_scores = []
    aligned_explanation_scores = []

    for i, (decision_token, decision_score) in enumerate(decision_attributions):
        if i >= len(explanation_attributions):
            break

        explanation_token, explanation_score = explanation_attributions[i]
        if decision_token == explanation_token:
            aligned_decision_scores.append(decision_score)
            aligned_explanation_scores.append(explanation_score)

    return aligned_decision_scores, aligned_explanation_scores


def _calculate_spearman(
    decision_scores: List[Tuple[str, float]],
    explanation_scores: List[Tuple[str, float]],
) -> Union[dict[str, Union[float, Any]], dict[str, None]]:
    """
    Calculate Spearman correlation and p-value after aligning tokens.

    Args:
        decision_scores: List of (token, score) tuples for decision.
        explanation_scores: List of (token, score) tuples for explanation.

    Returns:
        A dictionary containing the Spearman correlation coefficient and p-value.
    """
    aligned_decision_scores, aligned_explanation_scores = _align_tokens(
        decision_scores, explanation_scores
    )
    if len(aligned_decision_scores) == len(aligned_explanation_scores):
        correlation, p_value = spearmanr(
            aligned_decision_scores, aligned_explanation_scores
        )
        return {"correlation": correlation, "p_value": p_value}
    return {"correlation": None, "p_value": None}


import numpy as np
from scipy.stats import entropy


def compute_kl_divergence(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
):
    decision_scores, explanation_scores = _align_tokens(
        decision_attributions, explanation_attributions
    )

    if len(decision_scores) == len(explanation_scores) and len(decision_scores) > 0:
        # Convert to numpy arrays
        decision_scores = np.array(decision_scores, dtype=np.float64)
        explanation_scores = np.array(explanation_scores, dtype=np.float64)

        # Apply softmax to transform scores into a valid probability distribution
        def softmax(x):
            exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
            return exp_x / exp_x.sum()

        decision_probs = softmax(decision_scores)
        explanation_probs = softmax(explanation_scores)

        # Compute KL divergence
        kl_div = entropy(decision_probs, explanation_probs)  # ✅ Built-in method

        return {"kl_divergence": kl_div}

    return {"kl_divergence": None}
