

from typing import Any, List, Tuple, Union
from scipy.stats import spearmanr

def compute_spearman_score(decision_attributions: List[Tuple[str, float]], explanation_attributions: List[Tuple[str, float]]):
    return _calculate_spearman(decision_attributions, explanation_attributions)['correlation']

def _align_tokens(decision_attributions: List[Tuple[str, float]], explanation_attributions: List[Tuple[str, float]]) -> Tuple[
    List[float], List[float]]:
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

def _calculate_spearman(decision_scores: List[Tuple[str, float]],
                        explanation_scores: List[Tuple[str, float]]) -> \
        Union[dict[str, Union[float, Any]], dict[str, None]]:
    """
    Calculate Spearman correlation and p-value after aligning tokens.

    Args:
        decision_scores: List of (token, score) tuples for decision.
        explanation_scores: List of (token, score) tuples for explanation.

    Returns:
        A dictionary containing the Spearman correlation coefficient and p-value.
    """
    aligned_decision_scores, aligned_explanation_scores = _align_tokens(decision_scores, explanation_scores)
    if len(aligned_decision_scores) == len(aligned_explanation_scores):
        correlation, p_value = spearmanr(aligned_decision_scores, aligned_explanation_scores)
        return {"correlation": correlation, "p_value": p_value}
    return {"correlation": None, "p_value": None}



