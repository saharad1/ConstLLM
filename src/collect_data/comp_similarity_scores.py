from typing import Any, List, Tuple, Union

import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import rel_entr, softmax
from scipy.stats import entropy, spearmanr


def calculate_spearman_correlation(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
):
    return _calculate_spearman(decision_attributions, explanation_attributions)["correlation"]


def calculate_cosine_similarity(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
):
    return _calculate_cosine_similarity(decision_attributions, explanation_attributions)["cosine_similarity"]


def calculate_lma(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
):
    return _calculate_lma(decision_attributions, explanation_attributions)["lma_score"]


def calculate_jaccard_similarity(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
    top_k: int = 10,
):
    return _calculate_jaccard_similarity(decision_attributions, explanation_attributions, top_k)["jaccard_similarity"]


def calculate_jaccard_similarities(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
    top_k_values: List[int] = [10, 20],
):
    """Calculate Jaccard similarity for multiple top-k values."""
    results = {}
    for top_k in top_k_values:
        jaccard_score = _calculate_jaccard_similarity(decision_attributions, explanation_attributions, top_k)[
            "jaccard_similarity"
        ]
        results[f"jaccard_{top_k}"] = jaccard_score
    return results


def compute_kl_divergence(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
):
    return _calculate_kl_divergence(decision_attributions, explanation_attributions)["kl_divergence"]


def _align_tokens(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
) -> Tuple[List[float], List[float]]:
    """
    Align tokens and scores using a more robust matching approach.
    Only includes tokens that appear in both lists.

    Args:
        decision_attributions: List of (token, score) tuples for decision.
        explanation_attributions: List of (token, score) tuples for explanation.
    Returns:
        Aligned decision and explanation scores.
    """
    # Create dictionaries for faster lookup
    decision_dict = dict(decision_attributions)
    explanation_dict = dict(explanation_attributions)

    # Get only tokens that appear in both lists
    common_tokens = set(decision_dict.keys()) & set(explanation_dict.keys())

    aligned_decision_scores = []
    aligned_explanation_scores = []

    # For each common token, get scores from both dictionaries
    for token in common_tokens:
        decision_score = decision_dict[token]
        explanation_score = explanation_dict[token]

        aligned_decision_scores.append(decision_score)
        aligned_explanation_scores.append(explanation_score)

    return aligned_decision_scores, aligned_explanation_scores


def _calculate_spearman(
    decision_scores: List[Tuple[str, float]],
    explanation_scores: List[Tuple[str, float]],
) -> Union[dict[str, Union[float, Any]], dict[str, None]]:
    """
    Calculate Spearman correlation and p-value after aligning tokens.
    Uses absolute values of the scores as per the paper.

    Args:
        decision_scores: List of (token, score) tuples for decision.
        explanation_scores: List of (token, score) tuples for explanation.

    Returns:
        A dictionary containing the Spearman correlation coefficient and p-value.
    """
    aligned_decision_scores, aligned_explanation_scores = _align_tokens(decision_scores, explanation_scores)
    if len(aligned_decision_scores) == len(aligned_explanation_scores):
        # Convert to numpy arrays and take absolute values
        decision_array = np.abs(np.array(aligned_decision_scores, dtype=np.float64))
        explanation_array = np.abs(np.array(aligned_explanation_scores, dtype=np.float64))

        correlation, p_value = spearmanr(decision_array, explanation_array)
        return {"correlation": correlation, "p_value": p_value}
    return {"correlation": None, "p_value": None}


def _calculate_cosine_similarity(
    decision_scores: List[Tuple[str, float]],
    explanation_scores: List[Tuple[str, float]],
) -> Union[dict[str, Union[float, Any]], dict[str, None]]:
    """
    Calculate cosine similarity between raw attribution scores.
    Uses L1 normalization of raw values (not absolute values).

    Args:
        decision_scores: List of (token, score) tuples for decision.
        explanation_scores: List of (token, score) tuples for explanation.

    Returns:
        A dictionary containing the cosine similarity.
    """
    aligned_decision_scores, aligned_explanation_scores = _align_tokens(decision_scores, explanation_scores)

    if len(aligned_decision_scores) == len(aligned_explanation_scores) and len(aligned_decision_scores) > 0:
        # Convert lists to numpy arrays
        decision_array = np.array(aligned_decision_scores, dtype=np.float64)
        explanation_array = np.array(aligned_explanation_scores, dtype=np.float64)

        # L1 normalization without taking absolute values
        # Using sum of absolute values for normalization to avoid division by zero
        # when positive and negative values cancel each other
        decision_norm = np.sum(np.abs(decision_array))
        explanation_norm = np.sum(np.abs(explanation_array))

        if decision_norm > 0 and explanation_norm > 0:
            decision_array = decision_array / decision_norm
            explanation_array = explanation_array / explanation_norm

            # Calculate cosine similarity
            cosine_sim = np.dot(decision_array, explanation_array)
            return {"cosine_similarity": cosine_sim}

        return {"cosine_similarity": 0.0}  # Return 0 if either vector is zero

    return {"cosine_similarity": None}


def _calculate_kl_divergence(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
):
    decision_scores, explanation_scores = _align_tokens(decision_attributions, explanation_attributions)

    if len(decision_scores) == len(explanation_scores) and len(decision_scores) > 0:
        # Convert to numpy arrays
        decision_scores = np.array(decision_scores, dtype=np.float64)
        explanation_scores = np.array(explanation_scores, dtype=np.float64)

        # Convert to probability distributions using scipy's softmax
        decision_probs = softmax(decision_scores)
        explanation_probs = softmax(explanation_scores)

        # Compute KL divergence
        kl_div = entropy(decision_probs, explanation_probs)

        return {"kl_divergence": kl_div}

    return {"kl_divergence": None}


# Local Monotonicity Alignment (LMA) metric
def _calculate_lma(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
) -> Union[float, None]:
    """
    Calculate Local Monotonicity Alignment (LMA) between decision and explanation attributions.
    LMA is the proportion of token pairs whose relative importance ordering is preserved.

    Args:
        decision_attributions: List of (token, score) tuples for decision.
        explanation_attributions: List of (token, score) tuples for explanation.

    Returns:
        A float between 0 and 1 indicating the proportion of consistent pairwise orderings,
        or None if not enough aligned tokens exist.
    """
    aligned_decision_scores, aligned_explanation_scores = _align_tokens(decision_attributions, explanation_attributions)
    if len(aligned_decision_scores) == len(aligned_explanation_scores):
        n = len(aligned_decision_scores)
        if n < 2:
            return None  # Need at least 2 tokens to form a pair

        total_pairs = 0
        consistent_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                d_diff = aligned_decision_scores[i] - aligned_decision_scores[j]
                e_diff = aligned_explanation_scores[i] - aligned_explanation_scores[j]

                if d_diff == 0 and e_diff == 0:
                    consistent_pairs += 1
                elif d_diff * e_diff > 0:
                    consistent_pairs += 1

                total_pairs += 1

        lma = consistent_pairs / total_pairs if total_pairs > 0 else None
        return {"lma_score": lma}
    else:
        return {"lma_score": None}


def _calculate_jaccard_similarity(
    decision_attributions: List[Tuple[str, float]],
    explanation_attributions: List[Tuple[str, float]],
    top_k: int = 10,
) -> Union[dict[str, Union[float, Any]], dict[str, None]]:
    """
    Calculate Jaccard similarity between decision and explanation attributions.
    Jaccard similarity is computed as the intersection over union of the top-k tokens
    with highest absolute attribution values.

    Args:
        decision_attributions: List of (token, score) tuples for decision.
        explanation_attributions: List of (token, score) tuples for explanation.
        top_k: Number of top tokens to consider for Jaccard similarity (default: 10).

    Returns:
        A dictionary containing the Jaccard similarity score.
    """
    if not decision_attributions or not explanation_attributions:
        return {"jaccard_similarity": None}

    # Get top-k tokens based on absolute attribution values
    decision_top_k = set(
        [token for token, _ in sorted(decision_attributions, key=lambda x: abs(x[1]), reverse=True)[:top_k]]
    )
    explanation_top_k = set(
        [token for token, _ in sorted(explanation_attributions, key=lambda x: abs(x[1]), reverse=True)[:top_k]]
    )

    # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = len(decision_top_k & explanation_top_k)
    union = len(decision_top_k | explanation_top_k)

    if union == 0:
        return {"jaccard_similarity": 0.0}

    jaccard_sim = intersection / union
    return {"jaccard_similarity": jaccard_sim}
