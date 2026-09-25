"""Probability and logit aggregation strategies across ensembles and permutations."""

from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
from anydecision.scoring.normalization import softmax


def aggregate_distributions(
    distributions: Sequence[Dict[str, float]],
    method: str = "prob_mean",
) -> Dict[str, float]:
    """Aggregate a sequence of probability distributions over the same keys.

    Args:
        distributions: List of {key: prob} mappings.
        method: 'prob_mean', 'logit_mean', 'harmonic_mean', 'minimax', or 'borda'.

    Returns:
        Aggregated {key: prob} distribution summing to 1.0.
    """
    if not distributions:
        return {}
    if len(distributions) == 1:
        return dict(distributions[0])

    keys = list(distributions[0].keys())
    matrix = np.array([[dist[k] for k in keys] for dist in distributions], dtype=np.float64)

    if method == "prob_mean":
        agg = np.mean(matrix, axis=0)
    elif method == "logit_mean":
        # Transform probs to log domain (logits), average, and softmax
        eps = 1e-12
        clipped = np.clip(matrix, eps, 1.0 - eps)
        log_matrix = np.log(clipped)
        mean_logs = np.mean(log_matrix, axis=0)
        agg = softmax(mean_logs)
    elif method == "harmonic_mean":
        # Harmonic mean: n / sum(1/p)
        eps = 1e-12
        inv_sum = np.sum(1.0 / np.clip(matrix, eps, 1.0), axis=0)
        h_vals = len(distributions) / inv_sum
        agg = h_vals / np.sum(h_vals)
    elif method == "minimax":
        # Conservative lower-bound probability
        min_vals = np.min(matrix, axis=0)
        agg = min_vals / np.sum(min_vals)
    elif method == "borda":
        # Rank-based Borda score: each permutation gives points based on ranking
        # Higher probability gets higher rank
        points = np.zeros(len(keys), dtype=np.float64)
        for row in matrix:
            ranks = np.argsort(np.argsort(row))  # 0 to len(keys)-1
            points += ranks
        agg = (points + 1.0) / np.sum(points + 1.0)
    else:
        # Default to prob_mean
        agg = np.mean(matrix, axis=0)

    # Ensure clean sum to 1.0
    agg = np.clip(agg, 0.0, 1.0)
    total = np.sum(agg)
    if total > 0:
        agg = agg / total

    return {k: float(v) for k, v in zip(keys, agg)}

