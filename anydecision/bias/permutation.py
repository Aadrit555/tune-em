"""Permutation generation and option-order invariance analysis."""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Sequence, Tuple
import numpy as np


def generate_permutations(
    items: Sequence[str],
    max_permutations: int = 4,
    seed: int = 42,
) -> List[List[str]]:
    """Generate deterministic permutations of candidate items.

    Always includes:
    1. Canonical order
    2. Reversal (if len >= 2)
    3. Cyclic shifts
    4. Deterministic pseudo-random permutations up to max_permutations.
    """
    n = len(items)
    if n <= 1:
        return [list(items)]

    items_list = list(items)
    permutations: List[List[str]] = [items_list.copy()]

    # 1. Reverse
    rev = list(reversed(items_list))
    if rev not in permutations:
        permutations.append(rev)

    # 2. Cyclic shift
    for shift in range(1, min(n, max_permutations)):
        shifted = items_list[shift:] + items_list[:shift]
        if shifted not in permutations and len(permutations) < max_permutations:
            permutations.append(shifted)

    # 3. If more needed, generate deterministic shuffles
    if len(permutations) < max_permutations:
        rng = random.Random(seed)
        all_perms = list(itertools.permutations(items_list))
        rng.shuffle(all_perms)
        for p in all_perms:
            p_list = list(p)
            if p_list not in permutations:
                permutations.append(p_list)
            if len(permutations) >= max_permutations:
                break

    return permutations[:max_permutations]


def compute_permutation_invariance_metrics(
    permutation_distributions: List[Dict[str, float]],
    option_keys: List[str],
) -> Dict[str, float]:
    """Compute quantitative sensitivity and bias metrics across option permutations.

    Args:
        permutation_distributions: List of {key: prob} for each permutation.
        option_keys: Canonical keys.

    Returns:
        Dict with metrics:
        - raw_positional_variance
        - debiased_variance
        - permutation_agreement: fraction of permutations sharing the modal winner
        - rank_stability: stability of top-1 ranking [0, 1]
        - position_bias_score: mean total variation distance between permutations [0, 1]
    """
    if not permutation_distributions:
        return {
            "raw_positional_variance": 0.0,
            "debiased_variance": 0.0,
            "permutation_agreement": 1.0,
            "rank_stability": 1.0,
            "position_bias_score": 0.0,
        }

    k = len(option_keys)
    m = len(permutation_distributions)

    # Matrix: [m, k]
    prob_matrix = np.zeros((m, k), dtype=np.float64)
    winners: List[str] = []

    for i, dist in enumerate(permutation_distributions):
        row = [dist.get(opt, 0.0) for opt in option_keys]
        prob_matrix[i, :] = row
        winner = max(dist.items(), key=lambda kv: kv[1])[0]
        winners.append(winner)

    # Modal winner agreement
    from collections import Counter
    counts = Counter(winners)
    most_common_count = counts.most_common(1)[0][1]
    agreement = float(most_common_count / m)

    # Positional variance across rows for each option
    var_per_opt = np.var(prob_matrix, axis=0)
    raw_var = float(np.mean(var_per_opt))

    # Mean distribution
    mean_dist = np.mean(prob_matrix, axis=0)
    debiased_var = float(np.var(mean_dist))

    # Total variation distance between each permutation distribution and mean
    tv_distances = []
    for i in range(m):
        tv = 0.5 * np.sum(np.abs(prob_matrix[i, :] - mean_dist))
        tv_distances.append(tv)
    position_bias_score = float(np.mean(tv_distances))

    # Rank stability: check if the argmax changes
    rank_stability = 1.0 if agreement == 1.0 else agreement

    return {
        "raw_positional_variance": raw_var,
        "debiased_variance": debiased_var,
        "permutation_agreement": agreement,
        "rank_stability": rank_stability,
        "position_bias_score": position_bias_score,
    }

