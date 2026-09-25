"""Entropy, epistemic uncertainty, and information-theoretic metrics."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence
import numpy as np


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Compute normalized Shannon entropy in [0, 1].

    0.0 = completely deterministic (one option has prob 1.0)
    1.0 = maximum uncertainty (uniform distribution 1/K).
    """
    k = len(probabilities)
    if k <= 1:
        return 0.0

    entropy = 0.0
    for p in probabilities:
        if p > 1e-12:
            entropy -= p * math.log(p)

    max_entropy = math.log(k)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_mutual_information(
    ensemble_distributions: List[Dict[str, float]],
    option_keys: List[str],
) -> float:
    """Compute epistemic uncertainty via Mutual Information across ensemble/perturbation runs.

    MI = H(E[p]) - E[H(p)]
    Measures model's lack of knowledge (epistemic disagreement) separate from data noise (aleatoric).
    """
    if len(ensemble_distributions) <= 1:
        return 0.0

    k = len(option_keys)
    m = len(ensemble_distributions)
    prob_matrix = np.zeros((m, k), dtype=np.float64)

    for i, dist in enumerate(ensemble_distributions):
        prob_matrix[i, :] = [dist.get(opt, 0.0) for opt in option_keys]

    # Total predictive entropy H(E[p])
    mean_dist = np.mean(prob_matrix, axis=0)
    total_entropy = 0.0
    for p in mean_dist:
        if p > 1e-12:
            total_entropy -= p * math.log(p)

    # Expected entropy E[H(p)] (aleatoric)
    entropies = []
    for row in prob_matrix:
        h = 0.0
        for p in row:
            if p > 1e-12:
                h -= p * math.log(p)
        entropies.append(h)
    expected_entropy = float(np.mean(entropies))

    # Mutual Information = Total - Expected
    mi = max(0.0, total_entropy - expected_entropy)
    return float(mi)
