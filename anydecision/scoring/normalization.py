"""Probability normalization, logit transforms, and numerical utilities."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Union
import numpy as np


def softmax(logits: Union[np.ndarray, Sequence[float]], temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over 1D or 2D array."""
    arr = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-8)
    if arr.ndim == 1:
        shifted = arr - np.max(arr)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)
    else:
        shifted = arr - np.max(arr, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def log_softmax(logits: Union[np.ndarray, Sequence[float]], temperature: float = 1.0) -> np.ndarray:
    """Numerically stable log-softmax."""
    arr = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-8)
    if arr.ndim == 1:
        shifted = arr - np.max(arr)
        return shifted - np.log(np.sum(np.exp(shifted)))
    else:
        shifted = arr - np.max(arr, axis=-1, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def normalize_log_probabilities(logprobs: Dict[str, float]) -> Dict[str, float]:
    """Normalize a mapping of candidate -> log-probability into proper probabilities summing to 1.0."""
    if not logprobs:
        return {}
    keys = list(logprobs.keys())
    vals = np.array([logprobs[k] for k in keys], dtype=np.float64)
    probs = softmax(vals)
    return {k: float(p) for k, p in zip(keys, probs)}


def compute_entropy(probabilities: Sequence[float]) -> float:
    """Compute Shannon entropy in nats from probability sequence."""
    entropy = 0.0
    for p in probabilities:
        if p > 1e-12:
            entropy -= p * math.log(p)
    return float(entropy)

