"""Statistically rigorous calibration and selective prediction evaluation metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width binning.

    ECE = sum_{b=1}^B ( |B_b| / N ) * | acc(B_b) - conf(B_b) |
    """
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(accuracies, dtype=np.float64)
    n = len(conf)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (conf >= low) & (conf <= high if i == num_bins - 1 else conf < high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(acc[mask])
            bin_conf = np.mean(conf[mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_adaptive_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute Adaptive ECE using equal-frequency / equal-mass binning."""
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(accuracies, dtype=np.float64)
    n = len(conf)
    if n == 0:
        return 0.0

    # Sort by confidence
    sort_idx = np.argsort(conf)
    sorted_conf = conf[sort_idx]
    sorted_acc = acc[sort_idx]

    bin_size = max(1, n // num_bins)
    aece = 0.0

    for i in range(0, n, bin_size):
        b_conf = sorted_conf[i:i + bin_size]
        b_acc = sorted_acc[i:i + bin_size]
        b_n = len(b_conf)
        if b_n > 0:
            aece += (b_n / n) * np.abs(np.mean(b_acc) - np.mean(b_conf))

    return float(aece)


def compute_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute multi-class Brier score: (1/N) * sum_n ||p_n - y_n||_2^2."""
    probs = np.asarray(probabilities, dtype=np.float64)
    lbls = np.asarray(labels, dtype=np.int64)
    n, k = probs.shape
    if n == 0:
        return 0.0

    # Construct one-hot matrix
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), lbls] = 1.0

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
    return float(brier)


def compute_nll(
    probabilities: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute Negative Log-Likelihood (NLL)."""
    probs = np.asarray(probabilities, dtype=np.float64)
    lbls = np.asarray(labels, dtype=np.int64)
    n = len(lbls)
    if n == 0:
        return 0.0

    clipped = np.clip(probs[np.arange(n), lbls], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def compute_reliability_diagram_data(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> List[Dict[str, Any]]:
    """Generate detailed binning data for plotting reliability diagrams."""
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(accuracies, dtype=np.float64)
    n = len(conf)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    diagram_bins = []

    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (conf >= low) & (conf <= high if i == num_bins - 1 else conf < high)
        count = int(np.sum(mask))
        bin_acc = float(np.mean(acc[mask])) if count > 0 else 0.0
        bin_conf = float(np.mean(conf[mask])) if count > 0 else float((low + high) / 2)
        diagram_bins.append({
            "bin_index": i,
            "range": [float(low), float(high)],
            "count": count,
            "weight": float(count / n) if n > 0 else 0.0,
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "gap": float(np.abs(bin_acc - bin_conf)),
        })

    return diagram_bins


def compute_selective_prediction_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_thresholds: int = 20,
) -> List[Dict[str, float]]:
    """Compute Risk-Coverage and Selective Accuracy curve across confidence thresholds.

    Coverage = (selected decisions / total)
    Risk = 1.0 - selective_accuracy
    """
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(accuracies, dtype=np.float64)
    n = len(conf)
    if n == 0:
        return []

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    curve = []

    for th in thresholds:
        selected_mask = conf >= th
        count = np.sum(selected_mask)
        if count > 0:
            coverage = float(count / n)
            selective_acc = float(np.mean(acc[selected_mask]))
            risk = 1.0 - selective_acc
        else:
            coverage = 0.0
            selective_acc = 1.0
            risk = 0.0

        curve.append({
            "threshold": float(th),
            "coverage": coverage,
            "selective_accuracy": selective_acc,
            "risk": risk,
            "abstention_rate": 1.0 - coverage,
        })

    return curve

