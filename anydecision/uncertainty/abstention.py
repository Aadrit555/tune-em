"""Abstention controller and selective decision rules."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from anydecision.core.policies import AbstentionPolicy


class AbstentionController:
    """Manages selective prediction and decides whether to output a decision or abstain."""

    def __init__(self, policy: Optional[AbstentionPolicy] = None) -> None:
        self.policy = policy or AbstentionPolicy()

    def evaluate(
        self,
        confidence: float,
        probabilities: Dict[str, float],
        calibrated_risk: float,
        policy_override: Optional[AbstentionPolicy] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Determine whether to abstain.

        Returns:
            Tuple of (should_abstain: bool, reason: Optional[str])
        """
        active_policy = policy_override or self.policy
        return active_policy.evaluate(
            confidence=confidence,
            probabilities=probabilities,
            risk=calibrated_risk,
        )


def find_optimal_rejection_threshold(
    val_confidences: List[float],
    val_accuracies: List[float],
    target_error: float = 0.05,
) -> float:
    """Find the minimal confidence threshold that guarantees empirical error <= target_error."""
    import numpy as np

    confs = np.asarray(val_confidences)
    accs = np.asarray(val_accuracies)

    thresholds = np.sort(confs)
    best_th = 1.0

    for th in thresholds:
        mask = confs >= th
        if np.sum(mask) == 0:
            continue
        err = 1.0 - np.mean(accs[mask])
        if err <= target_error:
            best_th = float(th)
            break

    return best_th

