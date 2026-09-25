"""Conformal prediction for finite-sample coverage guarantees and prediction sets."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np


class ConformalPredictor:
    """Split conformal prediction for multiclass classification.

    Guarantees that the true label is contained within the returned prediction set
    with probability at least 1 - alpha on exchangeable test distributions:
        P(Y in C(X)) >= 1 - alpha.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = float(alpha)
        self.q_hat: float = 1.0
        self.fitted = False
        self.n_cal: int = 0

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> ConformalPredictor:
        """Compute the conformal non-conformity quantile on calibration data.

        Non-conformity score s_i = 1 - p(y_i | x_i).
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        lbls = np.asarray(labels, dtype=np.int64)
        n = len(lbls)
        self.n_cal = n

        # Non-conformity score for the true class
        true_class_probs = probs[np.arange(n), lbls]
        scores = 1.0 - true_class_probs

        # Finite-sample adjusted quantile level: ceil((n + 1) * (1 - alpha)) / n
        q_level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))

        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        self.fitted = True
        return self

    def predict_set(
        self,
        probabilities: np.ndarray,
        option_keys: List[str],
    ) -> List[str]:
        """Construct prediction set C(x) = {y : 1 - p(y|x) <= q_hat}."""
        probs = np.asarray(probabilities, dtype=np.float64)
        if not self.fitted:
            # Fallback: return top-1
            best_idx = int(np.argmax(probs))
            return [option_keys[best_idx]]

        pred_set = []
        for idx, key in enumerate(option_keys):
            score = 1.0 - probs[idx]
            if score <= self.q_hat:
                pred_set.append(key)

        # Non-empty guarantee: if empty, include top-1 argmax
        if not pred_set:
            best_idx = int(np.argmax(probs))
            pred_set.append(option_keys[best_idx])

        return pred_set

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "conformal_predictor",
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "n_cal": self.n_cal,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConformalPredictor:
        inst = cls(alpha=data.get("alpha", 0.10))
        inst.q_hat = data.get("q_hat", 1.0)
        inst.n_cal = data.get("n_cal", 0)
        inst.fitted = data.get("fitted", True)
        return inst
