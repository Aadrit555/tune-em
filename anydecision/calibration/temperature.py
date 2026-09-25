"""Temperature scaling calibration (Platt scaling generalization for multiclass)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize_scalar

from anydecision.calibration.base import BaseCalibrator
from anydecision.scoring.normalization import softmax


class TemperatureScaling(BaseCalibrator):
    """Calibrates predictions by optimizing a single learned temperature parameter T > 0.

    Preserves accuracy and argmax rankings while smoothing or sharpening probabilities
    to align with empirical validation accuracy.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)
        self.fitted = False

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_keys: Optional[List[str]] = None,
    ) -> TemperatureScaling:
        """Find optimal temperature minimizing Negative Log-Likelihood."""
        probs = np.asarray(probabilities, dtype=np.float64)
        lbls = np.asarray(labels, dtype=np.int64)

        # Convert probabilities to unnormalized pseudo-logits: log(p + eps)
        eps = 1e-12
        logits = np.log(np.clip(probs, eps, 1.0 - eps))

        def nll_objective(t: float) -> float:
            scaled_logits = logits / t
            # Log-softmax per row
            max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
            log_sum_exp = max_logits + np.log(np.sum(np.exp(scaled_logits - max_logits), axis=-1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp
            # Gather correct label log-probs
            correct_log_probs = log_probs[np.arange(len(lbls)), lbls]
            return -float(np.mean(correct_log_probs))

        res = minimize_scalar(nll_objective, bounds=(0.05, 10.0), method="bounded")
        if res.success:
            self.temperature = float(res.x)
        self.fitted = True
        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling."""
        probs = np.asarray(probabilities, dtype=np.float64)
        eps = 1e-12
        logits = np.log(np.clip(probs, eps, 1.0 - eps))
        return softmax(logits, temperature=self.temperature)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "temperature_scaling",
            "temperature": self.temperature,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemperatureScaling:
        inst = cls(temperature=data.get("temperature", 1.0))
        inst.fitted = data.get("fitted", True)
        return inst

