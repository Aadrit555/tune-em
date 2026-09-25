"""Vector scaling and diagonal affine calibration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize

from anydecision.calibration.base import BaseCalibrator
from anydecision.scoring.normalization import softmax


class VectorScaling(BaseCalibrator):
    """Calibrates predictions by learning class-specific diagonal weights and biases:

    z'_k = W_k * z_k + b_k
    """

    def __init__(self, weights: Optional[List[float]] = None, biases: Optional[List[float]] = None) -> None:
        self.weights = np.array(weights, dtype=np.float64) if weights is not None else None
        self.biases = np.array(biases, dtype=np.float64) if biases is not None else None
        self.num_classes = len(weights) if weights is not None else 0
        self.fitted = weights is not None

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_keys: Optional[List[str]] = None,
    ) -> VectorScaling:
        probs = np.asarray(probabilities, dtype=np.float64)
        lbls = np.asarray(labels, dtype=np.int64)
        n, k = probs.shape
        self.num_classes = k

        eps = 1e-12
        logits = np.log(np.clip(probs, eps, 1.0 - eps))

        # Initial parameters: weights=1.0, biases=0.0
        init_params = np.concatenate([np.ones(k), np.zeros(k)])

        def objective(params: np.ndarray) -> float:
            w = params[:k]
            b = params[k:]
            scaled = logits * w + b
            max_s = np.max(scaled, axis=-1, keepdims=True)
            log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=-1, keepdims=True))
            log_p = scaled - log_sum_exp
            nll = -np.mean(log_p[np.arange(n), lbls])
            # L2 regularization on weights and bias to avoid overfitting
            reg = 0.01 * (np.sum((w - 1.0) ** 2) + np.sum(b ** 2))
            return float(nll + reg)

        bounds = [(0.01, 10.0)] * k + [(-5.0, 5.0)] * k
        res = minimize(objective, init_params, method="L-BFGS-B", bounds=bounds)
        if res.success:
            self.weights = res.x[:k]
            self.biases = res.x[k:]
        else:
            self.weights = np.ones(k)
            self.biases = np.zeros(k)

        self.fitted = True
        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=np.float64)
        eps = 1e-12
        logits = np.log(np.clip(probs, eps, 1.0 - eps))
        if self.weights is None or self.biases is None:
            return probs
        scaled = logits * self.weights + self.biases
        return softmax(scaled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "vector_scaling",
            "weights": self.weights.tolist() if self.weights is not None else [],
            "biases": self.biases.tolist() if self.biases is not None else [],
            "num_classes": self.num_classes,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VectorScaling:
        inst = cls(weights=data.get("weights"), biases=data.get("biases"))
        inst.fitted = data.get("fitted", True)
        return inst

