"""Platt scaling and logistic post-hoc calibration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from anydecision.calibration.base import BaseCalibrator
from anydecision.scoring.normalization import softmax


class PlattScaling(BaseCalibrator):
    """Logistic calibration trained on pseudo-logits or model probability outputs."""

    def __init__(self, c_reg: float = 1.0) -> None:
        self.c_reg = c_reg
        self.classifier: Optional[LogisticRegression] = None
        self.fitted = False

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_keys: Optional[List[str]] = None,
    ) -> PlattScaling:
        probs = np.asarray(probabilities, dtype=np.float64)
        lbls = np.asarray(labels, dtype=np.int64)

        eps = 1e-12
        logits = np.log(np.clip(probs, eps, 1.0 - eps))

        self.classifier = LogisticRegression(
            C=self.c_reg,
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
        )
        self.classifier.fit(logits, lbls)
        self.fitted = True
        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.fitted or self.classifier is None:
            return probabilities

        probs = np.asarray(probabilities, dtype=np.float64)
        is_1d = probs.ndim == 1
        if is_1d:
            probs = probs[np.newaxis, :]

        eps = 1e-12
        logits = np.log(np.clip(probs, eps, 1.0 - eps))
        calibrated_probs = self.classifier.predict_proba(logits)

        return calibrated_probs.squeeze(0) if is_1d else calibrated_probs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "platt_scaling",
            "c_reg": self.c_reg,
            "coef": self.classifier.coef_.tolist() if self.classifier is not None else [],
            "intercept": self.classifier.intercept_.tolist() if self.classifier is not None else [],
            "classes": self.classifier.classes_.tolist() if self.classifier is not None else [],
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlattScaling:
        inst = cls(c_reg=data.get("c_reg", 1.0))
        if data.get("fitted") and data.get("coef"):
            clf = LogisticRegression()
            clf.coef_ = np.array(data["coef"], dtype=np.float64)
            clf.intercept_ = np.array(data["intercept"], dtype=np.float64)
            clf.classes_ = np.array(data["classes"], dtype=np.int64)
            inst.classifier = clf
            inst.fitted = True
        return inst
