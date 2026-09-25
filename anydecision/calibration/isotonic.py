"""Isotonic regression post-hoc calibration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression

from anydecision.calibration.base import BaseCalibrator


class IsotonicCalibration(BaseCalibrator):
    """Non-parametric isotonic regression calibrator (one-vs-rest per candidate class).

    Learns monotonic step mappings from predicted probabilities to empirical frequencies.
    """

    def __init__(self) -> None:
        self.regressors: List[IsotonicRegression] = []
        self.num_classes: int = 0
        self.fitted: bool = False

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_keys: Optional[List[str]] = None,
    ) -> IsotonicCalibration:
        probs = np.asarray(probabilities, dtype=np.float64)
        lbls = np.asarray(labels, dtype=np.int64)
        n, k = probs.shape
        self.num_classes = k
        self.regressors = []

        for c in range(k):
            # Binary target: 1 if true class is c else 0
            binary_y = (lbls == c).astype(np.float64)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs[:, c], binary_y)
            self.regressors.append(iso)

        self.fitted = True
        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=np.float64)
        is_1d = probs.ndim == 1
        if is_1d:
            probs = probs[np.newaxis, :]

        n, k = probs.shape
        if not self.regressors or k != self.num_classes:
            return probs.squeeze(0) if is_1d else probs

        calibrated = np.zeros_like(probs)
        for c in range(min(k, len(self.regressors))):
            calibrated[:, c] = self.regressors[c].predict(probs[:, c])

        # Normalize rows to sum to 1.0
        row_sums = np.sum(calibrated, axis=-1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized = calibrated / row_sums

        return normalized.squeeze(0) if is_1d else normalized

    def to_dict(self) -> Dict[str, Any]:
        # Serialize piecewise thresholds
        reg_data = []
        for iso in self.regressors:
            reg_data.append({
                "x_thresholds": iso.X_thresholds_.tolist() if hasattr(iso, "X_thresholds_") else [],
                "y_thresholds": iso.y_thresholds_.tolist() if hasattr(iso, "y_thresholds_") else [],
            })
        return {
            "type": "isotonic_calibration",
            "num_classes": self.num_classes,
            "regressors": reg_data,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IsotonicCalibration:
        inst = cls()
        inst.num_classes = data.get("num_classes", 0)
        inst.regressors = []
        for reg in data.get("regressors", []):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            x_th = np.array(reg.get("x_thresholds", []))
            y_th = np.array(reg.get("y_thresholds", []))
            iso.X_thresholds_ = x_th
            iso.y_thresholds_ = y_th
            iso.f_ = lambda x, x_th=x_th, y_th=y_th: np.interp(x, x_th, y_th)
            inst.regressors.append(iso)
        inst.fitted = data.get("fitted", True)
        return inst

