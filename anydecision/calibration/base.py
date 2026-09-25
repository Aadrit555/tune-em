"""Base class and interface for probability calibrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


class BaseCalibrator(ABC):
    """Abstract base class for lightweight post-hoc calibration layers."""

    @abstractmethod
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_keys: Optional[List[str]] = None,
    ) -> BaseCalibrator:
        """Fit calibration parameters using labeled validation/calibration data.

        Args:
            probabilities: Array of shape [N, K] containing uncalibrated model probabilities.
            labels: Array of shape [N] containing integer class indices (0 to K-1).
            option_keys: Optional list of class key strings.
        """
        pass

    @abstractmethod
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities [N, K] or [K] into calibrated probabilities."""
        pass

    def calibrate_dict(self, prob_dict: Dict[str, float]) -> Dict[str, float]:
        """Convenience method to calibrate a candidate -> probability dictionary."""
        keys = list(prob_dict.keys())
        raw_vals = np.array([prob_dict[k] for k in keys], dtype=np.float64)
        calibrated_vals = self.calibrate(raw_vals)
        # Ensure proper normalization
        calibrated_vals = np.clip(calibrated_vals, 0.0, 1.0)
        total = np.sum(calibrated_vals)
        if total > 0:
            calibrated_vals = calibrated_vals / total
        return {k: float(v) for k, v in zip(keys, calibrated_vals)}

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibrator parameters to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseCalibrator:
        """Instantiate calibrator from serialized dictionary."""
        pass

