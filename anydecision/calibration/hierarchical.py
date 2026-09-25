"""Hierarchical Bayesian Calibration with Empirical Bayes Shrinkage.

Constructs a fallback calibration hierarchy:
    Specific Question ID -> Domain Group / Question Cluster -> Global Prior

For questions with sparse observations, automatically shrinks the local temperature parameter
toward the parent group/global prior:
    theta_effective = lambda * theta_local + (1 - lambda) * theta_parent
    lambda = n / (n + n_0)

Guarantees robust calibration even in cold-start or low-sample regimes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

from anydecision.calibration.base import BaseCalibrator
from anydecision.calibration.temperature import TemperatureScaling
from anydecision.scoring.normalization import softmax


class HierarchicalDiagnostics(BaseModel):
    """Observable shrinkage breakdown for a calibrated query."""
    question_id: Optional[str] = None
    group_id: Optional[str] = None
    local_sample_count: int
    shrinkage_weight_lambda: float
    local_temperature: float
    parent_temperature: float
    effective_temperature: float
    resolution_level: str = Field(
        description="'local_shrunk', 'group_shrunk', or 'global_fallback'"
    )


class HierarchicalCalibrator(BaseCalibrator):
    """Three-tier hierarchical calibrator with empirical Bayes shrinkage."""

    type: str = "hierarchical"

    def __init__(
        self,
        shrinkage_prior_n0: float = 12.0,
        min_samples_local: int = 3,
    ) -> None:
        super().__init__()
        self.shrinkage_prior_n0 = shrinkage_prior_n0
        self.min_samples_local = min_samples_local
        self.global_calibrator = TemperatureScaling()
        self.group_calibrators: Dict[str, TemperatureScaling] = {}
        self.group_sample_counts: Dict[str, int] = {}
        self.local_temperatures: Dict[str, float] = {}
        self.local_sample_counts: Dict[str, int] = {}
        self.fitted = False

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_keys: Optional[Sequence[str]] = None,
    ) -> HierarchicalCalibrator:
        """Fit root global temperature scaling calibrator."""
        return self.fit_global(probabilities, labels, option_keys)

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Calibrate probability array using the root global temperature."""
        return self.global_calibrator.calibrate(probabilities)

    def fit_global(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_keys: Optional[Sequence[str]] = None,
    ) -> HierarchicalCalibrator:
        """Fit the root global temperature scaling calibrator."""
        self.global_calibrator.fit(probabilities, labels, option_keys)
        self.fitted = True
        return self

    def fit_group(
        self,
        group_id: str,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_keys: Optional[Sequence[str]] = None,
    ) -> None:
        """Fit group-level calibrator with empirical shrinkage toward global root."""
        cal = TemperatureScaling()
        cal.fit(probabilities, labels, option_keys)
        n = len(labels)
        self.group_sample_counts[group_id] = n

        # Empirical Bayes shrinkage: lambda = n / (n + n_0)
        lam = float(n / (n + self.shrinkage_prior_n0))
        global_t = self.global_calibrator.temperature if self.global_calibrator.fitted else 1.0
        shrunk_t = (lam * cal.temperature) + ((1.0 - lam) * global_t)
        cal.temperature = float(shrunk_t)
        self.group_calibrators[group_id] = cal

    def register_local_observations(
        self,
        question_id: str,
        temperature: float,
        sample_count: int,
    ) -> None:
        """Register sparse local observations for a specific question ID."""
        self.local_temperatures[question_id] = temperature
        self.local_sample_counts[question_id] = sample_count

    def resolve_effective_temperature(
        self,
        group_id: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> tuple[float, HierarchicalDiagnostics]:
        """Compute effective shrunk temperature and return diagnostic trace."""
        global_t = self.global_calibrator.temperature if self.global_calibrator.fitted else 1.0

        # 1. Determine parent group temperature
        parent_t = global_t
        res_level = "global_fallback"
        if group_id and group_id in self.group_calibrators:
            parent_t = self.group_calibrators[group_id].temperature
            res_level = "group_shrunk"

        # 2. Check local question temperature and apply shrinkage
        n_local = self.local_sample_counts.get(question_id, 0) if question_id else 0
        raw_local_t = self.local_temperatures.get(question_id, parent_t) if question_id else parent_t

        if n_local >= self.min_samples_local:
            lam = float(n_local / (n_local + self.shrinkage_prior_n0))
            eff_t = (lam * raw_local_t) + ((1.0 - lam) * parent_t)
            res_level = "local_shrunk"
        else:
            lam = 0.0
            eff_t = parent_t

        diagnostics = HierarchicalDiagnostics(
            question_id=question_id,
            group_id=group_id,
            local_sample_count=n_local,
            shrinkage_weight_lambda=lam,
            local_temperature=raw_local_t,
            parent_temperature=parent_t,
            effective_temperature=eff_t,
            resolution_level=res_level,
        )
        return eff_t, diagnostics

    def calibrate_dict(
        self,
        probabilities: Dict[str, float],
        group_id: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calibrate probabilities using the hierarchically shrunk temperature."""
        eff_t, _ = self.resolve_effective_temperature(group_id=group_id, question_id=question_id)
        keys = list(probabilities.keys())
        p_arr = np.array([probabilities[k] for k in keys], dtype=np.float64)

        # Scale logits: log(p) / T
        logits = np.log(np.clip(p_arr, 1e-12, 1.0))
        scaled_logits = logits / max(1e-4, eff_t)
        calibrated_p = softmax(scaled_logits)

        return {k: float(p) for k, p in zip(keys, calibrated_p)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "shrinkage_prior_n0": self.shrinkage_prior_n0,
            "global_temperature": self.global_calibrator.temperature,
            "groups": {g: c.temperature for g, c in self.group_calibrators.items()},
            "local_temperatures": self.local_temperatures,
            "local_sample_counts": self.local_sample_counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HierarchicalCalibrator:
        cal = cls(shrinkage_prior_n0=data.get("shrinkage_prior_n0", 12.0))
        cal.global_calibrator.temperature = data.get("global_temperature", 1.0)
        cal.global_calibrator.fitted = True
        cal.fitted = True
        for g, t in data.get("groups", {}).items():
            gc = TemperatureScaling()
            gc.temperature = t
            gc.fitted = True
            cal.group_calibrators[g] = gc
        cal.local_temperatures = data.get("local_temperatures", {})
        cal.local_sample_counts = data.get("local_sample_counts", {})
        return cal
