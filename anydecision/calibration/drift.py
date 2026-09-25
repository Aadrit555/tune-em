"""Calibration under distribution shift and runtime sequential drift monitoring.

Supports:
- Explicit Shift Modes (Domain, Prompt, Option-ordering, Temporal, Quantization)
- Statistical Drift Quantification (TV distance, ECE degradation, Brier drift)
- Runtime Sequential Calibration Drift Monitor
- Dynamic threshold adaptation under detected covariate / concept drift
"""

from __future__ import annotations

from collections import deque
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

from anydecision.calibration.metrics import compute_brier_score, compute_ece, compute_nll

if TYPE_CHECKING:
    from anydecision.core.decision import Decision
    from anydecision.core.engine import DecisionEngine
    from anydecision.evaluation.benchmark import BenchmarkSample


class DistributionShiftType(str, Enum):
    """Class of distribution perturbation evaluated."""
    IID = "iid"
    DOMAIN_SHIFT = "domain_shift"
    PROMPT_SHIFT = "prompt_shift"
    OPTION_SHIFT = "option_shift"
    TEMPORAL_SHIFT = "temporal_shift"
    QUANTIZATION_SHIFT = "quantization_shift"


class DistributionShiftReport(BaseModel):
    """Comparative report measuring calibration degradation across distribution shifts."""
    shift_type: DistributionShiftType
    source_domain: str
    target_domain: str
    source_ece: float
    target_ece: float
    ece_drift_pct: float
    source_accuracy: float
    target_accuracy: float
    accuracy_drop_pct: float
    source_brier: float
    target_brier: float
    tv_distance_probabilities: float
    recommendation: str

    def summary(self) -> str:
        lines = [
            f"=== Distribution Shift Report ({self.shift_type.value.upper()}) ===",
            f"Source Domain:               {self.source_domain}",
            f"Target Domain:               {self.target_domain}",
            "",
            f"{'Metric':<25} | {'Source':<12} | {'Target (Shifted)':<16} | {'Delta':<12}",
            "-" * 72,
            f"{'Expected Calib Error (ECE)':<25} | {self.source_ece:>10.4f}   | {self.target_ece:>14.4f}   | {self.ece_drift_pct:>+9.1f}%",
            f"{'Accuracy':<25} | {self.source_accuracy*100:>9.2f}%  | {self.target_accuracy*100:>13.2f}%  | {self.accuracy_drop_pct:>+9.1f}%",
            f"{'Brier Score':<25} | {self.source_brier:>10.4f}   | {self.target_brier:>14.4f}   | {self.target_brier - self.source_brier:>+9.4f}",
            "-" * 72,
            f"Total Variation Shift:       {self.tv_distance_probabilities:.4f}",
            f"Production Recommendation:   {self.recommendation}",
            "========================================================================",
        ]
        return "\n".join(lines)


class DriftAlert(BaseModel):
    """Sequential runtime notification when calibration drift exceeds safety limits."""
    severity: str = Field(description="'info', 'warning', or 'critical'")
    drift_score: float = Field(description="Normalized statistical drift score [0.0, 1.0].")
    metric: str = Field(description="Triggering metric, e.g. 'confidence_tv_distance' or 'ece_spike'.")
    message: str
    recommended_action: str


class CalibrationDriftMonitor:
    """Tracks running distribution of decisions and detects sequential calibration drift."""

    def __init__(
        self,
        window_size: int = 100,
        warning_threshold: float = 0.15,
        critical_threshold: float = 0.28,
        min_samples_to_eval: int = 20,
    ) -> None:
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_samples_to_eval = min_samples_to_eval

        self._reference_confidences: List[float] = []
        self._sliding_window: Deque[Decision] = deque(maxlen=window_size)
        self._sliding_labels: Deque[Optional[str]] = deque(maxlen=window_size)
        self._drift_history: List[DriftAlert] = []

    def set_reference_distribution(self, decisions: Sequence[Decision]) -> None:
        """Initialize baseline confidence distribution from certified calibration set."""
        self._reference_confidences = [d.confidence for d in decisions if not d.abstained]

    def record_decision(
        self,
        decision: Decision,
        observed_label: Optional[str] = None,
    ) -> Optional[DriftAlert]:
        """Record live production decision and evaluate whether drift alert triggers."""
        self._sliding_window.append(decision)
        self._sliding_labels.append(observed_label)

        if len(self._sliding_window) < self.min_samples_to_eval:
            return None

        # Compute empirical confidence distribution shift
        current_confs = [d.confidence for d in self._sliding_window if not d.abstained]
        if not current_confs or not self._reference_confidences:
            return None

        # Compute 1D Wasserstein / Total Variation across 10 confidence bins
        bins = np.linspace(0.0, 1.0, 11)
        ref_hist, _ = np.histogram(self._reference_confidences, bins=bins, density=True)
        cur_hist, _ = np.histogram(current_confs, bins=bins, density=True)
        ref_p = ref_hist / (np.sum(ref_hist) + 1e-9)
        cur_p = cur_hist / (np.sum(cur_hist) + 1e-9)
        tv_distance = float(0.5 * np.sum(np.abs(ref_p - cur_p)))

        alert = None
        if tv_distance >= self.critical_threshold:
            alert = DriftAlert(
                severity="critical",
                drift_score=tv_distance,
                metric="confidence_tv_distance",
                message=f"Critical distribution drift detected (TV distance: {tv_distance:.3f} >= {self.critical_threshold:.2f}).",
                recommended_action="Tighten abstention threshold and trigger active recalibration.",
            )
        elif tv_distance >= self.warning_threshold:
            alert = DriftAlert(
                severity="warning",
                drift_score=tv_distance,
                metric="confidence_tv_distance",
                message=f"Warning: Moderate calibration drift detected (TV distance: {tv_distance:.3f}).",
                recommended_action="Monitor closely or fallback to L1 permutation debiasing.",
            )

        if alert:
            self._drift_history.append(alert)
        return alert

    def adapt_abstention_threshold(self, base_threshold: float) -> float:
        """Dynamically increase abstention threshold if recent drift alerts were triggered."""
        if not self._drift_history:
            return base_threshold

        recent_alerts = self._drift_history[-5:]
        max_drift = max(a.drift_score for a in recent_alerts)

        if max_drift >= self.critical_threshold:
            # Shift threshold upward by up to +0.15 to prevent confident hallucinations under drift
            penalty = 0.12 * (max_drift / max(1e-6, self.critical_threshold))
            return min(0.98, float(base_threshold + penalty))
        elif max_drift >= self.warning_threshold:
            penalty = 0.05
            return min(0.95, float(base_threshold + penalty))

        return base_threshold


class DistributionShiftEvaluator:
    """Evaluates calibration degradation when a calibrated model is transferred across distributions."""

    @staticmethod
    def evaluate_shift(
        source_samples: Sequence[BenchmarkSample],
        target_samples: Sequence[BenchmarkSample],
        engine: DecisionEngine,
        shift_type: DistributionShiftType = DistributionShiftType.DOMAIN_SHIFT,
        source_name: str = "source_domain",
        target_name: str = "target_domain",
    ) -> DistributionShiftReport:
        # Evaluate Source
        src_decs = [engine.decide(s.question, level="L2" if engine.calibrator else "L1") for s in source_samples]
        src_confs = np.array([d.confidence for d in src_decs])
        src_correct = np.array([1 if d.answer == s.ground_truth else 0 for d, s in zip(src_decs, source_samples)])
        src_acc = float(np.mean(src_correct))
        src_ece = compute_ece(src_confs, src_correct)

        keys = source_samples[0].question.option_keys()
        k_to_idx = {k: i for i, k in enumerate(keys)}
        src_probs = np.array([[d.probabilities.get(k, 0.0) for k in keys] for d in src_decs])
        src_lbls = np.array([k_to_idx.get(s.ground_truth, 0) for s in source_samples])
        src_brier = compute_brier_score(src_probs, src_lbls)

        # Evaluate Target
        tgt_decs = [engine.decide(s.question, level="L2" if engine.calibrator else "L1") for s in target_samples]
        tgt_confs = np.array([d.confidence for d in tgt_decs])
        tgt_correct = np.array([1 if d.answer == s.ground_truth else 0 for d, s in zip(tgt_decs, target_samples)])
        tgt_acc = float(np.mean(tgt_correct))
        tgt_ece = compute_ece(tgt_confs, tgt_correct)

        tgt_probs = np.array([[d.probabilities.get(k, 0.0) for k in keys] for d in tgt_decs])
        tgt_lbls = np.array([k_to_idx.get(s.ground_truth, 0) for s in target_samples])
        tgt_brier = compute_brier_score(tgt_probs, tgt_lbls)

        # Relative drifts
        ece_drift = ((tgt_ece - src_ece) / max(1e-6, src_ece)) * 100.0
        acc_drop = ((tgt_acc - src_acc) / max(1e-6, src_acc)) * 100.0

        # TV distance between source and target predicted distributions
        src_mean_p = np.mean(src_probs, axis=0)
        tgt_mean_p = np.mean(tgt_probs, axis=0)
        tv_dist = float(0.5 * np.sum(np.abs(src_mean_p - tgt_mean_p)))

        if ece_drift > 50.0 or tv_dist > 0.25:
            rec = "CRITICAL: Calibration broke under shift. Immediate recalibration required."
        elif ece_drift > 20.0 or tv_dist > 0.15:
            rec = "WARNING: Moderate calibration degradation. Recommend increasing abstention threshold."
        else:
            rec = "ROBUST: Calibrator demonstrated resilience to this distribution shift."

        return DistributionShiftReport(
            shift_type=shift_type,
            source_domain=source_name,
            target_domain=target_name,
            source_ece=src_ece,
            target_ece=tgt_ece,
            ece_drift_pct=ece_drift,
            source_accuracy=src_acc,
            target_accuracy=tgt_acc,
            accuracy_drop_pct=acc_drop,
            source_brier=src_brier,
            target_brier=tgt_brier,
            tv_distance_probabilities=tv_dist,
            recommendation=rec,
        )
