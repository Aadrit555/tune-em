"""Calibration report generator and evaluation summary."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from anydecision.calibration.metrics import (
    compute_adaptive_ece,
    compute_brier_score,
    compute_ece,
    compute_nll,
    compute_reliability_diagram_data,
    compute_selective_prediction_curve,
)


class CalibrationReport(BaseModel):
    """Statistical calibration and selective prediction report."""
    num_samples: int
    accuracy: float
    ece: float
    adaptive_ece: float
    brier_score: float
    nll: float
    mean_confidence: float
    selective_accuracy_at_80_coverage: float
    coverage_at_target_error_5pct: float
    abstention_rate: float
    reliability_diagram: List[Dict[str, Any]] = Field(default_factory=list)
    risk_coverage_curve: List[Dict[str, float]] = Field(default_factory=list)

    def summary(self) -> str:
        """Formatted human-readable summary table."""
        lines = [
            "=" * 50,
            "       ANYDECISION STATISTICAL CALIBRATION REPORT",
            "=" * 50,
            f"Samples Evaluated:             {self.num_samples:,}",
            f"Raw Accuracy:                  {self.accuracy * 100:.1f}%",
            f"Mean Confidence:               {self.mean_confidence * 100:.1f}%",
            f"Expected Calibration Error:    {self.ece:.4f}",
            f"Adaptive ECE:                  {self.adaptive_ece:.4f}",
            f"Brier Score:                   {self.brier_score:.4f}",
            f"Negative Log-Likelihood:       {self.nll:.4f}",
            "-" * 50,
            "Selective Prediction (Risk-Coverage):",
            f"Selective Acc (80% Coverage):  {self.selective_accuracy_at_80_coverage * 100:.1f}%",
            f"Coverage @ 5% Target Error:    {self.coverage_at_target_error_5pct * 100:.1f}%",
            f"Abstention Rate @ 5% Target:   {self.abstention_rate * 100:.1f}%",
            "=" * 50,
            "[STATISTICAL TRANSPARENCY NOTICE]",
            "A calibrated probability indicates empirical frequency across exchangeable",
            "samples. It does NOT guarantee correctness on individual out-of-distribution",
            "instances or adversarial shift.",
            "=" * 50,
        ]
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent)

    @classmethod
    def evaluate(
        cls,
        probabilities: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
    ) -> CalibrationReport:
        """Compute full statistical calibration report from probability predictions and true labels."""
        probs = np.asarray(probabilities, dtype=np.float64)
        lbls = np.asarray(labels, dtype=np.int64)
        n = len(lbls)
        if n == 0:
            return cls(
                num_samples=0,
                accuracy=0.0,
                ece=0.0,
                adaptive_ece=0.0,
                brier_score=0.0,
                nll=0.0,
                mean_confidence=0.0,
                selective_accuracy_at_80_coverage=0.0,
                coverage_at_target_error_5pct=0.0,
                abstention_rate=0.0,
            )

        preds = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        accuracies = (preds == lbls).astype(np.float64)

        acc = float(np.mean(accuracies))
        mean_conf = float(np.mean(confidences))
        ece = compute_ece(confidences, accuracies, num_bins=num_bins)
        aece = compute_adaptive_ece(confidences, accuracies, num_bins=num_bins)
        brier = compute_brier_score(probs, lbls)
        nll = compute_nll(probs, lbls)

        diag = compute_reliability_diagram_data(confidences, accuracies, num_bins=num_bins)
        curve = compute_selective_prediction_curve(confidences, accuracies, num_thresholds=20)

        # Selective accuracy at ~80% coverage
        sel_acc_80 = acc
        cov_5pct = 0.0
        abst_5pct = 1.0

        for pt in curve:
            if pt["coverage"] <= 0.80 and sel_acc_80 == acc:
                sel_acc_80 = pt["selective_accuracy"]
            if pt["risk"] <= 0.05 and pt["coverage"] > cov_5pct:
                cov_5pct = pt["coverage"]
                abst_5pct = pt["abstention_rate"]

        return cls(
            num_samples=n,
            accuracy=acc,
            ece=ece,
            adaptive_ece=aece,
            brier_score=brier,
            nll=nll,
            mean_confidence=mean_conf,
            selective_accuracy_at_80_coverage=sel_acc_80,
            coverage_at_target_error_5pct=cov_5pct,
            abstention_rate=abst_5pct,
            reliability_diagram=diag,
            risk_coverage_curve=curve,
        )
