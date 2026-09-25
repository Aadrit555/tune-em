"""Comprehensive research benchmark suite for decision readouts and calibration."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from anydecision.calibration.report import CalibrationReport
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.core.types import DecisionLevel


class BenchmarkSample(BaseModel):
    question: Question
    ground_truth: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    benchmark_name: str
    num_samples: int
    level: str
    accuracy: float
    ece: float
    adaptive_ece: float
    brier_score: float
    nll: float
    abstention_rate: float
    selective_accuracy: float
    mean_confidence: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_permutation_agreement: float
    mean_position_bias: float
    total_time_seconds: float
    details: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Benchmark: {self.benchmark_name} (Level: {self.level}) ===",
            f"Evaluated Samples:         {self.num_samples:,}",
            f"Accuracy:                  {self.accuracy * 100:.2f}%",
            f"Expected Calibration Error:{self.ece:.4f}",
            f"Adaptive ECE:              {self.adaptive_ece:.4f}",
            f"Brier Score:               {self.brier_score:.4f}",
            f"Negative Log-Likelihood:   {self.nll:.4f}",
            f"Selective Accuracy:        {self.selective_accuracy * 100:.2f}%",
            f"Abstention Rate:           {self.abstention_rate * 100:.2f}%",
            f"Mean Permutation Agreement:{self.mean_permutation_agreement * 100:.2f}%",
            f"Mean Position Bias Score:  {self.mean_position_bias:.4f}",
            f"Mean Latency:              {self.mean_latency_ms:.2f} ms (p95: {self.p95_latency_ms:.2f} ms)",
            f"Total Benchmark Time:      {self.total_time_seconds:.2f} s",
        ]
        return "\n".join(lines)


class BenchmarkRunner:
    """Evaluates decision engine across datasets and compares L0 vs L1 vs L2."""

    def __init__(self, engine: DecisionEngine) -> None:
        self.engine = engine

    def run(
        self,
        samples: List[BenchmarkSample],
        name: str = "custom_benchmark",
        level: DecisionLevel = DecisionLevel.L0,
        min_confidence: Optional[float] = None,
        target_error: Optional[float] = None,
    ) -> BenchmarkResult:
        start_bench = time.perf_counter()
        latencies = []
        accuracies = []
        confidences = []
        prob_matrix = []
        labels = []
        perm_agreements = []
        bias_scores = []
        abstained_count = 0
        selective_correct = 0
        selective_total = 0

        for sample in samples:
            q = sample.question
            t0 = time.perf_counter()
            dec = self.engine.decide(
                q,
                level=level,
                min_confidence=min_confidence,
                target_error=target_error,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            keys = q.option_keys()
            prob_row = [dec.probabilities.get(k, 0.0) for k in keys]
            prob_matrix.append(prob_row)
            gt = sample.ground_truth
            lbl_idx = keys.index(gt) if gt in keys else 0
            labels.append(lbl_idx)

            is_correct = dec.answer == gt
            accuracies.append(1.0 if is_correct else 0.0)
            confidences.append(dec.confidence)

            if dec.abstained:
                abstained_count += 1
            else:
                selective_total += 1
                if is_correct:
                    selective_correct += 1

            if dec.diagnostics:
                perm_agreements.append(dec.diagnostics.template_agreement)
                bias_scores.append(dec.diagnostics.option_order_sensitivity)

        total_time = time.perf_counter() - start_bench
        probs_np = np.array(prob_matrix, dtype=np.float64)
        labels_np = np.array(labels, dtype=np.int64)

        report = CalibrationReport.evaluate(probs_np, labels_np)

        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        sel_acc = float(selective_correct / selective_total) if selective_total > 0 else 1.0
        abst_rate = float(abstained_count / len(samples)) if samples else 0.0

        return BenchmarkResult(
            benchmark_name=name,
            num_samples=len(samples),
            level=level.value,
            accuracy=report.accuracy,
            ece=report.ece,
            adaptive_ece=report.adaptive_ece,
            brier_score=report.brier_score,
            nll=report.nll,
            abstention_rate=abst_rate,
            selective_accuracy=sel_acc,
            mean_confidence=report.mean_confidence,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            mean_permutation_agreement=float(np.mean(perm_agreements)) if perm_agreements else 1.0,
            mean_position_bias=float(np.mean(bias_scores)) if bias_scores else 0.0,
            total_time_seconds=total_time,
            details={
                "risk_coverage": report.risk_coverage_curve,
                "reliability_diagram": report.reliability_diagram,
            },
        )

