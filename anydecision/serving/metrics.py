"""Observability, telemetry metrics, and Prometheus-compatible exposition."""

from __future__ import annotations

import time
from typing import Any, Dict, List
import numpy as np


class SystemMetrics:
    """Tracks latency percentiles, decision counters, and cache hit rates."""

    def __init__(self) -> None:
        self.decision_count: int = 0
        self.abstention_count: int = 0
        self.calibration_count: int = 0
        self.errors_count: int = 0
        self.backend_calls: int = 0
        self.latencies_ms: List[float] = []

    def record_decision(self, latency_ms: float, abstained: bool = False, calls: int = 1) -> None:
        self.decision_count += 1
        if abstained:
            self.abstention_count += 1
        self.backend_calls += calls
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 10000:
            self.latencies_ms.pop(0)

    def record_calibration(self) -> None:
        self.calibration_count += 1

    def record_error(self) -> None:
        self.errors_count += 1

    def to_dict(self) -> Dict[str, Any]:
        lats = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        return {
            "decision_count": self.decision_count,
            "abstention_count": self.abstention_count,
            "abstention_rate": float(self.abstention_count / self.decision_count) if self.decision_count > 0 else 0.0,
            "calibration_count": self.calibration_count,
            "errors": self.errors_count,
            "backend_calls": self.backend_calls,
            "average_latency_ms": float(np.mean(lats)),
            "p50_latency_ms": float(np.percentile(lats, 50)),
            "p95_latency_ms": float(np.percentile(lats, 95)),
            "p99_latency_ms": float(np.percentile(lats, 99)),
        }

    def prometheus_format(self) -> str:
        d = self.to_dict()
        lines = [
            f"# HELP anydecision_decisions_total Total decisions processed",
            f"# TYPE anydecision_decisions_total counter",
            f"anydecision_decisions_total {d['decision_count']}",
            f"# HELP anydecision_abstentions_total Total abstained decisions",
            f"# TYPE anydecision_abstentions_total counter",
            f"anydecision_abstentions_total {d['abstention_count']}",
            f"# HELP anydecision_latency_ms_avg Average latency in ms",
            f"# TYPE anydecision_latency_ms_avg gauge",
            f"anydecision_latency_ms_avg {d['average_latency_ms']:.2f}",
            f"# HELP anydecision_latency_ms_p95 95th percentile latency in ms",
            f"# TYPE anydecision_latency_ms_p95 gauge",
            f"anydecision_latency_ms_p95 {d['p95_latency_ms']:.2f}",
        ]
        return "\n".join(lines) + "\n"


GLOBAL_METRICS = SystemMetrics()
