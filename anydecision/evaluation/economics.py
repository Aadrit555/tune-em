"""Decision Economics and Compute Efficiency benchmark metrics.

Quantifies:
- Quality / Compute (Accuracy / FLOPs or Backend Forward Passes)
- Quality / Dollar (Safe Decisions per $1.00 at specified SLAs)
- Safe Decisions / sec (Throughput of validated, non-abstained decisions)
- Latency vs. Accuracy Trade-Off Curves
- Early-Exit Savings Ratio (Adaptive routing vs fixed full execution)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

from anydecision.adaptive.router import AdaptiveComputeConfig, AdaptiveComputeRouter
from anydecision.core.decision import Decision
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.core.types import DecisionLevel
from anydecision.evaluation.benchmark import BenchmarkSample


class EconomicsConfig(BaseModel):
    """Pricing and compute assumptions for cost modeling."""
    cost_per_million_tokens: float = Field(
        default=0.50,
        description="Cost in USD per 1M tokens processed (e.g., hosted/cloud endpoints)."
    )
    hardware_cost_per_hour: float = Field(
        default=1.50,
        description="Amortized GPU/server cost per hour for self-hosted instances."
    )
    default_human_review_cost: float = Field(
        default=0.25,
        description="Cost in USD for human agent escalation / review per case."
    )
    default_error_cost: float = Field(
        default=2.50,
        description="Estimated loss / penalty in USD for an incorrect uncalibrated decision."
    )


class LevelComparisonRecord(BaseModel):
    """Economics performance record for a single evaluation mode."""
    level: str
    accuracy: float
    abstention_rate: float
    safe_decisions_per_sec: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_backend_calls: float
    mean_tokens: float
    cost_per_1k_decisions_usd: float
    net_economic_value_per_1k_usd: float


class DecisionEconomicsReport(BaseModel):
    """Structured report comparing compute, latency, safety, and dollar economics."""
    total_samples: int
    adaptive_accuracy: float
    adaptive_abstention_rate: float
    early_exit_savings_pct: float
    compute_path_distribution: Dict[str, int]
    mean_adaptive_latency_ms: float
    mean_adaptive_backend_calls: float
    safe_decisions_per_sec: float
    cost_per_1k_decisions_usd: float
    levels_comparison: List[LevelComparisonRecord]
    latency_accuracy_frontier: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "==================================================================",
            "                   DECISION ECONOMICS BENCHMARK                   ",
            "==================================================================",
            f"Total Evaluated Decisions:   {self.total_samples:,}",
            f"Adaptive Accuracy:           {self.adaptive_accuracy * 100:.2f}%",
            f"Adaptive Abstention Rate:    {self.adaptive_abstention_rate * 100:.2f}%",
            f"Early-Exit Compute Savings:  {self.early_exit_savings_pct:.1f}%",
            f"Mean Latency (Adaptive):     {self.mean_adaptive_latency_ms:.2f} ms",
            f"Safe Decisions / Sec:        {self.safe_decisions_per_sec:.2f}",
            f"Cost / 1,000 Decisions:      ${self.cost_per_1k_decisions_usd:.4f} USD",
            "",
            "--- Compute Path Routing Distribution ---",
        ]
        for path, count in sorted(self.compute_path_distribution.items()):
            pct = (count / max(1, self.total_samples)) * 100.0
            lines.append(f"  * Path [{path}]: {count:,} ({pct:.1f}%)")

        lines.extend([
            "",
            "--- Multi-Level Economic Efficiency Frontier ---",
            f"{'Level':<10} | {'Accuracy':<10} | {'Mean Lat (ms)':<14} | {'Calls':<8} | {'Cost/1k USD':<12} | {'Safe Dec/s':<10}",
            "-" * 72,
        ])
        for rec in self.levels_comparison:
            lines.append(
                f"{rec.level:<10} | {rec.accuracy*100:>8.2f}% | {rec.mean_latency_ms:>12.2f} | "
                f"{rec.mean_backend_calls:>6.1f} | ${rec.cost_per_1k_decisions_usd:>10.4f} | {rec.safe_decisions_per_sec:>9.1f}"
            )
        lines.append("==================================================================")
        return "\n".join(lines)


class DecisionEconomicsEvaluator:
    """Evaluates cost, throughput, latency, and compute savings of decision policies."""

    def __init__(
        self,
        engine: DecisionEngine,
        economics_config: Optional[EconomicsConfig] = None,
        adaptive_config: Optional[AdaptiveComputeConfig] = None,
    ) -> None:
        self.engine = engine
        self.config = economics_config or EconomicsConfig()
        self.router = AdaptiveComputeRouter(adaptive_config or AdaptiveComputeConfig())

    def evaluate(
        self,
        samples: Sequence[BenchmarkSample],
        compare_static_levels: bool = True,
    ) -> DecisionEconomicsReport:
        """Run comprehensive decision economics evaluation on a labeled dataset."""
        n = len(samples)
        if n == 0:
            raise ValueError("Evaluation samples sequence cannot be empty.")

        # 1. Run Adaptive Compute Decisions
        adaptive_decisions: List[Decision] = []
        path_counts: Dict[str, int] = {}
        t0_adaptive = time.perf_counter()

        for sample in samples:
            dec = self.router.decide_adaptive(self.engine, sample.question)
            adaptive_decisions.append(dec)
            path_key = " -> ".join(dec.compute_path) if dec.compute_path else "L0"
            path_counts[path_key] = path_counts.get(path_key, 0) + 1

        total_adaptive_time = max(1e-6, time.perf_counter() - t0_adaptive)

        # Accuracy & Safety
        correct = sum(
            1 for d, s in zip(adaptive_decisions, samples)
            if not d.abstained and d.answer == s.ground_truth
        )
        abstained = sum(1 for d in adaptive_decisions if d.abstained)
        decided_count = n - abstained
        adaptive_acc = (correct / decided_count) if decided_count > 0 else 0.0
        abstention_rate = abstained / n
        safe_decisions_per_sec = correct / total_adaptive_time

        latencies = [d.latency_ms for d in adaptive_decisions]
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        calls = [d.backend_calls for d in adaptive_decisions]
        mean_calls = float(np.mean(calls)) if calls else 1.0
        tokens = [d.tokens_processed for d in adaptive_decisions]
        mean_tokens = float(np.mean(tokens)) if tokens else 50.0

        # Cost Modeling
        total_tokens = sum(tokens)
        token_cost_usd = (total_tokens / 1_000_000.0) * self.config.cost_per_million_tokens
        compute_hour_cost_usd = (total_adaptive_time / 3600.0) * self.config.hardware_cost_per_hour
        total_direct_cost = token_cost_usd + compute_hour_cost_usd
        cost_per_1k = (total_direct_cost / n) * 1000.0

        # 2. Compare against static levels if requested
        comparison_records: List[LevelComparisonRecord] = []
        frontier_points: List[Dict[str, float]] = []

        levels_to_test = [DecisionLevel.L0, DecisionLevel.L1]
        if self.engine.calibrator is not None:
            levels_to_test.append(DecisionLevel.L2)

        # Baseline worst-case calls (e.g. if we had run full L1 or L2 on all samples)
        worst_case_calls_per_sample = 4 if DecisionLevel.L1 in levels_to_test else 1
        total_worst_case_calls = n * worst_case_calls_per_sample
        actual_adaptive_calls = sum(calls)
        savings_pct = max(0.0, 100.0 * (1.0 - (actual_adaptive_calls / max(1, total_worst_case_calls))))

        if compare_static_levels:
            for lvl in levels_to_test:
                lvl_decs: List[Decision] = []
                t0_lvl = time.perf_counter()
                for s in samples:
                    d = self.engine.decide(s.question, level=lvl)
                    lvl_decs.append(d)
                lvl_time = max(1e-6, time.perf_counter() - t0_lvl)

                lvl_correct = sum(
                    1 for d, s in zip(lvl_decs, samples)
                    if not d.abstained and d.answer == s.ground_truth
                )
                lvl_abstained = sum(1 for d in lvl_decs if d.abstained)
                lvl_decided = n - lvl_abstained
                lvl_acc = (lvl_correct / lvl_decided) if lvl_decided > 0 else 0.0
                lvl_abstention_rate = lvl_abstained / n
                lvl_safe_per_sec = lvl_correct / lvl_time

                lvl_calls = [
                    d.diagnostics.number_of_backend_calls if d.diagnostics else 1
                    for d in lvl_decs
                ]
                lvl_latencies = [
                    d.diagnostics.latency_ms if (d.diagnostics and d.diagnostics.latency_ms > 0) else (lvl_time / n * 1000)
                    for d in lvl_decs
                ]
                m_calls = float(np.mean(lvl_calls))
                m_lat = float(np.mean(lvl_latencies))
                p95_lat = float(np.percentile(lvl_latencies, 95))
                m_tok = m_calls * (sum(len(s.question.text.split()) for s in samples) / n + 20)

                lvl_cost_usd = ((m_tok * n) / 1_000_000.0) * self.config.cost_per_million_tokens + (lvl_time / 3600.0) * self.config.hardware_cost_per_hour
                lvl_cost_per_1k = (lvl_cost_usd / n) * 1000.0

                # Value: reward correct, penalize errors, penalize review
                net_val = (
                    (lvl_correct * 1.0)
                    - ((lvl_decided - lvl_correct) * self.config.default_error_cost)
                    - (lvl_abstained * self.config.default_human_review_cost)
                    - lvl_cost_usd
                )
                net_val_1k = (net_val / n) * 1000.0

                comparison_records.append(
                    LevelComparisonRecord(
                        level=lvl.value,
                        accuracy=lvl_acc,
                        abstention_rate=lvl_abstention_rate,
                        safe_decisions_per_sec=lvl_safe_per_sec,
                        mean_latency_ms=m_lat,
                        p95_latency_ms=p95_lat,
                        mean_backend_calls=m_calls,
                        mean_tokens=m_tok,
                        cost_per_1k_decisions_usd=lvl_cost_per_1k,
                        net_economic_value_per_1k_usd=net_val_1k,
                    )
                )
                frontier_points.append({
                    "level": lvl.value,
                    "accuracy": lvl_acc,
                    "latency_ms": m_lat,
                    "calls": m_calls,
                    "cost_per_1k": lvl_cost_per_1k,
                })

        # Add adaptive to comparison table
        adaptive_net_val = (
            (correct * 1.0)
            - ((decided_count - correct) * self.config.default_error_cost)
            - (abstained * self.config.default_human_review_cost)
            - total_direct_cost
        )
        adaptive_rec = LevelComparisonRecord(
            level="Adaptive",
            accuracy=adaptive_acc,
            abstention_rate=abstention_rate,
            safe_decisions_per_sec=safe_decisions_per_sec,
            mean_latency_ms=mean_lat,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else mean_lat,
            mean_backend_calls=mean_calls,
            mean_tokens=mean_tokens,
            cost_per_1k_decisions_usd=cost_per_1k,
            net_economic_value_per_1k_usd=(adaptive_net_val / n) * 1000.0,
        )
        comparison_records.append(adaptive_rec)
        frontier_points.append({
            "level": "Adaptive",
            "accuracy": adaptive_acc,
            "latency_ms": mean_lat,
            "calls": mean_calls,
            "cost_per_1k": cost_per_1k,
        })

        return DecisionEconomicsReport(
            total_samples=n,
            adaptive_accuracy=adaptive_acc,
            adaptive_abstention_rate=abstention_rate,
            early_exit_savings_pct=savings_pct,
            compute_path_distribution=path_counts,
            mean_adaptive_latency_ms=mean_lat,
            mean_adaptive_backend_calls=mean_calls,
            safe_decisions_per_sec=safe_decisions_per_sec,
            cost_per_1k_decisions_usd=cost_per_1k,
            levels_comparison=comparison_records,
            latency_accuracy_frontier=frontier_points,
            metadata={"config": self.config.model_dump()},
        )
