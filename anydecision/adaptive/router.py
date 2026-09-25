"""Adaptive compute runtime with confidence-triggered early exit and latency SLA."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pydantic import BaseModel, Field

from anydecision.core.decision import Decision
from anydecision.core.policies import DecisionPolicy
from anydecision.core.question import Question
from anydecision.core.types import DecisionLevel

if TYPE_CHECKING:
    from anydecision.core.engine import DecisionEngine


class AdaptiveComputeConfig(BaseModel):
    """Configuration governing adaptive compute and early-exit thresholds."""
    early_exit_l0_confidence: float = Field(
        default=0.90,
        description="Confidence threshold on L0 to exit immediately without running permutations."
    )
    early_exit_l1_confidence: float = Field(
        default=0.80,
        description="Confidence threshold on L1 to exit without requiring L2 recalibration."
    )
    max_l0_entropy: float = Field(
        default=0.45,
        description="Maximum Shannon entropy on L0 to permit early exit."
    )
    min_l1_template_agreement: float = Field(
        default=0.75,
        description="Minimum agreement across templates on L1 to permit early exit."
    )
    max_latency_ms: Optional[float] = Field(
        default=None,
        description="Maximum permissible latency budget."
    )
    max_backend_calls: int = Field(
        default=12,
        description="Maximum total model forward passes allowed before early stopping."
    )


class AdaptiveComputeRouter:
    """Dynamically routes decisions through L0 -> L1 -> L2 -> Abstain based on confidence."""

    def __init__(self, config: Optional[AdaptiveComputeConfig] = None) -> None:
        self.config = config or AdaptiveComputeConfig()

    def decide_adaptive(
        self,
        engine: DecisionEngine,
        question: Question,
        policy: Optional[DecisionPolicy] = None,
        **kwargs: Any,
    ) -> Decision:
        """Execute decision using minimal compute necessary to satisfy confidence requirements."""
        start_time = time.perf_counter()
        active_policy = policy or engine.policy
        compute_path: List[str] = []
        total_backend_calls = 0
        total_tokens = 0

        # Step 1: Run cheap Level L0 (Raw Single Pass)
        dec_l0 = engine.decide(
            question=question,
            level=DecisionLevel.L0,
            policy=active_policy,
            **kwargs,
        )
        compute_path.append("L0")
        total_backend_calls += dec_l0.diagnostics.number_of_backend_calls if dec_l0.diagnostics else 1
        approx_q_tokens = len(question.text.split()) + 20
        total_tokens += approx_q_tokens

        # Check Early Exit on L0: high confidence & low entropy
        entropy_val = dec_l0.diagnostics.entropy if dec_l0.diagnostics else 0.0
        can_exit_l0 = (
            dec_l0.confidence >= self.config.early_exit_l0_confidence
            and entropy_val <= self.config.max_l0_entropy
            and not dec_l0.abstained
        )

        if can_exit_l0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return self._finalize_adaptive_decision(
                decision=dec_l0,
                compute_path=compute_path,
                backend_calls=total_backend_calls,
                tokens=total_tokens,
                latency_ms=elapsed_ms,
                engine=engine,
            )

        # Check Latency SLA
        elapsed_so_far = (time.perf_counter() - start_time) * 1000.0
        if self.config.max_latency_ms and elapsed_so_far >= self.config.max_latency_ms:
            return self._finalize_adaptive_decision(
                decision=dec_l0,
                compute_path=compute_path,
                backend_calls=total_backend_calls,
                tokens=total_tokens,
                latency_ms=elapsed_so_far,
                engine=engine,
            )

        # Step 2: Escalate to Level L1 (Zero-Label Permutation Debiasing)
        dec_l1 = engine.decide(
            question=question,
            level=DecisionLevel.L1,
            policy=active_policy,
            **kwargs,
        )
        compute_path.append("L1")
        l1_calls = dec_l1.diagnostics.number_of_backend_calls if dec_l1.diagnostics else 4
        total_backend_calls += l1_calls
        total_tokens += approx_q_tokens * l1_calls

        # Check Early Exit on L1
        agreement_val = dec_l1.diagnostics.template_agreement if dec_l1.diagnostics else 1.0
        can_exit_l1 = (
            dec_l1.confidence >= self.config.early_exit_l1_confidence
            and agreement_val >= self.config.min_l1_template_agreement
            and not dec_l1.abstained
        )

        # If confident or if no L2 calibrator exists, return L1
        has_l2_calibrator = engine.calibrator is not None or engine.adapter.calibrator.fitted
        if can_exit_l1 or not has_l2_calibrator:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return self._finalize_adaptive_decision(
                decision=dec_l1,
                compute_path=compute_path,
                backend_calls=total_backend_calls,
                tokens=total_tokens,
                latency_ms=elapsed_ms,
                engine=engine,
            )

        # Step 3: Escalate to Level L2 (Statistical Calibration)
        dec_l2 = engine.decide(
            question=question,
            level=DecisionLevel.L2,
            policy=active_policy,
            **kwargs,
        )
        compute_path.append("L2")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._finalize_adaptive_decision(
            decision=dec_l2,
            compute_path=compute_path,
            backend_calls=total_backend_calls,
            tokens=total_tokens,
            latency_ms=elapsed_ms,
            engine=engine,
        )

    def _finalize_adaptive_decision(
        self,
        decision: Decision,
        compute_path: List[str],
        backend_calls: int,
        tokens: int,
        latency_ms: float,
        engine: DecisionEngine,
    ) -> Decision:
        num_layers = engine.metadata.num_layers or 32
        d = decision.model_copy()
        d.compute_path = compute_path
        d.backend_calls = backend_calls
        d.tokens_processed = tokens
        d.latency_ms = latency_ms
        d.layers_executed = num_layers * len(compute_path)

        if d.diagnostics:
            d.diagnostics.compute_path = compute_path
            d.diagnostics.number_of_backend_calls = backend_calls
            d.diagnostics.tokens_processed = tokens
            d.diagnostics.latency_ms = latency_ms
            d.diagnostics.layers_executed = d.layers_executed

        return d
