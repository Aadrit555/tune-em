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

    def _measure_prompt_tokens(self, engine: DecisionEngine, question: Question) -> tuple[int, bool]:
        """Exact tokenizer count when the backend provides one, else labeled estimate."""
        try:
            n = int(engine.backend.count_tokens(question.text))
        except Exception:
            n = max(1, len(question.text.split()))
            return n, True
        exact = bool(getattr(engine.backend, "token_counts_exact", False))
        return n, (not exact)

    def _predict_calls(self, engine: DecisionEngine, level: DecisionLevel, question: Question) -> int:
        """Predict forward passes for a stage without executing it (budget pre-check)."""
        try:
            if level == DecisionLevel.L0:
                return len(question.options) if question.answer_type.value == "multi_choice" else 1
            n_perms = int(getattr(engine.policy, "num_permutations", 4))
            n_templates = len(getattr(engine.policy, "templates", ["minimal"]) or ["minimal"])
            if level == DecisionLevel.L1:
                return max(1, n_perms * max(1, n_templates))
            return max(1, n_perms * max(1, n_templates))
        except Exception:
            return 1

    def decide_adaptive(
        self,
        engine: DecisionEngine,
        question: Question,
        policy: Optional[DecisionPolicy] = None,
        **kwargs: Any,
    ) -> Decision:
        """Execute decision using minimal compute necessary to satisfy confidence requirements.

        Token accounting uses the backend tokenizer when available
        (``tokens_estimated=False``); otherwise counts are whitespace estimates
        (``tokens_estimated=True``).  ``max_backend_calls`` is enforced as a
        hard budget: a stage is never started when its predicted calls would
        exceed the remaining budget.
        """
        start_time = time.perf_counter()
        active_policy = policy or engine.policy
        compute_path: List[str] = []
        total_backend_calls = 0
        total_tokens = 0

        prompt_tokens, tokens_estimated = self._measure_prompt_tokens(engine, question)
        per_call_tokens = prompt_tokens + 20  # fixed framing overhead per forward pass

        def elapsed_ms() -> float:
            return (time.perf_counter() - start_time) * 1000.0

        def finalize(decision: Decision, exit_reason: str) -> Decision:
            return self._finalize_adaptive_decision(
                decision=decision,
                compute_path=compute_path,
                backend_calls=total_backend_calls,
                tokens=total_tokens,
                tokens_estimated=tokens_estimated,
                exit_reason=exit_reason,
                latency_ms=elapsed_ms(),
                engine=engine,
            )

        # Step 1: L0 (single pass; always within budget unless max < 1)
        if total_backend_calls + self._predict_calls(engine, DecisionLevel.L0, question) > self.config.max_backend_calls:
            raise ValueError(
                f"max_backend_calls={self.config.max_backend_calls} is too small to run even L0."
            )
        dec_l0 = engine.decide(
            question=question,
            level=DecisionLevel.L0,
            policy=active_policy,
            **kwargs,
        )
        compute_path.append("L0")
        l0_calls = dec_l0.diagnostics.number_of_backend_calls if dec_l0.diagnostics else 1
        total_backend_calls += l0_calls
        total_tokens += per_call_tokens * l0_calls

        entropy_val = dec_l0.diagnostics.entropy if dec_l0.diagnostics else 0.0
        can_exit_l0 = (
            dec_l0.confidence >= self.config.early_exit_l0_confidence
            and entropy_val <= self.config.max_l0_entropy
            and not dec_l0.abstained
        )
        if can_exit_l0:
            return finalize(dec_l0, "l0_confident")
        if self.config.max_latency_ms and elapsed_ms() >= self.config.max_latency_ms:
            return finalize(dec_l0, "latency_sla")
        if total_backend_calls >= self.config.max_backend_calls:
            return finalize(dec_l0, "budget_exhausted")

        # Step 2: L1 (predicted-budget pre-check: never start if it would exceed)
        if total_backend_calls + self._predict_calls(engine, DecisionLevel.L1, question) > self.config.max_backend_calls:
            return finalize(dec_l0, "budget_exhausted")
        dec_l1 = engine.decide(
            question=question,
            level=DecisionLevel.L1,
            policy=active_policy,
            **kwargs,
        )
        compute_path.append("L1")
        l1_calls = dec_l1.diagnostics.number_of_backend_calls if dec_l1.diagnostics else 4
        total_backend_calls += l1_calls
        total_tokens += per_call_tokens * l1_calls

        agreement_val = dec_l1.diagnostics.template_agreement if dec_l1.diagnostics else 1.0
        can_exit_l1 = (
            dec_l1.confidence >= self.config.early_exit_l1_confidence
            and agreement_val >= self.config.min_l1_template_agreement
            and not dec_l1.abstained
        )
        has_l2_calibrator = engine.calibrator is not None or engine.adapter.calibrator.fitted
        if can_exit_l1:
            return finalize(dec_l1, "l1_confident")
        if not has_l2_calibrator:
            return finalize(dec_l1, "no_l2_calibrator")
        if self.config.max_latency_ms and elapsed_ms() >= self.config.max_latency_ms:
            return finalize(dec_l1, "latency_sla")
        if total_backend_calls >= self.config.max_backend_calls:
            return finalize(dec_l1, "budget_exhausted")

        # Step 3: L2 (predicted-budget pre-check)
        if total_backend_calls + self._predict_calls(engine, DecisionLevel.L2, question) > self.config.max_backend_calls:
            return finalize(dec_l1, "budget_exhausted")
        dec_l2 = engine.decide(
            question=question,
            level=DecisionLevel.L2,
            policy=active_policy,
            **kwargs,
        )
        compute_path.append("L2")
        l2_calls = dec_l2.diagnostics.number_of_backend_calls if dec_l2.diagnostics else 1
        total_backend_calls += l2_calls
        total_tokens += per_call_tokens * l2_calls

        return finalize(dec_l2, "l2_complete")

    def _finalize_adaptive_decision(
        self,
        decision: Decision,
        compute_path: List[str],
        backend_calls: int,
        tokens: int,
        tokens_estimated: bool,
        exit_reason: str,
        latency_ms: float,
        engine: DecisionEngine,
    ) -> Decision:
        num_layers = engine.metadata.num_layers or 32
        d = decision.model_copy()
        d.compute_path = compute_path
        d.backend_calls = backend_calls
        d.tokens_processed = tokens
        d.tokens_estimated = tokens_estimated
        d.exit_reason = exit_reason
        d.latency_ms = latency_ms
        d.layers_executed = num_layers * len(compute_path)

        if d.diagnostics:
            d.diagnostics.compute_path = compute_path
            d.diagnostics.number_of_backend_calls = backend_calls
            d.diagnostics.tokens_processed = tokens
            d.diagnostics.tokens_estimated = tokens_estimated
            d.diagnostics.exit_reason = exit_reason
            d.diagnostics.latency_ms = latency_ms
            d.diagnostics.layers_executed = d.layers_executed

        return d
