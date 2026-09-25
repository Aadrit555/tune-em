"""Decision execution plan compiler and optimization inspector."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from anydecision.core.policies import DecisionPolicy
from anydecision.core.question import Question
from anydecision.core.types import ReadoutStrategy


class CompiledExecutionPlan(BaseModel):
    """An optimized, deterministic execution plan for evaluating a typed question."""
    question_id: str
    question_text: str
    target_level: str
    readout_strategy: str
    num_candidate_options: int
    prompt_templates: List[str]
    num_permutations: int
    total_forward_passes: int
    backend_execution: str
    calibration_method: Optional[str]
    selective_abstention_enabled: bool
    utility_optimization_enabled: bool
    estimated_tokens_per_pass: int
    estimated_latency_ms: float

    def explain(self) -> str:
        """Return a formatted terminal-style breakdown of the compiled execution pipeline."""
        lines = [
            "=" * 60,
            f"  COMPILED DECISION EXECUTION PLAN :: [{self.question_id}]",
            "=" * 60,
            f"Target Decision Level:        {self.target_level}",
            f"Readout Strategy:             {self.readout_strategy}",
            f"Candidate Options:            {self.num_candidate_options} choices",
            f"Prompt Templates:             {', '.join(self.prompt_templates)}",
            f"Permutation Plan:             {self.num_permutations} orderings",
            f"Total Backend Forward Passes: {self.total_forward_passes}",
            f"Backend Engine:               {self.backend_execution}",
            f"Calibration Strategy:         {self.calibration_method or 'None (Raw L0/L1)'}",
            f"Selective Abstention:         {'Active' if self.selective_abstention_enabled else 'Disabled'}",
            f"Decision-Theoretic Utility:   {'Active (EU maximization)' if self.utility_optimization_enabled else 'Direct Argmax'}",
            "-" * 60,
            f"Estimated Token Overhead:     ~{self.estimated_tokens_per_pass * self.total_forward_passes} tokens",
            f"Estimated End-to-End Latency: ~{self.estimated_latency_ms:.2f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)


class DecisionCompiler:
    """Compiles a Question and DecisionPolicy into an optimized execution graph."""

    @staticmethod
    def compile_plan(
        question: Question,
        policy: DecisionPolicy,
        backend_name: str = "mock",
        calibration_name: Optional[str] = None,
        has_utility_matrix: bool = False,
    ) -> CompiledExecutionPlan:
        # Determine number of templates
        templates = (
            ["minimal"]
            if policy.level.value == "L0"
            else policy.templates
        )
        # Determine permutations
        num_perms = 1 if policy.level.value == "L0" else policy.num_permutations
        total_passes = len(templates) * num_perms

        # Readout strategy
        is_multi = question.readout_strategy == ReadoutStrategy.MULTI_TOKEN_SEQUENCE
        strategy_str = "multi_token_sequence" if is_multi else "next_token_logits"

        # Estimates
        approx_q_tokens = len(question.text.split()) + 30
        approx_latency = 0.15 * total_passes if backend_name == "mock" else 2.5 * total_passes

        return CompiledExecutionPlan(
            question_id=question.id,
            question_text=question.text[:80] + ("..." if len(question.text) > 80 else ""),
            target_level=policy.level.value,
            readout_strategy=strategy_str,
            num_candidate_options=len(question.options),
            prompt_templates=templates,
            num_permutations=num_perms,
            total_forward_passes=total_passes,
            backend_execution=backend_name,
            calibration_method=calibration_name,
            selective_abstention_enabled=policy.abstention.allow_abstain,
            utility_optimization_enabled=has_utility_matrix or policy.utility_matrix is not None,
            estimated_tokens_per_pass=approx_q_tokens,
            estimated_latency_ms=approx_latency,
        )
