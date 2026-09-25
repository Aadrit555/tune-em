"""Selective prediction and decision policies."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from anydecision.core.types import DecisionLevel


class AbstentionPolicy(BaseModel):
    """Selective prediction policy determining when to abstain from a decision.

    Guarantees selective risk control: if confidence is below threshold, or
    entropy/risk exceeds permissible tolerance, the decision engine explicitly
    abstains rather than returning an unreliable answer.
    """
    allow_abstain: bool = Field(
        default=True,
        description="Whether abstention is permitted."
    )
    min_confidence: Optional[float] = Field(
        default=None,
        description="Minimum probability required on the top answer. E.g. 0.85"
    )
    max_uncertainty: Optional[float] = Field(
        default=None,
        description="Maximum permissible uncertainty (1.0 - confidence)."
    )
    max_entropy: Optional[float] = Field(
        default=None,
        description="Maximum Shannon entropy (in nats) before abstaining."
    )
    target_error: Optional[float] = Field(
        default=None,
        description="Maximum permissible risk or expected error rate (e.g. 0.05 for 95% selective accuracy)."
    )

    def evaluate(
        self,
        confidence: float,
        probabilities: Dict[str, float],
        risk: float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate if decision should abstain based on confidence, distribution, and risk."""
        if not self.allow_abstain:
            return False, None

        # Check target error / risk threshold
        if self.target_error is not None:
            # If expected risk exceeds target error, abstain
            effective_risk = risk if risk > 0 else (1.0 - confidence)
            if effective_risk > self.target_error:
                return True, f"risk_{effective_risk:.3f}_exceeds_target_error_{self.target_error:.3f}"

        # Check minimum confidence threshold
        if self.min_confidence is not None and confidence < self.min_confidence:
            return True, f"confidence_{confidence:.3f}_below_min_{self.min_confidence:.3f}"

        # Check max uncertainty
        uncertainty = 1.0 - confidence
        if self.max_uncertainty is not None and uncertainty > self.max_uncertainty:
            return True, f"uncertainty_{uncertainty:.3f}_exceeds_max_{self.max_uncertainty:.3f}"

        # Check max entropy
        if self.max_entropy is not None:
            entropy = -sum(p * math.log(p + 1e-12) for p in probabilities.values() if p > 0)
            if entropy > self.max_entropy:
                return True, f"entropy_{entropy:.3f}_exceeds_max_{self.max_entropy:.3f}"

        return False, None


class DecisionPolicy(BaseModel):
    """Configuration governing readout, debiasing, calibration, and output formatting."""
    level: DecisionLevel = Field(
        default=DecisionLevel.L0,
        description="Target decision level: L0 (raw), L1 (zero-label), L2 (calibrated)."
    )
    abstention: AbstentionPolicy = Field(
        default_factory=AbstentionPolicy,
        description="Selective prediction / abstention criteria."
    )
    # L1 Zero-label debiasing configuration
    num_permutations: int = Field(
        default=4,
        description="Number of option permutations for position debiasing (L1)."
    )
    templates: List[str] = Field(
        default_factory=lambda: ["minimal", "structured"],
        description="Prompt template names for ensemble aggregation."
    )
    aggregation_method: str = Field(
        default="prob_mean",
        description="Aggregation strategy across permutations/templates ('prob_mean', 'logit_mean', 'harmonic', 'minimax')."
    )
    # L2 Calibration configuration
    calibration_method: str = Field(
        default="temperature",
        description="Calibration algorithm ('temperature', 'vector', 'isotonic', 'platt', 'conformal')."
    )
    conformal_alpha: float = Field(
        default=0.10,
        description="Error coverage for conformal prediction sets (1 - alpha coverage)."
    )
    # Performance & Diagnostics
    use_prefix_cache: bool = Field(
        default=True,
        description="Enable KV prefix reuse where backend supports it."
    )
    enable_trace: bool = Field(
        default=False,
        description="Generate full step-by-step decision trace."
    )
    enable_ood_diagnostics: bool = Field(
        default=True,
        description="Compute out-of-distribution and distribution-shift heuristics."
    )

