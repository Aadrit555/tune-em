"""Strongly typed Decision output object."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from anydecision.core.types import DecisionTrace, Diagnostics


class Decision(BaseModel):
    """A strongly typed decision produced directly from LLM probability distributions.

    Unlike free-form generated text, this contains the mathematically normalized
    probabilities, calibration state, uncertainty quantification, and selective
    prediction / abstention flags.
    """
    answer: Optional[str] = Field(
        default=None,
        description="The winning option key, or None if the engine abstained."
    )
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized probability distribution over all valid candidate keys."
    )
    confidence: float = Field(
        default=0.0,
        description="Probability or calibrated score assigned to the top choice [0.0, 1.0]."
    )
    uncertainty: float = Field(
        default=0.0,
        description="Epistemic or aleatoric uncertainty measure (e.g. 1.0 - confidence or normalized entropy)."
    )
    level: str = Field(
        default="L0",
        description="Decision pipeline level: L0 (raw), L1 (zero-label debiased), L2 (calibrated)."
    )
    method: str = Field(
        default="raw",
        description="Readout / aggregation method used (e.g. 'zero_label', 'temperature_scaling', 'conformal')."
    )
    calibrated: bool = Field(
        default=False,
        description="Whether a statistical calibration transform was applied."
    )
    abstained: bool = Field(
        default=False,
        description="True if the engine decided NOT to force a single prediction due to insufficient confidence or excessive risk."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason code if abstained (e.g. 'insufficient_confidence', 'target_error_exceeded', 'entropy_too_high')."
    )
    risk: float = Field(
        default=0.0,
        description="Estimated risk or posterior error probability for taking this decision."
    )
    prediction_set: Optional[List[str]] = Field(
        default=None,
        description="Calibrated candidate prediction set with coverage guarantees (e.g. from conformal prediction)."
    )
    expected_value: Optional[float] = Field(
        default=None,
        description="Probability-weighted expected value for numeric score or ordinal questions."
    )
    diagnostics: Optional[Diagnostics] = Field(
        default=None,
        description="In-depth observable decision diagnostics and system metrics."
    )
    trace: Optional[DecisionTrace] = Field(
        default=None,
        description="Step-by-step execution trace when requested."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a dictionary representation."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize decision to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        if self.abstained:
            return f"Decision: ABSTAINED (reason: {self.reason}, risk: {self.risk:.3f}, level: {self.level})"
        return (
            f"Decision: '{self.answer}' (confidence: {self.confidence:.2%}, "
            f"uncertainty: {self.uncertainty:.3f}, level: {self.level}, method: {self.method})"
        )

