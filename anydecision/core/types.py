"""Core types and data models for anydecision."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnswerType(str, Enum):
    """Supported question and answer paradigms."""
    BINARY = "binary"
    CHOICE = "choice"
    ORDINAL = "ordinal"
    NUMERIC_SCORE = "numeric_score"
    MULTI_CHOICE = "multi_choice"
    CUSTOM = "custom"


class DecisionLevel(str, Enum):
    """Decision confidence and robustness level."""
    L0 = "L0"  # Raw model probabilities
    L1 = "L1"  # Zero-label invariance & prompt aggregation
    L2 = "L2"  # Statistically calibrated layer


class ReadoutStrategy(str, Enum):
    """Method used to extract token or sequence probability."""
    NEXT_TOKEN = "next_token"
    MULTI_TOKEN_SEQUENCE = "multi_token_sequence"
    LENGTH_NORMALIZED = "length_normalized"
    CONDITIONAL_SPAN = "conditional_span"


class OptionDefinition(BaseModel):
    """Definition of an individual candidate answer option."""
    key: str = Field(description="Unique key or identifier for this option within the question.")
    label: str = Field(description="Display text or verbatim string the model is prompted on.")
    description: Optional[str] = Field(default=None, description="Optional semantic context or explanation.")
    ordinal_rank: Optional[int] = Field(default=None, description="Integer rank for ordinal questions.")
    numeric_value: Optional[float] = Field(default=None, description="Continuous or discrete score value.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata.")


class Diagnostics(BaseModel):
    """Observable statistics and diagnostics for a decision."""
    backend: str = Field(default="unknown", description="Inference backend name.")
    latency_ms: float = Field(default=0.0, description="End-to-end decision latency in milliseconds.")
    model: str = Field(default="unknown", description="Model identifier.")
    level: str = Field(default="L0", description="Decision level (L0, L1, L2).")
    raw_probabilities: Dict[str, float] = Field(default_factory=dict, description="Uncalibrated probabilities.")
    calibrated_probabilities: Optional[Dict[str, float]] = Field(default=None, description="Calibrated probabilities.")
    entropy: float = Field(default=0.0, description="Shannon entropy of the distribution in nats.")
    risk: float = Field(default=0.0, description="Estimated posterior risk or expected error.")
    abstention_threshold: Optional[float] = Field(default=None, description="Effective threshold for selective prediction.")
    number_of_permutations: int = Field(default=1, description="Number of option orderings evaluated.")
    template_agreement: float = Field(default=1.0, description="Agreement fraction across prompt templates [0, 1].")
    option_order_sensitivity: float = Field(default=0.0, description="Permutation TV distance / variance [0, 1].")
    cache_usage: Dict[str, Any] = Field(default_factory=dict, description="Prefix cache metrics.")
    number_of_backend_calls: int = Field(default=1, description="Total backend forward calls made.")
    compute_path: List[str] = Field(default_factory=lambda: ["L0"], description="Sequence of levels/readouts executed.")
    layers_executed: Optional[int] = Field(default=None, description="Number of model layers executed.")
    tokens_processed: int = Field(default=0, description="Total tokens processed across all forward passes.")
    ood_score: Optional[float] = Field(default=None, description="Out-of-distribution distance/entropy diagnostic.")
    ood_warning: Optional[str] = Field(default=None, description="OOD diagnostic caution message if triggered.")
    decision_emergence_layer: Optional[int] = Field(default=None, description="Earliest transformer layer where decision emerged.")
    layer_trajectory: Optional[Dict[str, Any]] = Field(default=None, description="Trajectory dynamics across layer depth.")


class TraceStep(BaseModel):
    """A single step in a decision trace."""
    step_name: str
    description: str
    data: Dict[str, Any] = Field(default_factory=dict)


class DecisionTrace(BaseModel):
    """Full research-grade step-by-step audit trace for a decision."""
    question_id: str
    steps: List[TraceStep] = Field(default_factory=list)

    def add_step(self, step_name: str, description: str, **data: Any) -> None:
        self.steps.append(TraceStep(step_name=step_name, description=description, data=data))

