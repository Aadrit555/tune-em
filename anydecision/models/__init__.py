"""Native model architectures and non-autoregressive decision heads."""

from anydecision.models.decision_head import (
    FastOptionScorer,
    NonAutoregressiveDecisionHead,
)

__all__ = [
    "FastOptionScorer",
    "NonAutoregressiveDecisionHead",
]
