"""Uncertainty quantification, selective prediction, and shift-suspicion heuristics."""

from anydecision.uncertainty.abstention import (
    AbstentionController,
    find_optimal_rejection_threshold,
)
from anydecision.uncertainty.entropy import compute_mutual_information, normalized_entropy
from anydecision.uncertainty.ood import (
    DistributionShiftHeuristic,
    OODDetector,
    OODDiagnosticsResult,
)

__all__ = [
    "AbstentionController",
    "DistributionShiftHeuristic",
    "OODDetector",
    "OODDiagnosticsResult",
    "compute_mutual_information",
    "find_optimal_rejection_threshold",
    "normalized_entropy",
]

