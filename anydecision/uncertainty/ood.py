"""Out-of-distribution (OOD) and distribution shift diagnostics.

IMPORTANT RESEARCH NOTICE:
These statistical signals serve as observable heuristics and cautionary diagnostics.
They are NOT a guarantee against catastrophic distribution shift or adversarial evasion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from anydecision.uncertainty.entropy import normalized_entropy


class OODDiagnosticsResult(BaseModel):
    """Observable OOD and distribution shift signals for a decision."""
    ood_score: float = Field(
        default=0.0,
        description="Composite heuristic score [0, 1] where higher indicates higher shift suspicion."
    )
    is_shift_suspected: bool = Field(
        default=False,
        description="Whether heuristic threshold was exceeded."
    )
    signals: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual observable metrics (entropy, instability, collapse, sensitivity)."
    )
    warning: Optional[str] = Field(
        default=None,
        description="Human-readable warning explaining why shift was flagged."
    )


class OODDetector:
    """Computes distribution shift diagnostics across prompt perturbations and entropy profiles."""

    def __init__(
        self,
        entropy_threshold: float = 0.85,
        instability_threshold: float = 0.35,
        shift_score_cutoff: float = 0.60,
    ) -> None:
        self.entropy_threshold = entropy_threshold
        self.instability_threshold = instability_threshold
        self.shift_score_cutoff = shift_score_cutoff

    def diagnose(
        self,
        probabilities: Dict[str, float],
        permutation_distributions: Optional[List[Dict[str, float]]] = None,
        template_distributions: Optional[List[Dict[str, float]]] = None,
        reference_centroid: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> OODDiagnosticsResult:
        """Compute observable shift metrics for a given inference request."""
        prob_vals = list(probabilities.values())
        k = len(prob_vals)

        # 1. Entropy signal: high normalized entropy means the model has little confidence distinction
        norm_ent = normalized_entropy(prob_vals)

        # 2. Probability collapse: check if max probability is barely above uniform 1/K
        uniform_p = 1.0 / max(1, k)
        max_p = max(prob_vals) if prob_vals else 0.0
        prob_collapse = max(0.0, 1.0 - (max_p - uniform_p) / (1.0 - uniform_p + 1e-8))

        # 3. Permutation / Order instability
        perm_instability = 0.0
        if permutation_distributions and len(permutation_distributions) > 1:
            # Measure standard deviation of top choice across permutations
            keys = list(probabilities.keys())
            rows = [[d.get(k, 0.0) for k in keys] for d in permutation_distributions]
            perm_instability = float(np.mean(np.std(rows, axis=0)))

        # 4. Template sensitivity
        template_sensitivity = 0.0
        if template_distributions and len(template_distributions) > 1:
            keys = list(probabilities.keys())
            rows = [[d.get(k, 0.0) for k in keys] for d in template_distributions]
            template_sensitivity = float(np.mean(np.std(rows, axis=0)))

        # 5. Feature / Embedding distance if provided
        embedding_distance = 0.0
        if embedding is not None and reference_centroid is not None:
            # Cosine distance
            norm_e = embedding / (np.linalg.norm(embedding) + 1e-12)
            norm_c = reference_centroid / (np.linalg.norm(reference_centroid) + 1e-12)
            embedding_distance = float(max(0.0, 1.0 - np.dot(norm_e, norm_c)))

        signals = {
            "normalized_entropy": float(norm_ent),
            "probability_collapse": float(prob_collapse),
            "order_instability": float(perm_instability),
            "template_sensitivity": float(template_sensitivity),
            "embedding_distance": float(embedding_distance),
        }

        # Composite score
        weights = [0.35, 0.25, 0.20, 0.20]
        active_vals = [norm_ent, prob_collapse, perm_instability, template_sensitivity]
        composite = float(np.average(active_vals, weights=weights))

        is_suspected = composite >= self.shift_score_cutoff or norm_ent >= self.entropy_threshold
        warning = None
        if is_suspected:
            warning = (
                f"Caution: High distribution-shift suspicion (score={composite:.2f}). "
                f"High entropy ({norm_ent:.2f}) or prompt instability ({perm_instability:.2f}) "
                f"suggests model lacks clear semantic grounding for this question structure."
            )

        return OODDiagnosticsResult(
            ood_score=composite,
            is_shift_suspected=is_suspected,
            signals=signals,
            warning=warning,
        )

