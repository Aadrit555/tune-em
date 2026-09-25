"""Layer-trajectory uncertainty and internal representation dynamics analysis.

Inspects hidden representations and candidate probabilities across transformer layers
(e.g., layers 4, 8, 12, 16, 20, 24, 28, 32). Extracts decision emergence layer,
prediction stability, confidence growth, representation convergence, and layer disagreement.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

from anydecision.scoring.normalization import compute_entropy

if TYPE_CHECKING:
    from anydecision.backends.base import BaseBackend
    from anydecision.core.question import Question


class LayerCheckpoint(BaseModel):
    """Snapshot of model beliefs and representation at a specific layer checkpoint."""
    layer_idx: int
    top_answer: str
    confidence: float
    probabilities: Dict[str, float]
    entropy: float
    cosine_similarity_to_next: Optional[float] = None
    hidden_norm: Optional[float] = None


class LayerTrajectoryResult(BaseModel):
    """Derived trajectory dynamics across language model layer depth."""
    question_id: str
    trajectory: List[LayerCheckpoint]
    decision_emergence_layer: int = Field(
        description="First layer where the final answer permanently emerged with required confidence."
    )
    prediction_stability: float = Field(
        description="Ratio of post-emergence layers that maintain the final decision (1.0 = rock solid)."
    )
    confidence_growth_rate: float = Field(
        description="Average slope of confidence increase per transformer layer (dConfidence / dLayer)."
    )
    representation_convergence: float = Field(
        description="Mean cosine similarity between consecutive layer hidden vectors."
    )
    layer_disagreement_entropy: float = Field(
        description="Shannon entropy over the distribution of answers chosen across all layers."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Layer-Trajectory Dynamics (Question: {self.question_id}) ===",
            f"{'Layer':<8} | {'Top Answer':<14} | {'Confidence':<12} | {'Entropy':<9} | {'Next CosSim':<12}",
            "-" * 65,
        ]
        for cp in self.trajectory:
            cos_str = f"{cp.cosine_similarity_to_next:.4f}" if cp.cosine_similarity_to_next is not None else "---"
            mark = " <-- Emergence" if cp.layer_idx == self.decision_emergence_layer else ""
            lines.append(
                f"L{cp.layer_idx:<6} | {cp.top_answer:<14} | {cp.confidence*100:>9.2f}% | "
                f"{cp.entropy:>8.4f} | {cos_str:>11}{mark}"
            )
        lines.extend([
            "-" * 65,
            f"Decision Emergence Layer:    Layer {self.decision_emergence_layer}",
            f"Prediction Stability:        {self.prediction_stability * 100:.1f}%",
            f"Confidence Growth Rate:      {self.confidence_growth_rate:+.4f} / layer",
            f"Representation Convergence:  {self.representation_convergence:.4f}",
            f"Layer Disagreement Entropy:  {self.layer_disagreement_entropy:.4f}",
            "=================================================================",
        ])
        return "\n".join(lines)


class LayerTrajectoryAnalyzer:
    """Computes internal decision trajectory across transformer depth."""

    def __init__(self, emergence_confidence_threshold: float = 0.65) -> None:
        self.emergence_threshold = emergence_confidence_threshold

    def analyze(
        self,
        backend: BaseBackend,
        prompt: str,
        candidate_strings: Dict[str, str],
        question_id: str = "q_trajectory",
        layers: Optional[Sequence[int]] = None,
    ) -> LayerTrajectoryResult:
        """Analyze trajectory of probability and hidden state emergence across layers."""
        meta = backend.get_metadata()
        total_layers = meta.num_layers or 32

        if layers is None:
            # Checkpoints across depth
            checkpoints = [l for l in [4, 8, 12, 16, 20, 24, 28, total_layers] if l <= total_layers]
            if not checkpoints:
                checkpoints = [total_layers]
        else:
            checkpoints = sorted(list(layers))

        layer_logprobs = backend.get_layer_logprobs(prompt, candidate_strings, layers=checkpoints)
        hidden_states = backend.get_layer_hidden_states(prompt, layers=checkpoints)

        checkpoints_data: List[LayerCheckpoint] = []
        keys = list(candidate_strings.keys())

        for idx, l in enumerate(checkpoints):
            lps = layer_logprobs.get(l, {})
            if not lps:
                continue

            # Convert log-probs to probabilities
            lp_arr = np.array([lps.get(k, -100.0) for k in keys], dtype=np.float64)
            # Subtract max for numeric stability
            probs_arr = np.exp(lp_arr - np.max(lp_arr))
            probs_arr /= np.sum(probs_arr)
            prob_dict = {k: float(p) for k, p in zip(keys, probs_arr)}

            sorted_c = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            top_ans, top_conf = sorted_c[0]
            entropy_val = compute_entropy(list(prob_dict.values()))

            # Cosine similarity to next layer representation if available
            cos_sim = None
            if l in hidden_states and (idx + 1 < len(checkpoints)):
                next_l = checkpoints[idx + 1]
                if next_l in hidden_states:
                    v1 = hidden_states[l]
                    v2 = hidden_states[next_l]
                    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
                    cos_sim = float(np.dot(v1, v2) / denom)

            h_norm = float(np.linalg.norm(hidden_states[l])) if l in hidden_states else None

            checkpoints_data.append(
                LayerCheckpoint(
                    layer_idx=l,
                    top_answer=top_ans,
                    confidence=top_conf,
                    probabilities=prob_dict,
                    entropy=entropy_val,
                    cosine_similarity_to_next=cos_sim,
                    hidden_norm=h_norm,
                )
            )

        if not checkpoints_data:
            # Fallback if empty
            checkpoints_data = [
                LayerCheckpoint(
                    layer_idx=total_layers,
                    top_answer=keys[0] if keys else "unknown",
                    confidence=1.0,
                    probabilities={k: 1.0 / len(keys) for k in keys},
                    entropy=0.0,
                )
            ]

        final_cp = checkpoints_data[-1]
        final_answer = final_cp.top_answer

        # Find decision emergence layer:
        # First layer where answer == final_answer and maintains answer until the end,
        # with confidence >= threshold (or top confidence if none cross threshold)
        emergence_layer = checkpoints_data[0].layer_idx
        for i, cp in enumerate(checkpoints_data):
            # Check if all subsequent layers also choose final_answer
            all_subsequent_agree = all(c.top_answer == final_answer for c in checkpoints_data[i:])
            if all_subsequent_agree and cp.confidence >= self.emergence_threshold:
                emergence_layer = cp.layer_idx
                break
        else:
            # If no layer passed threshold, find first layer matching final answer
            for cp in checkpoints_data:
                if cp.top_answer == final_answer:
                    emergence_layer = cp.layer_idx
                    break

        # Compute stability: fraction of post-emergence layers agreeing with final answer
        post_emergence = [cp for cp in checkpoints_data if cp.layer_idx >= emergence_layer]
        stability = (
            sum(1 for cp in post_emergence if cp.top_answer == final_answer) / max(1, len(post_emergence))
        )

        # Compute confidence growth rate via linear regression slope
        layer_indices = np.array([cp.layer_idx for cp in checkpoints_data], dtype=np.float64)
        confidences = np.array([cp.confidence for cp in checkpoints_data], dtype=np.float64)
        if len(layer_indices) > 1 and np.var(layer_indices) > 0:
            cov = np.cov(layer_indices, confidences)[0, 1]
            var = np.var(layer_indices)
            growth_rate = float(cov / var)
        else:
            growth_rate = 0.0

        # Representation convergence: average cosine similarity across consecutive layers
        cos_sims = [cp.cosine_similarity_to_next for cp in checkpoints_data if cp.cosine_similarity_to_next is not None]
        mean_convergence = float(np.mean(cos_sims)) if cos_sims else 1.0

        # Layer disagreement entropy across all layers' top predictions
        counts: Dict[str, int] = {}
        for cp in checkpoints_data:
            counts[cp.top_answer] = counts.get(cp.top_answer, 0) + 1
        total_cps = len(checkpoints_data)
        freqs = [c / total_cps for c in counts.values()]
        disagreement_entropy = float(-sum(p * math.log(max(1e-12, p)) for p in freqs))

        return LayerTrajectoryResult(
            question_id=question_id,
            trajectory=checkpoints_data,
            decision_emergence_layer=emergence_layer,
            prediction_stability=stability,
            confidence_growth_rate=growth_rate,
            representation_convergence=mean_convergence,
            layer_disagreement_entropy=disagreement_entropy,
        )
