"""Multi-model ensemble agreement and inter-model divergence as uncertainty signals.

Evaluates an ensemble of open-weight LLMs (e.g. Qwen, Llama, Mistral) on identical
typed questions. Extracts individual model distributions and derives:
- Inter-model disagreement (Jensen-Shannon Divergence / Total Variation distance)
- Consensus probability mixture (Linear opinion pool)
- Unanimous decision consensus flag
- Model contribution matrix
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

from anydecision.core.decision import Decision
from anydecision.scoring.normalization import compute_entropy

if TYPE_CHECKING:
    from anydecision.core.engine import DecisionEngine
    from anydecision.core.question import Question


class MultiModelDecisionResult(BaseModel):
    """Ensemble decision bundle with inter-model divergence uncertainty quantification."""
    question_id: str
    consensus_answer: str
    consensus_probabilities: Dict[str, float]
    consensus_confidence: float
    model_disagreement: float = Field(
        description="Mean Jensen-Shannon Divergence [0, 1] across model probability distributions."
    )
    tv_disagreement: float = Field(
        description="Maximum Total Variation distance across any pair of models."
    )
    unanimous: bool = Field(
        description="True if 100% of participating models agree on the top choice."
    )
    model_decisions: Dict[str, str] = Field(
        description="Map of model_name -> top predicted answer."
    )
    model_contributions: Dict[str, Dict[str, float]] = Field(
        description="Full probability distributions per participating model."
    )
    inter_model_entropy: float = Field(
        description="Shannon entropy of the blended consensus distribution in nats."
    )

    def summary(self) -> str:
        lines = [
            f"=== Multi-Model Ensemble Consensus (Question: {self.question_id}) ===",
            f"Consensus Decision:      '{self.consensus_answer}' (confidence: {self.consensus_confidence*100:.2f}%)",
            f"Unanimous Agreement:     {self.unanimous}",
            f"Model Disagreement (JSD):{self.model_disagreement:.4f}",
            f"Max TV Distance:         {self.tv_disagreement:.4f}",
            f"Inter-Model Entropy:     {self.inter_model_entropy:.4f} nats",
            "",
            f"{'Model Name':<24} | {'Top Choice':<14} | {'Top Confidence':<14}",
            "-" * 58,
        ]
        for m_name, dist in sorted(self.model_contributions.items()):
            top_choice = self.model_decisions.get(m_name, "unknown")
            top_conf = dist.get(top_choice, 0.0)
            lines.append(f"{m_name:<24} | {top_choice:<14} | {top_conf*100:>12.2f}%")
        lines.append("==========================================================")
        return "\n".join(lines)


class MultiModelEnsemble:
    """Ensemble runtime evaluating disagreement across multiple diverse open-weight models."""

    def __init__(
        self,
        engines: Sequence[DecisionEngine],
        model_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if not engines:
            raise ValueError("MultiModelEnsemble requires at least one DecisionEngine instance.")
        self.engines = list(engines)
        self.model_weights = model_weights or {}

    def decide(
        self,
        question: Question,
        level: str = "L0",
        **kwargs: Any,
    ) -> MultiModelDecisionResult:
        """Run decision across all models in parallel and compute inter-model disagreement."""
        keys = question.option_keys()
        k_to_idx = {k: i for i, k in enumerate(keys)}
        num_models = len(self.engines)

        prob_distributions: Dict[str, Dict[str, float]] = {}
        model_predictions: Dict[str, str] = {}
        prob_matrix = np.zeros((num_models, len(keys)), dtype=np.float64)

        for m_idx, eng in enumerate(self.engines):
            m_name = eng.metadata.model_name
            # If multiple models have identical name, distinguish with index
            if m_name in prob_distributions:
                m_name = f"{m_name}_{m_idx}"

            dec = eng.decide(question, level=level, **kwargs)
            prob_distributions[m_name] = dec.probabilities
            model_predictions[m_name] = dec.answer or "abstained"

            for k, p in dec.probabilities.items():
                if k in k_to_idx:
                    prob_matrix[m_idx, k_to_idx[k]] = p

        # Normalize matrix rows
        row_sums = np.sum(prob_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        prob_matrix = prob_matrix / row_sums

        # Compute Consensus Mixture (Linear Opinion Pool with weights)
        raw_weights = np.array([
            self.model_weights.get(eng.metadata.model_name, 1.0)
            for eng in self.engines
        ], dtype=np.float64)
        weights = raw_weights / np.sum(raw_weights)

        consensus_vec = np.sum(prob_matrix * weights[:, None], axis=0)
        consensus_probs = {k: float(p) for k, p in zip(keys, consensus_vec)}
        sorted_consensus = sorted(consensus_probs.items(), key=lambda x: x[1], reverse=True)
        top_ans, top_conf = sorted_consensus[0]

        # Check Unanimity
        unique_answers = set(model_predictions.values())
        unanimous = len(unique_answers) == 1 and ("abstained" not in unique_answers)

        # Compute Jensen-Shannon Divergence (JSD)
        # JSD(P_1, ..., P_M) = H(sum w_m P_m) - sum w_m H(P_m)
        h_consensus = compute_entropy(list(consensus_vec))
        sum_h_models = sum(
            w * compute_entropy(list(prob_matrix[i]))
            for i, w in enumerate(weights)
        )
        jsd = max(0.0, float(h_consensus - sum_h_models))

        # Compute Max Pairwise Total Variation Distance
        max_tv = 0.0
        for i in range(num_models):
            for j in range(i + 1, num_models):
                tv = 0.5 * np.sum(np.abs(prob_matrix[i] - prob_matrix[j]))
                if tv > max_tv:
                    max_tv = float(tv)

        return MultiModelDecisionResult(
            question_id=question.id,
            consensus_answer=top_ans,
            consensus_probabilities=consensus_probs,
            consensus_confidence=top_conf,
            model_disagreement=jsd,
            tv_disagreement=max_tv,
            unanimous=unanimous,
            model_decisions=model_predictions,
            model_contributions=prob_distributions,
            inter_model_entropy=float(h_consensus),
        )


# Alias for explicit naming
MultiModelEnsembleDisagreement = MultiModelEnsemble
