"""Multi-token sequence probability scoring strategies."""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence
import numpy as np
from pydantic import BaseModel, Field


class SequenceScoringMethod(str, Enum):
    """Supported aggregation schemes for multi-token sequences."""
    SUM = "sum"
    EXACT_JOINT = "exact_joint"
    MEAN = "mean"
    LENGTH_NORMALIZED = "length_normalized"
    CONDITIONAL = "conditional"


class SequenceScoreResult(BaseModel):
    """Detailed score breakdown for a multi-token sequence."""
    joint_logprob: float = Field(
        description="Exact joint sequence log probability: sum_{t=1}^T log P(y_t | x, y_{<t})."
    )
    mean_logprob: float = Field(
        description="Mean per-token log probability: joint_logprob / T (ranking heuristic)."
    )
    length_normalized_score: float = Field(
        description="Wu et al. length-penalized score: joint_logprob / ((5 + T)/6)^alpha."
    )
    score: float = Field(
        description="Effective scalar ranking score according to configured SequenceScoringMethod."
    )
    scoring_method: SequenceScoringMethod = Field(
        description="Method used to compute the effective score."
    )
    token_count: int = Field(description="Number of tokens evaluated in sequence.")
    token_logprobs: List[float] = Field(default_factory=list, description="Per-token log probabilities.")


class SequenceScorer:
    """Scores multi-token candidate answers from per-token conditional log probabilities."""

    def __init__(
        self,
        method: SequenceScoringMethod = SequenceScoringMethod.LENGTH_NORMALIZED,
        length_penalty_alpha: float = 0.7,
    ):
        self.method = method
        self.length_penalty_alpha = length_penalty_alpha

    def score_sequence(self, token_logprobs: Sequence[float]) -> SequenceScoreResult:
        """Compute comprehensive candidate sequence score breakdown from token log-probabilities.

        Args:
            token_logprobs: Log P(token_t | prompt, token_<t) for each token in candidate.

        Returns:
            SequenceScoreResult with exact joint logprob, mean logprob, length-normalized score,
            and effective score according to the configured method.
        """
        if not token_logprobs:
            return SequenceScoreResult(
                joint_logprob=-1e9,
                mean_logprob=-1e9,
                length_normalized_score=-1e9,
                score=-1e9,
                scoring_method=self.method,
                token_count=0,
                token_logprobs=[],
            )

        n_tokens = len(token_logprobs)
        token_lps = [float(lp) for lp in token_logprobs]
        sum_logp = float(np.sum(token_lps))
        mean_logp = sum_logp / float(n_tokens)

        # Wu et al. (GNMT) penalty: ((5 + L) / 6) ** alpha
        penalty = ((5.0 + n_tokens) / 6.0) ** self.length_penalty_alpha
        length_norm = sum_logp / penalty

        if self.method in (SequenceScoringMethod.SUM, SequenceScoringMethod.EXACT_JOINT, SequenceScoringMethod.CONDITIONAL):
            effective_score = sum_logp
        elif self.method == SequenceScoringMethod.MEAN:
            effective_score = mean_logp
        elif self.method == SequenceScoringMethod.LENGTH_NORMALIZED:
            effective_score = length_norm
        else:
            effective_score = length_norm

        return SequenceScoreResult(
            joint_logprob=sum_logp,
            mean_logprob=mean_logp,
            length_normalized_score=length_norm,
            score=effective_score,
            scoring_method=self.method,
            token_count=n_tokens,
            token_logprobs=token_lps,
        )

    def score_tokens(self, token_logprobs: Sequence[float]) -> float:
        """Compute effective scalar score (log-domain) from token log-probabilities."""
        return self.score_sequence(token_logprobs).score

    def score_batch(self, candidate_token_logprobs: List[Sequence[float]]) -> List[float]:
        """Score multiple candidate token sequences."""
        return [self.score_tokens(logp_seq) for logp_seq in candidate_token_logprobs]

    def score_batch_detailed(
        self, candidate_token_logprobs: List[Sequence[float]]
    ) -> List[SequenceScoreResult]:
        """Score multiple candidate token sequences returning detailed breakdowns."""
        return [self.score_sequence(logp_seq) for logp_seq in candidate_token_logprobs]
