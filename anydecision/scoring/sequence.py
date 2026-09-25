"""Multi-token sequence probability scoring strategies."""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence
import numpy as np


class SequenceScoringMethod(str, Enum):
    """Supported aggregation schemes for multi-token sequences."""
    SUM = "sum"
    MEAN = "mean"
    LENGTH_NORMALIZED = "length_normalized"
    CONDITIONAL = "conditional"


class SequenceScorer:
    """Scores multi-token candidate answers from per-token conditional log probabilities."""

    def __init__(
        self,
        method: SequenceScoringMethod = SequenceScoringMethod.LENGTH_NORMALIZED,
        length_penalty_alpha: float = 0.7,
    ):
        self.method = method
        self.length_penalty_alpha = length_penalty_alpha

    def score_tokens(self, token_logprobs: Sequence[float]) -> float:
        """Compute candidate sequence score from list of token log-probabilities.

        Args:
            token_logprobs: Log P(token_t | prompt, token_<t) for each token in candidate.

        Returns:
            Scalar score (log-domain).
        """
        if not token_logprobs:
            return -1e9

        n_tokens = len(token_logprobs)
        sum_logp = float(np.sum(token_logprobs))

        if self.method == SequenceScoringMethod.SUM or self.method == SequenceScoringMethod.CONDITIONAL:
            return sum_logp
        elif self.method == SequenceScoringMethod.MEAN:
            return sum_logp / float(n_tokens)
        elif self.method == SequenceScoringMethod.LENGTH_NORMALIZED:
            # Wu et al. (GNMT) penalty: ((5 + L) / 6) ** alpha
            # This balances short vs long options without penalizing multi-word answers unfairly.
            penalty = ((5.0 + n_tokens) / 6.0) ** self.length_penalty_alpha
            return sum_logp / penalty
        else:
            return sum_logp / float(n_tokens)

    def score_batch(self, candidate_token_logprobs: List[Sequence[float]]) -> List[float]:
        """Score multiple candidate token sequences."""
        return [self.score_tokens(logp_seq) for logp_seq in candidate_token_logprobs]

