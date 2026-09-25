"""Scoring and readout strategies for single and multi-token candidate answers."""

from anydecision.scoring.normalization import compute_entropy, log_softmax, normalize_log_probabilities, softmax
from anydecision.scoring.sequence import SequenceScorer, SequenceScoringMethod
from anydecision.scoring.token import extract_candidate_logprobs_from_logits

__all__ = [
    "compute_entropy",
    "extract_candidate_logprobs_from_logits",
    "log_softmax",
    "normalize_log_probabilities",
    "SequenceScorer",
    "SequenceScoringMethod",
    "softmax",
]
