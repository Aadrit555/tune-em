"""Unit tests for token and multi-token sequence scoring."""

import numpy as np
import pytest
from anydecision.scoring.normalization import compute_entropy, normalize_log_probabilities, softmax
from anydecision.scoring.sequence import SequenceScorer, SequenceScoringMethod
from anydecision.scoring.token import extract_candidate_logprobs_from_logits


def test_softmax_sums_to_one():
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]


def test_normalize_log_probabilities():
    logprobs = {"a": -0.5, "b": -1.2, "c": -2.0}
    norm = normalize_log_probabilities(logprobs)
    assert np.isclose(sum(norm.values()), 1.0)
    assert norm["a"] > norm["b"] > norm["c"]


def test_entropy_calculation():
    # Uniform 4 options -> log(4) approx 1.3863
    probs = [0.25, 0.25, 0.25, 0.25]
    ent = compute_entropy(probs)
    assert np.isclose(ent, np.log(4.0), atol=1e-3)

    # Deterministic -> entropy = 0.0
    det_probs = [1.0, 0.0, 0.0]
    assert np.isclose(compute_entropy(det_probs), 0.0, atol=1e-5)


def test_sequence_scorer_methods():
    logp = [-0.2, -0.4, -0.6]  # 3 tokens
    sum_scorer = SequenceScorer(SequenceScoringMethod.SUM)
    mean_scorer = SequenceScorer(SequenceScoringMethod.MEAN)
    norm_scorer = SequenceScorer(SequenceScoringMethod.LENGTH_NORMALIZED)

    s_sum = sum_scorer.score_tokens(logp)
    s_mean = mean_scorer.score_tokens(logp)
    s_norm = norm_scorer.score_tokens(logp)

    assert np.isclose(s_sum, -1.2)
    assert np.isclose(s_mean, -0.4)
    assert s_sum < s_norm < s_mean


def test_extract_candidate_logprobs():
    logits = np.array([0.1, 10.0, 2.0, 5.0])
    mapping = {"yes": 1, "no": 2}
    lps = extract_candidate_logprobs_from_logits(logits, mapping)
    assert "yes" in lps and "no" in lps
    assert lps["yes"] > lps["no"]

