"""Unit tests for token and multi-token sequence scoring and tokenizer-aware candidate alignment."""

from __future__ import annotations

import numpy as np
import pytest

from anydecision.scoring.candidate_tokenizer import (
    CandidateTokenInfo,
    get_shared_prefix_token_ids,
    requires_sequence_scoring_for_candidates,
    tokenize_candidate,
    tokenize_candidate_set,
)
from anydecision.scoring.normalization import compute_entropy, normalize_log_probabilities, softmax
from anydecision.scoring.sequence import (
    SequenceScoreResult,
    SequenceScorer,
    SequenceScoringMethod,
)
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


def test_sequence_scorer_detailed_result():
    logp = [-0.5, -1.5]
    scorer = SequenceScorer(SequenceScoringMethod.LENGTH_NORMALIZED)
    res = scorer.score_sequence(logp)

    assert isinstance(res, SequenceScoreResult)
    assert np.isclose(res.joint_logprob, -2.0)
    assert np.isclose(res.mean_logprob, -1.0)
    assert res.token_count == 2
    assert res.length_normalized_score > res.joint_logprob

    # Empty sequence handling
    empty_res = scorer.score_sequence([])
    assert empty_res.token_count == 0
    assert empty_res.joint_logprob == -1e9


def test_extract_candidate_logprobs():
    logits = np.array([0.1, 10.0, 2.0, 5.0])
    mapping = {"yes": 1, "no": 2}
    lps = extract_candidate_logprobs_from_logits(logits, mapping)
    assert "yes" in lps and "no" in lps
    assert lps["yes"] > lps["no"]


def test_candidate_tokenizer_synthetic_fallback():
    prompt = "Answer the question:"
    info_multi = tokenize_candidate(None, prompt, "urgent technical support", "urg")
    assert info_multi.is_multi_token is True
    assert info_multi.is_empty is False
    assert info_multi.is_whitespace is False

    info_empty = tokenize_candidate(None, prompt, "", "emp")
    assert info_empty.is_empty is True
    assert info_empty.candidate_token_ids == []

    info_punct = tokenize_candidate(None, prompt, "...", "dot")
    assert info_punct.is_punctuation is True

    info_uni = tokenize_candidate(None, prompt, "pi = \u03c0", "pi")
    assert info_uni.is_unicode is True


def test_candidate_tokenizer_gpt2_comprehensive():
    """Verify tokenizer-aware alignment on known real causal BPE tokenizer (GPT-2)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("transformers or gpt2 tokenizer not available")

    prompts = [
        "Question: Identify language.\nAnswer:",
        "Question: Identify language.\nAnswer: ",  # Trailing whitespace boundary
    ]

    for prompt in prompts:
        # 1. Single-token candidates
        info_yes = tokenize_candidate(tok, prompt, "yes", "yes")
        info_no = tokenize_candidate(tok, prompt, "no", "no")
        assert not info_yes.is_multi_token
        assert not info_no.is_multi_token
        assert len(info_yes.candidate_token_ids) == 1
        assert len(info_no.candidate_token_ids) == 1

        info_true = tokenize_candidate(tok, prompt, "true", "true")
        info_false = tokenize_candidate(tok, prompt, "false", "false")
        assert not info_true.is_multi_token
        assert not info_false.is_multi_token

        # 2. Multi-token without spaces (C++, SQL-Injection, New_York)
        info_cpp = tokenize_candidate(tok, prompt, "C++", "cpp")
        assert info_cpp.is_multi_token is True
        assert len(info_cpp.candidate_token_ids) >= 2

        info_sql = tokenize_candidate(tok, prompt, "SQL-Injection", "sql")
        assert info_sql.is_multi_token is True
        assert len(info_sql.candidate_token_ids) >= 2

        info_ny = tokenize_candidate(tok, prompt, "New_York", "ny")
        assert info_ny.is_multi_token is True

        # 3. Multi-token with space & leading space
        info_urgent = tokenize_candidate(tok, prompt, " urgent technical support", "urg")
        assert info_urgent.is_multi_token is True
        assert len(info_urgent.candidate_token_ids) == 3

        # 4. Punctuation and Unicode
        info_dot = tokenize_candidate(tok, prompt, ".", "dot")
        assert info_dot.is_punctuation is True

        info_uni = tokenize_candidate(tok, prompt, "\u00e4\u00f6\u00fc", "umlaut")
        assert info_uni.is_unicode is True

        # 5. Empty string
        info_emp = tokenize_candidate(tok, prompt, "", "empty")
        assert info_emp.is_empty is True
        assert len(info_emp.candidate_token_ids) == 0


def test_shared_prefix_tokens_and_sequence_detection():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("transformers or gpt2 tokenizer not available")

    prompt = "Question: Select city.\nAnswer:"
    candidates = {
        "ny": "New York",
        "nj": "New Jersey",
    }
    c_set = tokenize_candidate_set(tok, prompt, candidates)
    assert len(c_set) == 2
    assert c_set["ny"].is_multi_token is True
    assert c_set["nj"].is_multi_token is True

    # Both start with ' New' token
    shared = get_shared_prefix_token_ids(list(c_set.values()))
    assert len(shared) >= 1

    # Check requires_sequence_scoring_for_candidates
    assert requires_sequence_scoring_for_candidates(tok, prompt, candidates) is True
    assert requires_sequence_scoring_for_candidates(
        tok, prompt, {"y": "yes", "n": "no"}
    ) is False
