"""Unit tests for backends and probability extraction."""

from __future__ import annotations

import numpy as np
import pytest
from anydecision.backends.mock import MockBackend
from anydecision.backends.registry import get_backend
from anydecision.scoring.sequence import SequenceScoreResult


def test_mock_backend_metadata():
    backend = MockBackend(model_name="mock-7b")
    meta = backend.get_metadata()
    assert meta.model_name == "mock-7b"
    assert meta.backend_name == "mock"
    assert meta.fingerprint() != ""


def test_mock_backend_next_token():
    backend = MockBackend()
    res = backend.next_token_logprobs("Is this good?", {"yes": "yes", "no": "no"})
    assert "yes" in res and "no" in res
    assert isinstance(res["yes"], float)


def test_mock_backend_sequence_logprobs():
    backend = MockBackend()
    res = backend.sequence_logprobs(
        "Choose category:",
        {"tech": "urgent technical support", "bill": "standard billing question"},
        scoring_method="length_normalized",
    )
    assert "tech" in res and "bill" in res


def test_mock_backend_sequence_logprobs_detailed():
    backend = MockBackend()
    res = backend.sequence_logprobs_detailed(
        "Choose category:",
        {"tech": "urgent technical support", "bill": "standard billing question"},
        scoring_method="length_normalized",
    )
    assert "conditional_logprobs" in res
    assert "joint_logprobs" in res
    assert "mean_logprobs" in res
    assert "length_normalized_scores" in res
    assert "candidate_scores" in res

    # Verify score types
    for key in ("tech", "bill"):
        score_res = res["candidate_scores"][key]
        assert isinstance(score_res, SequenceScoreResult)
        assert score_res.joint_logprob == res["joint_logprobs"][key]
        assert score_res.mean_logprob == res["mean_logprobs"][key]
        assert score_res.token_count > 0


def test_get_backend_factory():
    b_mock = get_backend("mock")
    assert isinstance(b_mock, MockBackend)


def test_mock_backend_requires_sequence_scoring():
    backend = MockBackend()
    # Simple single words
    assert not backend.requires_sequence_scoring("Is this true?", {"yes": "yes", "no": "no"})
    # Contains symbol or spaces
    assert backend.requires_sequence_scoring("Which language?", {"cpp": "C++", "py": "python"})
    assert backend.requires_sequence_scoring("Topic?", {"sec": "SQL-Injection", "db": "database"})


def test_mock_backend_next_token_logprobs_detailed():
    backend = MockBackend()
    details = backend.next_token_logprobs_detailed("Is this active?", {"yes": "yes", "no": "no"})
    assert "conditional_logprobs" in details
    assert "raw_vocab_logprobs" in details
    assert "yes" in details["conditional_logprobs"]
    assert "yes" in details["raw_vocab_logprobs"]
    # Check conditional logprobs sum to ~ 1.0 in probability space
    probs = [np.exp(lp) for lp in details["conditional_logprobs"].values()]
    assert pytest.approx(sum(probs), 1e-4) == 1.0


def test_vllm_import_guidance():
    """Verify vllm backend gives clear installation instructions when not installed."""
    try:
        from anydecision.backends.vllm import VLLMBackend
        with pytest.raises(ImportError, match="pip install 'anydecision\\[vllm\\]'"):
            VLLMBackend("dummy/model")
    except ImportError:
        pass
