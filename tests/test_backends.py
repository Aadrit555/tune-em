"""Unit tests for backends and probability extraction."""

import pytest
from anydecision.backends.mock import MockBackend
from anydecision.backends.registry import get_backend


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


def test_get_backend_factory():
    b_mock = get_backend("mock")
    assert isinstance(b_mock, MockBackend)
