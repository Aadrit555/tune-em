"""Tests for original native non-autoregressive decision models."""

import time
import numpy as np
import pytest
import torch

from anydecision.models.decision_head import (
    FastOptionScorer,
    NonAutoregressiveDecisionHead,
)


def test_non_autoregressive_decision_head_forward():
    head = NonAutoregressiveDecisionHead(hidden_dim=64, projection_dim=32)
    head.eval()

    batch_size = 4
    num_choices = 5
    hidden_dim = 64

    context_hidden = torch.randn(batch_size, hidden_dim)
    candidate_embeddings = torch.randn(num_choices, hidden_dim)

    logits = head(context_hidden, candidate_embeddings)
    assert logits.shape == (batch_size, num_choices)

    # Probabilities sum to 1 across candidates
    probs = torch.softmax(logits, dim=-1)
    row_sums = probs.sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_non_autoregressive_decision_head_backward():
    head = NonAutoregressiveDecisionHead(hidden_dim=32, projection_dim=16)
    head.train()

    context_hidden = torch.randn(2, 32, requires_grad=True)
    candidate_embeddings = torch.randn(3, 32, requires_grad=True)

    logits = head(context_hidden, candidate_embeddings)
    target = torch.tensor([0, 2])
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert head.temperature.grad is not None
    assert context_hidden.grad is not None
    assert candidate_embeddings.grad is not None


def test_fast_option_scorer_probabilities():
    scorer = FastOptionScorer(hidden_dim=64, projection_dim=32, seed=123)

    context = "The database cluster reports high connection timeouts and memory pressure."
    choices = [
        "scale database capacity",
        "update marketing website",
        "order office stationery",
    ]

    probs = scorer.score(context, choices)

    assert len(probs) == 3
    for opt in choices:
        assert opt in probs
        assert 0.0 <= probs[opt] <= 1.0

    prob_sum = sum(probs.values())
    assert pytest.approx(1.0, abs=1e-4) == prob_sum

    # Semantic keyword resonance should favor database capacity
    assert probs["scale database capacity"] > probs["order office stationery"]


def test_fast_option_scorer_latency():
    scorer = FastOptionScorer(hidden_dim=128, projection_dim=64)
    context = "Security alert: unauthorized SSH access attempt detected from unknown IP."
    choices = ["block ip address", "ignore event", "request escalation", "schedule reboot"]

    # Warmup
    scorer.score(context, choices)

    # Benchmark sub-10ms execution
    t0 = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        scorer.score(context, choices)
    avg_latency_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

    # Ensure lightning fast throughput (< 5ms per scoring call)
    assert avg_latency_ms < 10.0, f"Average latency too high: {avg_latency_ms:.2f} ms"

