"""Unit tests for batch inference, numerical equivalence, and async decision execution."""

from __future__ import annotations

import numpy as np
import pytest
from anydecision import DecisionEngine, Question
from anydecision.core.types import DecisionLevel


@pytest.mark.asyncio
async def test_async_decide():
    engine = DecisionEngine(model="mock")
    q = Question.binary("Is this transaction legit?")
    res = await engine.async_decide(q)
    assert res.answer in ("yes", "no")
    assert res.confidence > 0.0


@pytest.mark.asyncio
async def test_async_batch_decide():
    engine = DecisionEngine(model="mock")
    questions = [
        Question.binary(f"Item {i} is approved?")
        for i in range(5)
    ]
    results = await engine.async_batch_decide(questions)
    assert len(results) == 5
    for r in results:
        assert r.answer in ("yes", "no")


def test_synchronous_batch_decide():
    engine = DecisionEngine(model="mock")
    questions = [
        Question.choice(f"Topic {i}", ["billing", "tech", "sales"])
        for i in range(4)
    ]
    results = engine.batch_decide(questions)
    assert len(results) == 4
    for r in results:
        assert r.answer in ("billing", "tech", "sales")
        assert r.method == "batched_vectorized"
        assert r.backend_calls == 1


def test_batch_decide_numerical_parity_with_individual_decide():
    """Verify batch_decide produces exact identical probabilities to sequential decide()."""
    engine = DecisionEngine(model="mock", default_level="L0")

    # Questions with heterogeneous candidate counts and varying prompt lengths
    q1 = Question.binary("Short prompt: approve?")
    q2 = Question.choice(
        "Medium length prompt with background context: classify the customer inquiry into category.",
        ["billing", "technical", "sales", "general"],
    )
    q3 = Question.choice(
        "Evaluate the severity level of this security incident report.",
        ["low", "medium", "critical"],
    )

    questions = [q1, q2, q3]

    # Individual sequential decisions
    individual_decisions = [engine.decide(q, level=DecisionLevel.L0) for q in questions]

    # Batched decisions
    batch_decisions = engine.batch_decide(questions, level=DecisionLevel.L0)

    assert len(batch_decisions) == len(individual_decisions)

    for idx, (ind_dec, batch_dec) in enumerate(zip(individual_decisions, batch_decisions)):
        assert ind_dec.answer == batch_dec.answer
        assert np.isclose(ind_dec.confidence, batch_dec.confidence, atol=1e-5)
        assert np.isclose(ind_dec.uncertainty, batch_dec.uncertainty, atol=1e-5)
        assert set(ind_dec.probabilities.keys()) == set(batch_dec.probabilities.keys())

        for k in ind_dec.probabilities:
            assert np.isclose(
                ind_dec.probabilities[k],
                batch_dec.probabilities[k],
                atol=1e-5,
            ), f"Mismatch at item {idx}, option {k}: {ind_dec.probabilities[k]} vs {batch_dec.probabilities[k]}"


def test_batch_decide_ordering_invariance():
    """Verify batch_decide is permutation invariant with respect to question order in batch."""
    engine = DecisionEngine(model="mock", default_level="L0")

    q1 = Question.binary("Is server online?")
    q2 = Question.choice("Priority tier?", ["p1", "p2", "p3"])
    q3 = Question.choice("Action required?", ["reboot", "ignore"])

    order_a = [q1, q2, q3]
    order_b = [q3, q1, q2]

    res_a = engine.batch_decide(order_a, level=DecisionLevel.L0)
    res_b = engine.batch_decide(order_b, level=DecisionLevel.L0)

    # res_a[0] is q1, res_b[1] is q1
    assert res_a[0].answer == res_b[1].answer
    for k in res_a[0].probabilities:
        assert np.isclose(res_a[0].probabilities[k], res_b[1].probabilities[k], atol=1e-5)

    # res_a[2] is q3, res_b[0] is q3
    assert res_a[2].answer == res_b[0].answer
    for k in res_a[2].probabilities:
        assert np.isclose(res_a[2].probabilities[k], res_b[0].probabilities[k], atol=1e-5)


def test_batch_decide_multi_token_fallback():
    """Verify questions with multi-token options gracefully fall back to sequence evaluation."""
    engine = DecisionEngine(model="mock", default_level="L0")

    q_multi = Question.choice(
        "Select department",
        ["urgent technical support", "routine billing question"],
    )
    q_single = Question.binary("Approve action?")

    questions = [q_multi, q_single]
    results = engine.batch_decide(questions, level=DecisionLevel.L0)

    assert len(results) == 2
    assert results[0].answer in ("urgent technical support", "routine billing question")
    assert results[1].answer in ("yes", "no")
