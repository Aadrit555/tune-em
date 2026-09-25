"""Unit tests for batch inference and async decision execution."""

import pytest
from anydecision import DecisionEngine, Question


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

