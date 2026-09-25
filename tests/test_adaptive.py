"""Unit and integration tests for adaptive compute routing and decision economics."""

from __future__ import annotations

import pytest

from anydecision.adaptive.router import AdaptiveComputeConfig, AdaptiveComputeRouter
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.evaluation.datasets import create_customer_escalation_benchmark
from anydecision.evaluation.economics import (
    DecisionEconomicsEvaluator,
    DecisionEconomicsReport,
    EconomicsConfig,
)


def test_adaptive_router_early_exit_l0():
    """Test that highly confident L0 decisions exit early without escalating to L1."""
    engine = DecisionEngine(model="mock")
    q = Question.choice(
        "Is payment fraud suspected on this high-value transfer?",
        ["yes", "no"]
    )
    # Set moderate L0 threshold that mock backend easily exceeds
    config = AdaptiveComputeConfig(
        early_exit_l0_confidence=0.40,
        max_l0_entropy=1.0,
    )
    decision = engine.decide_adaptive(q, adaptive_config=config)

    assert decision.answer in ["yes", "no"]
    assert decision.compute_path == ["L0"]
    assert decision.backend_calls == 1
    assert decision.tokens_processed > 0
    assert decision.latency_ms >= 0.0
    assert decision.layers_executed > 0


def test_adaptive_router_escalates_to_l1():
    """Test that when L0 confidence is below threshold, router escalates to L1."""
    engine = DecisionEngine(model="mock")
    q = Question.choice(
        "Categorize ambiguous inquiry",
        ["billing", "technical", "sales", "general"]
    )
    # Require 0.9999 confidence to force escalation from L0 to L1
    config = AdaptiveComputeConfig(
        early_exit_l0_confidence=0.9999,
        early_exit_l1_confidence=0.40,
    )
    router = AdaptiveComputeRouter(config=config)
    decision = router.decide_adaptive(engine, q)

    assert "L0" in decision.compute_path
    assert "L1" in decision.compute_path
    assert decision.backend_calls > 1
    assert decision.tokens_processed > 0


def test_engine_decide_adaptive_flag():
    """Test engine.decide(..., adaptive=True) matches engine.decide_adaptive(...)."""
    engine = DecisionEngine(model="mock")
    q = Question.binary("Is user authenticated?")

    decision = engine.decide(q, adaptive=True)
    assert decision.compute_path is not None
    assert len(decision.compute_path) >= 1
    assert decision.confidence > 0.0
    assert not decision.abstained


def test_decision_economics_evaluator():
    """Test comprehensive decision economics evaluation on benchmark samples."""
    engine = DecisionEngine(model="mock")
    samples = create_customer_escalation_benchmark()[:6]

    econ_cfg = EconomicsConfig(
        cost_per_million_tokens=0.50,
        hardware_cost_per_hour=1.20,
    )
    evaluator = DecisionEconomicsEvaluator(
        engine=engine,
        economics_config=econ_cfg,
        adaptive_config=AdaptiveComputeConfig(early_exit_l0_confidence=0.50),
    )
    report = evaluator.evaluate(samples, compare_static_levels=True)

    assert isinstance(report, DecisionEconomicsReport)
    assert report.total_samples == 6
    assert 0.0 <= report.adaptive_accuracy <= 1.0
    assert 0.0 <= report.adaptive_abstention_rate <= 1.0
    assert report.safe_decisions_per_sec >= 0.0
    assert len(report.compute_path_distribution) >= 1
    assert len(report.levels_comparison) >= 2  # L0, L1, Adaptive

    # Test summary string generation
    summary = report.summary()
    assert "DECISION ECONOMICS BENCHMARK" in summary
    assert "Adaptive" in summary
    assert "$" in summary
