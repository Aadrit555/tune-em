"""Unit and integration tests for Active Calibration and Distribution Shift / Drift Monitor."""

from __future__ import annotations

import pytest

from anydecision.calibration.active import (
    ActiveCalibrationBenchmark,
    ActiveCalibrationReport,
    ActiveCalibrator,
    ActiveSelectionCriterion,
    CandidateSampleScore,
)
from anydecision.calibration.drift import (
    CalibrationDriftMonitor,
    DistributionShiftEvaluator,
    DistributionShiftReport,
    DistributionShiftType,
    DriftAlert,
)
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.evaluation.datasets import (
    create_customer_escalation_benchmark,
    create_topic_categorization_benchmark,
)


def test_active_calibrator_score_and_suggest():
    """Test candidate scoring and high-information sample suggestion."""
    engine = DecisionEngine(model="mock")
    questions = [
        Question.binary("Is server memory exhausted?"),
        Question.binary("Is latency spiking above SLA threshold?"),
        Question.choice("Category of customer ticket", ["billing", "tech", "sales"]),
        Question.choice("Incident severity", ["low", "med", "high", "critical"]),
    ]

    calibrator = ActiveCalibrator(criterion=ActiveSelectionCriterion.HYBRID)
    score = calibrator.score_question(questions[0], engine)

    assert isinstance(score, CandidateSampleScore)
    assert score.information_score >= 0.0
    assert score.entropy >= 0.0
    assert score.margin >= 0.0

    # Test engine.suggest_calibration_examples API
    suggested = engine.suggest_calibration_examples(questions, budget=2, criterion="hybrid")
    assert len(suggested) == 2
    assert all(isinstance(q, Question) for q in suggested)


def test_active_calibration_benchmark():
    """Test active vs random calibration benchmark simulation."""
    samples = create_customer_escalation_benchmark()[:10]

    report = ActiveCalibrationBenchmark.run_benchmark(
        samples=samples,
        engine_factory=lambda: DecisionEngine(model="mock"),
        budget=4,
        dataset_name="active_escalation_bench",
    )

    assert isinstance(report, ActiveCalibrationReport)
    assert report.pool_size == 10
    assert report.budget == 4
    assert len(report.selected_sample_ids) == 4

    summary = report.summary()
    assert "Active Calibration Benchmark" in summary
    assert "Random Sampling" in summary
    assert "Active Calibration" in summary


def test_distribution_shift_evaluator():
    """Test distribution shift evaluation between source and target datasets."""
    engine = DecisionEngine(model="mock")
    src = create_customer_escalation_benchmark()[:6]
    tgt = create_customer_escalation_benchmark()[6:12]

    report = DistributionShiftEvaluator.evaluate_shift(
        source_samples=src,
        target_samples=tgt,
        engine=engine,
        shift_type=DistributionShiftType.DOMAIN_SHIFT,
        source_name="support_chat",
        target_name="email_tickets",
    )

    assert isinstance(report, DistributionShiftReport)
    assert report.source_domain == "support_chat"
    assert report.target_domain == "email_tickets"
    assert 0.0 <= report.source_accuracy <= 1.0
    assert 0.0 <= report.target_accuracy <= 1.0

    summary = report.summary()
    assert "Distribution Shift Report" in summary
    assert "Expected Calib Error" in summary
    assert "Production Recommendation" in summary


def test_calibration_drift_monitor_sequential():
    """Test runtime sequential drift detection and threshold adaptation."""
    engine = DecisionEngine(model="mock")
    q = Question.binary("Is credit card authorization valid?")

    # Build certified baseline reference decisions
    ref_decisions = [engine.decide(q) for _ in range(30)]

    monitor = CalibrationDriftMonitor(
        window_size=30,
        warning_threshold=0.10,
        critical_threshold=0.20,
        min_samples_to_eval=10,
    )
    monitor.set_reference_distribution(ref_decisions)

    # Stream in identical decisions (no drift)
    for _ in range(15):
        dec = engine.decide(q)
        alert = monitor.record_decision(dec)

    base_threshold = 0.85
    adapted = monitor.adapt_abstention_threshold(base_threshold)
    assert adapted == base_threshold  # No drift yet

    # Now simulate severe distribution drift by injecting synthetic perturbed decisions with low confidence
    drift_alert_triggered = False
    for i in range(25):
        d_drift = engine.decide(q).model_copy()
        # Degrade confidence severely to simulate covariate shift
        d_drift.confidence = 0.50 + 0.01 * (i % 3)
        alert = monitor.record_decision(d_drift)
        if alert is not None:
            drift_alert_triggered = True

    assert drift_alert_triggered
    # Verify threshold was dynamically increased to protect against hallucinations under drift
    adapted_under_drift = monitor.adapt_abstention_threshold(base_threshold)
    assert adapted_under_drift > base_threshold
