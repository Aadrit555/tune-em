"""Unit and integration tests for Selective Conformal Risk Control and Hierarchical Calibration."""

from __future__ import annotations

import numpy as np
import pytest

from anydecision.calibration.hierarchical import (
    HierarchicalCalibrator,
    HierarchicalDiagnostics,
)
from anydecision.calibration.selective_conformal import (
    SelectiveConformalPredictor,
    SelectiveConformalResult,
)
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question


def test_selective_conformal_predictor():
    """Test conformal risk control fitting, certified risk bounds, and selective prediction."""
    rng = np.random.RandomState(42)
    n = 60
    # Synthetic calibration set
    labels = rng.choice([0, 1], size=n)
    # Inject slight label noise so low thresholds have non-zero empirical risk
    for i in range(8):
        labels[i] = 1 - labels[i]

    probs = np.zeros((n, 2))
    for i in range(n):
        if labels[i] == 1:
            p1 = rng.uniform(0.65, 0.98)
        else:
            p1 = rng.uniform(0.02, 0.35)
        probs[i] = [1.0 - p1, p1]

    keys = ["no", "yes"]
    scp = SelectiveConformalPredictor(risk_limit=0.05, min_coverage=0.70)
    scp.fit(probs, labels, keys)

    assert scp.fitted
    assert 0.55 <= scp.selection_threshold <= 0.99
    assert scp.certified_risk_bound <= 0.25

    # High confidence test
    confident_res = scp.predict({"no": 0.02, "yes": 0.98})
    assert isinstance(confident_res, SelectiveConformalResult)
    assert confident_res.selected is True
    assert "yes" in confident_res.prediction_set
    assert confident_res.guarantee_type == "conformal_exact"

    # Ambiguous test
    ambiguous_res = scp.predict({"no": 0.51, "yes": 0.49})
    assert ambiguous_res.selected is False


def test_engine_selective_conformal_risk_control():
    """Test engine.decide with risk_limit and coverage_target returns certified guarantees."""
    engine = DecisionEngine(model="mock")
    q = Question.binary("Is cloud cluster unhealthy?")

    decision = engine.decide(q, risk_limit=0.05, coverage_target=0.75)

    assert decision.risk_guarantee is not None
    assert decision.guarantee_type in ["conformal_exact", "empirical_approximate"]
    assert decision.selected in [True, False]
    assert decision.prediction_set is not None
    assert len(decision.prediction_set) >= 1


def test_hierarchical_calibrator_empirical_bayes_shrinkage():
    """Test 3-tier hierarchical calibration and Empirical Bayes shrinkage toward prior."""
    rng = np.random.RandomState(42)
    n_global = 80
    y_global = rng.choice([0, 1], size=n_global)
    p_global = np.stack([1.0 - y_global * 0.7, y_global * 0.7], axis=1)

    hier_cal = HierarchicalCalibrator(shrinkage_prior_n0=12.0)
    hier_cal.fit_global(p_global, y_global, ["no", "yes"])
    assert hier_cal.fitted
    global_t = hier_cal.global_calibrator.temperature

    # Fit domain group (e.g. "finance")
    y_fin = rng.choice([0, 1], size=40)
    p_fin = np.stack([1.0 - y_fin * 0.8, y_fin * 0.8], axis=1)
    hier_cal.fit_group("finance", p_fin, y_fin, ["no", "yes"])
    group_t = hier_cal.group_calibrators["finance"].temperature

    # Register sparse local question observations (5 samples)
    hier_cal.register_local_observations("q_rare_fraud", temperature=0.60, sample_count=5)

    # 1. Evaluate fallback to global for unknown group and question
    t_global, diag_global = hier_cal.resolve_effective_temperature(group_id=None, question_id=None)
    assert diag_global.resolution_level == "global_fallback"
    assert t_global == global_t

    # 2. Evaluate fallback to group for unknown question in "finance"
    t_group, diag_group = hier_cal.resolve_effective_temperature(group_id="finance", question_id="q_unknown")
    assert diag_group.resolution_level == "group_shrunk"
    assert t_group == group_t

    # 3. Evaluate shrinkage for sparse local question in "finance"
    t_local, diag_local = hier_cal.resolve_effective_temperature(group_id="finance", question_id="q_rare_fraud")
    assert diag_local.resolution_level == "local_shrunk"
    expected_lambda = 5.0 / (5.0 + 12.0)
    assert pytest.approx(diag_local.shrinkage_weight_lambda, 1e-4) == expected_lambda
    # Verify effective temperature is properly shrunk between local (0.60) and group
    assert min(0.60, group_t) <= t_local <= max(0.60, group_t)

    # Test calibrate_dict
    calibrated = hier_cal.calibrate_dict({"no": 0.20, "yes": 0.80}, group_id="finance", question_id="q_rare_fraud")
    assert pytest.approx(sum(calibrated.values()), 1e-4) == 1.0


def test_engine_hierarchical_calibration_integration():
    """Test engine decision flow using HierarchicalCalibrator and group_id."""
    hier_cal = HierarchicalCalibrator()
    hier_cal.global_calibrator.temperature = 1.35
    hier_cal.global_calibrator.fitted = True
    hier_cal.fitted = True

    engine = DecisionEngine(model="mock", calibrator=hier_cal)
    q = Question.choice("Choose optimal action", ["hold", "buy", "sell"])

    dec = engine.decide(q, level="L2", group_id="equities_trading")
    assert dec.level == "L2"
    assert dec.calibrated is True
    assert dec.method == "hierarchical"
    assert pytest.approx(sum(dec.probabilities.values()), 1e-4) == 1.0
