"""Unit and integration tests for Selective Conformal Risk Control and Hierarchical Calibration."""

from __future__ import annotations

import numpy as np
import pytest

from anydecision.calibration.active import (
    ActiveCalibrationBenchmark,
)
from anydecision.calibration.hierarchical import (
    HierarchicalCalibrator,
)
from anydecision.calibration.selective_conformal import (
    SelectiveConformalPredictor,
    SelectiveConformalResult,
)
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.evaluation.benchmark import BenchmarkSample


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
    assert scp.certified_risk_bound <= 0.35

    # High confidence test
    confident_res = scp.predict({"no": 0.02, "yes": 0.98})
    assert isinstance(confident_res, SelectiveConformalResult)
    assert confident_res.selected is True
    assert "yes" in confident_res.prediction_set
    assert confident_res.guarantee_type in ["formal_conformal", "high_confidence_empirical", "heuristic"]

    # Ambiguous test
    ambiguous_res = scp.predict({"no": 0.51, "yes": 0.49})
    assert ambiguous_res.selected is False


def test_selective_conformal_small_sample_is_heuristic():
    """Verify small calibration sets (<20) are marked HEURISTIC and never FORMAL_CONFORMAL."""
    scp = SelectiveConformalPredictor(risk_limit=0.05, min_coverage=0.70)
    tiny_probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
    tiny_labels = np.array([0, 1, 0])
    scp.fit(tiny_probs, tiny_labels, ["no", "yes"])

    res = scp.predict({"no": 0.95, "yes": 0.05})
    assert res.guarantee_type == "heuristic"
    assert res.guarantee_type != "formal_conformal"


def test_selective_conformal_data_validation():
    """Verify SelectiveConformalPredictor fails loudly on invalid, empty, or NaN inputs."""
    scp = SelectiveConformalPredictor()

    # Empty inputs
    with pytest.raises(ValueError, match="cannot be empty"):
        scp.fit(np.array([]), np.array([]))

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        scp.fit(np.array([[0.5, 0.5]]), np.array([0, 1]))

    # NaN in probabilities
    with pytest.raises(ValueError, match="NaN or Inf"):
        scp.fit(np.array([[np.nan, 0.5]]), np.array([0]))

    # Label outside candidate range
    with pytest.raises(ValueError, match="invalid class indices"):
        scp.fit(np.array([[0.8, 0.2]]), np.array([5]))


def test_conformal_coverage_simulation():
    """Simulate exchangeable trials and verify conformal prediction set coverage >= 1 - alpha."""
    rng = np.random.RandomState(1337)
    n_calib = 500
    n_test = 1000
    alpha = 0.10

    # Synthetic generative model
    def generate_data(size):
        y = rng.choice([0, 1], size=size, p=[0.5, 0.5])
        # Model predicted probabilities with calibration noise
        noise = rng.normal(0, 0.1, size=size)
        p1 = np.clip(np.where(y == 1, 0.85 + noise, 0.15 + noise), 0.01, 0.99)
        p = np.stack([1.0 - p1, p1], axis=1)
        return p, y

    calib_p, calib_y = generate_data(n_calib)
    test_p, test_y = generate_data(n_test)

    scp = SelectiveConformalPredictor(risk_limit=alpha, min_coverage=0.60)
    scp.fit(calib_p, calib_y, ["class_0", "class_1"])

    covered = 0
    for i in range(n_test):
        res = scp.predict({"class_0": float(test_p[i, 0]), "class_1": float(test_p[i, 1])})
        true_label = f"class_{test_y[i]}"
        if true_label in res.prediction_set:
            covered += 1

    emp_coverage = covered / n_test
    # 1 - alpha = 0.90; test with 3-sigma tolerance: sqrt(0.9 * 0.1 / 1000) ~ 0.0095 -> at least 0.88
    assert emp_coverage >= 0.88, f"Empirical coverage {emp_coverage} below theoretical bound"


def test_active_calibration_label_validation():
    """Verify active calibration fails loudly when ground truth label is invalid instead of defaulting to 0."""
    q = Question.binary("Is server online?")
    invalid_sample = BenchmarkSample(question=q, ground_truth="invalid_label_xyz")

    with pytest.raises(ValueError, match="not among option keys"):
        ActiveCalibrationBenchmark.run_benchmark(
            samples=[invalid_sample] * 5,
            engine_factory=lambda: DecisionEngine(model="mock"),
            budget=2,
        )


def test_active_calibration_budget_curve():
    """Verify evaluate_budget_curve returns reports across multiple budget points."""
    samples = [
        BenchmarkSample(
            question=Question.choice(f"Item {i}", ["billing", "tech", "sales"]),
            ground_truth="tech" if i % 2 == 0 else "billing",
        )
        for i in range(30)
    ]

    reports = ActiveCalibrationBenchmark.evaluate_budget_curve(
        samples=samples,
        engine_factory=lambda: DecisionEngine(model="mock"),
        budgets=[5, 10, 15],
    )
    assert len(reports) == 3
    assert [r.budget for r in reports] == [5, 10, 15]
    for r in reports:
        assert r.test_size > 0
        assert r.random_nll >= 0.0 and r.active_brier >= 0.0
        assert 0.0 <= r.active_selective_risk <= 1.0
        # Honest wording: single-budget delta is not a cost saving.
        assert "NOT an annotation-cost saving" in r.summary()


def test_active_calibration_no_cost_claim_without_curve():
    """A single-budget ECE delta must never be presented as annotation-cost reduction."""
    samples = [
        BenchmarkSample(
            question=Question.choice(f"Item {i}", ["billing", "tech", "sales"]),
            ground_truth="tech" if i % 2 == 0 else "billing",
        )
        for i in range(20)
    ]
    report = ActiveCalibrationBenchmark.run_benchmark(
        samples=samples,
        engine_factory=lambda: DecisionEngine(model="mock"),
        budget=5,
    )
    assert "fewer labels required" not in report.summary()
    assert "budget curve" in report.summary()


def test_estimate_label_savings_requires_curve_evidence():
    """estimate_label_savings interpolates within evaluated budgets, never extrapolates."""
    from anydecision.calibration.active import ActiveCalibrationReport

    def _rep(budget: int, ece: float, which: str) -> ActiveCalibrationReport:
        base = dict(
            dataset_name="d", pool_size=100, budget=budget, test_size=50,
            random_accuracy=0.5, active_accuracy=0.5, random_nll=1.0, active_nll=1.0,
            random_brier=0.25, active_brier=0.25, random_selective_risk=0.5,
            active_selective_risk=0.5, relative_ece_change_pct=0.0,
            selected_sample_ids=[],
        )
        if which == "random":
            return ActiveCalibrationReport(random_ece=ece, active_ece=ece, **base)
        return ActiveCalibrationReport(random_ece=ece, active_ece=ece, **base)

    random_curve = [_rep(10, 0.20, "random"), _rep(20, 0.10, "random"), _rep(40, 0.05, "random")]
    active_curve = [_rep(10, 0.10, "active")]
    res = ActiveCalibrationBenchmark.estimate_label_savings(active_curve, random_curve)
    pt = res["points"][0]
    assert pt["demonstrated"] is True
    assert pt["matched_random_budget"] == 20.0
    assert pt["estimated_savings_pct"] == 50.0

    unreachable = [_rep(10, 0.001, "active")]
    res2 = ActiveCalibrationBenchmark.estimate_label_savings(unreachable, random_curve)
    assert res2["points"][0]["demonstrated"] is False
    assert res2["points"][0]["matched_random_budget"] is None


def test_engine_selective_conformal_risk_control():
    """Test engine.decide with risk_limit and coverage_target returns certified guarantees."""
    engine = DecisionEngine(model="mock")
    q = Question.binary("Is cloud cluster unhealthy?")

    decision = engine.decide(q, risk_limit=0.05, coverage_target=0.75)

    assert decision.risk_guarantee is not None
    assert decision.guarantee_type in ["formal_conformal", "high_confidence_empirical", "heuristic", "unavailable"]
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
