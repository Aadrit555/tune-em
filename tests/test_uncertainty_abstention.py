"""Unit tests for uncertainty quantification, selective abstention, and OOD diagnostics."""

import pytest
from anydecision.core.policies import AbstentionPolicy
from anydecision.uncertainty.abstention import AbstentionController, find_optimal_rejection_threshold
from anydecision.uncertainty.entropy import compute_mutual_information, normalized_entropy
from anydecision.uncertainty.ood import OODDetector


def test_normalized_entropy():
    # Uniform 2 choices -> 1.0
    assert pytest.approx(normalized_entropy([0.5, 0.5]), abs=1e-4) == 1.0
    # Deterministic -> 0.0
    assert pytest.approx(normalized_entropy([1.0, 0.0]), abs=1e-4) == 0.0


def test_mutual_information_epistemic_disagreement():
    # 2 runs with strong disagreement
    dist1 = {"a": 0.99, "b": 0.01}
    dist2 = {"a": 0.01, "b": 0.99}
    mi = compute_mutual_information([dist1, dist2], ["a", "b"])
    assert mi > 0.5


def test_abstention_policy_min_confidence():
    policy = AbstentionPolicy(min_confidence=0.80)
    controller = AbstentionController(policy)

    # High confidence -> no abstain
    abstain, reason = controller.evaluate(confidence=0.85, probabilities={"yes": 0.85, "no": 0.15}, calibrated_risk=0.15)
    assert abstain is False

    # Low confidence -> abstain
    abstain, reason = controller.evaluate(confidence=0.75, probabilities={"yes": 0.75, "no": 0.25}, calibrated_risk=0.25)
    assert abstain is True
    assert "below_min" in str(reason)


def test_abstention_policy_target_error():
    policy = AbstentionPolicy(target_error=0.05)
    controller = AbstentionController(policy)

    # Risk 0.02 <= 0.05 -> no abstain
    abstain, _ = controller.evaluate(confidence=0.98, probabilities={"yes": 0.98, "no": 0.02}, calibrated_risk=0.02)
    assert abstain is False

    # Risk 0.10 > 0.05 -> abstain
    abstain, reason = controller.evaluate(confidence=0.90, probabilities={"yes": 0.90, "no": 0.10}, calibrated_risk=0.10)
    assert abstain is True


def test_ood_diagnostics():
    detector = OODDetector(entropy_threshold=0.85)

    # Flat uniform distribution -> triggers high entropy warning
    diag = detector.diagnose(probabilities={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    assert diag.is_shift_suspected is True
    assert diag.warning is not None
