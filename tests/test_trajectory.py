"""Tests for layer-trajectory uncertainty and multi-layer representation fusion."""

from __future__ import annotations

import numpy as np
import pytest

from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.representations.fusion import (
    FusionComparisonExperiment,
    FusionExperimentReport,
    FusionStrategy,
    MultiLayerFusionHead,
)
from anydecision.representations.trajectory import (
    LayerTrajectoryAnalyzer,
    LayerTrajectoryResult,
)


def test_layer_trajectory_analysis():
    """Test layer trajectory analysis on transformer depth checkpoints."""
    engine = DecisionEngine(model="mock")
    q = Question.choice(
        "Is this medical query safe to answer directly?",
        ["safe", "unsafe"]
    )
    result = engine.analyze_layer_trajectory(q)

    assert isinstance(result, LayerTrajectoryResult)
    assert len(result.trajectory) >= 5
    assert result.decision_emergence_layer >= 4
    assert 0.0 <= result.prediction_stability <= 1.0
    assert result.representation_convergence > 0.0
    assert result.layer_disagreement_entropy >= 0.0

    # Test summary rendering
    summary_str = result.summary()
    assert "Layer-Trajectory Dynamics" in summary_str
    assert "Decision Emergence Layer" in summary_str
    assert "Prediction Stability" in summary_str


def test_engine_decide_with_trajectory_tracking():
    """Test engine.decide with track_layer_trajectory=True populates emergence metrics."""
    engine = DecisionEngine(model="mock")
    q = Question.binary("Is invoice verified?")
    decision = engine.decide(q, track_layer_trajectory=True)

    assert decision.decision_emergence_layer is not None
    assert decision.decision_emergence_layer in [4, 8, 12, 16, 20, 24, 28, 32]
    assert decision.layer_trajectory is not None
    assert "trajectory" in decision.layer_trajectory
    assert decision.diagnostics is not None
    assert decision.diagnostics.decision_emergence_layer == decision.decision_emergence_layer


def test_multilayer_fusion_heads():
    """Test training and predicting with concatenation, weighted average, and gating fusion."""
    rng = np.random.RandomState(42)
    n_samples = 60
    dim = 32
    layers = [12, 18, 24]

    # Create synthetic representations where layer 24 has high signal
    signal = rng.choice([0, 1], size=n_samples)
    X_by_layer = {}
    for l in layers:
        noise_level = 1.0 / np.sqrt(l)
        X_by_layer[l] = rng.randn(n_samples, dim) + signal[:, None] * (1.0 / noise_level)

    y = signal
    option_keys = ["class_0", "class_1"]

    for strat in [FusionStrategy.CONCATENATION, FusionStrategy.WEIGHTED_AVERAGE, FusionStrategy.LEARNED_GATING]:
        head = MultiLayerFusionHead(layers=layers, strategy=strat)
        head.fit(X_by_layer, y, option_keys)

        probs = head.predict_proba(X_by_layer)
        assert probs.shape == (n_samples, 2)
        assert np.allclose(np.sum(probs, axis=1), 1.0)

        # Single dict test
        single_X = {l: X_by_layer[l][:1] for l in layers}
        p_dict = head.predict_dict(single_X)
        assert set(p_dict.keys()) == set(option_keys)
        assert pytest.approx(sum(p_dict.values()), 1e-4) == 1.0


def test_fusion_comparison_experiment():
    """Test comparative benchmark of single-layer vs multi-layer fusion heads."""
    rng = np.random.RandomState(42)
    n_samples = 80
    dim = 24
    layers = [12, 18, 24]

    y = rng.choice([0, 1], size=n_samples)
    X_by_layer = {
        l: rng.randn(n_samples, dim) + y[:, None] * (float(l) / 10.0)
        for l in layers
    }
    option_keys = ["no", "yes"]

    report = FusionComparisonExperiment.run_comparison(
        X_by_layer=X_by_layer,
        y=y,
        option_keys=option_keys,
        train_split=0.7,
        dataset_name="synthetic_safety_bench",
        candidate_layers=layers,
    )

    assert isinstance(report, FusionExperimentReport)
    assert report.num_samples == 80
    assert report.best_architecture in ["concatenation", "weighted_average", "learned_gating"]
    assert len(report.fusion_architectures) == 3

    summary = report.summary()
    assert "Multi-Layer Representation Fusion Experiment" in summary
    assert "Single Best Layer" in summary
    assert "concatenation" in summary


def test_real_model_layer_trajectory_matches_config():
    """Real HF causal model: hidden states inspected, layer count matches config.

    Uses sshleifer/tiny-gpt2 (2 layers). Skipped when offline or when the
    hf/torch stack is unavailable. Verifies: reported layers against the model
    config, per-layer hidden states are distinct (not repeated final values),
    per-layer logit-lens logprobs are normalized, and the emergence layer is
    mathematically defined.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from anydecision.backends.transformers import TransformersBackend

        backend = TransformersBackend("sshleifer/tiny-gpt2")
    except Exception as e:
        pytest.skip(f"tiny-gpt2 unavailable (offline?): {e}")

    meta = backend.get_metadata()
    assert meta.num_layers == 2  # tiny-gpt2 config: n_layer=2

    prompt = "Is this safe? Answer yes or no:"
    candidates = {"yes": "yes", "no": "no"}
    hidden = backend.get_layer_hidden_states(prompt)
    assert sorted(hidden.keys()) == [1, 2]
    assert hidden[1].shape == hidden[2].shape
    assert not np.allclose(hidden[1], hidden[2]), "layers must expose distinct states"

    layer_lps = backend.get_layer_logprobs(prompt, candidates)
    assert sorted(layer_lps.keys()) == [1, 2]
    for lps in layer_lps.values():
        probs = np.exp(list(lps.values()))
        assert pytest.approx(float(np.sum(probs)), abs=1e-5) == 1.0

    engine = DecisionEngine(model=backend)
    result = engine.analyze_layer_trajectory(
        Question.choice("Is this safe?", ["yes", "no"])
    )
    assert isinstance(result, LayerTrajectoryResult)
    assert all(1 <= cp.layer_idx <= meta.num_layers for cp in result.trajectory)
    assert result.decision_emergence_layer in [cp.layer_idx for cp in result.trajectory]
