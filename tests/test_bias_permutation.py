"""Unit tests for option permutation and bias invariance."""

import pytest
from anydecision.bias.aggregation import aggregate_distributions
from anydecision.bias.permutation import (
    compute_permutation_invariance_metrics,
    generate_permutations,
)
from anydecision.bias.templates import DEFAULT_TEMPLATES, PromptTemplate
from anydecision.core.question import Question


def test_generate_permutations():
    items = ["optA", "optB", "optC"]
    perms = generate_permutations(items, max_permutations=4)
    assert len(perms) <= 4
    # First permutation is canonical
    assert perms[0] == ["optA", "optB", "optC"]
    # Second is reversed
    assert perms[1] == ["optC", "optB", "optA"]


def test_permutation_invariance_metrics():
    dist1 = {"a": 0.8, "b": 0.2}
    dist2 = {"a": 0.7, "b": 0.3}
    metrics = compute_permutation_invariance_metrics([dist1, dist2], ["a", "b"])
    assert metrics["permutation_agreement"] == 1.0
    assert metrics["position_bias_score"] >= 0.0


def test_aggregate_distributions():
    dists = [
        {"a": 0.8, "b": 0.2},
        {"a": 0.6, "b": 0.4},
    ]
    mean_agg = aggregate_distributions(dists, method="prob_mean")
    assert round(mean_agg["a"], 2) == 0.70
    assert round(mean_agg["b"], 2) == 0.30

    logit_agg = aggregate_distributions(dists, method="logit_mean")
    assert "a" in logit_agg and "b" in logit_agg


def test_prompt_template_rendering():
    q = Question.choice("Choose color", ["red", "blue"])
    tmpl = DEFAULT_TEMPLATES.get("minimal")
    rendered = tmpl.render(q)
    assert "Choose color" in rendered
    assert "red" in rendered
    assert "blue" in rendered

