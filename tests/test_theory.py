"""Unit tests for decision theory, expected utility maximization, and execution compiler."""

import pytest
from anydecision import DecisionEngine, DecisionPolicy, Question
from anydecision.theory.compiler import DecisionCompiler
from anydecision.theory.utility import UtilityMatrix


def test_utility_matrix_expected_utility():
    # States: fraud (0.2), legitimate (0.8)
    # Actions: block, allow, review
    actions = {
        "block": {"fraud": 10.0, "legitimate": -50.0},
        "allow": {"fraud": -100.0, "legitimate": 10.0},
        "review": {"fraud": 0.0, "legitimate": -2.0},
    }
    matrix = UtilityMatrix(
        actions=["block", "allow", "review"],
        states=["fraud", "legitimate"],
        matrix=actions,
    )
    probs = {"fraud": 0.20, "legitimate": 0.80}
    eus = matrix.compute_expected_utilities(probs)

    # EU(block) = 0.2*10 + 0.8*(-50) = 2 - 40 = -38
    # EU(allow) = 0.2*(-100) + 0.8*(10) = -20 + 8 = -12
    # EU(review) = 0.2*(0) + 0.8*(-2) = -1.6
    assert pytest.approx(eus["block"], abs=1e-3) == -38.0
    assert pytest.approx(eus["allow"], abs=1e-3) == -12.0
    assert pytest.approx(eus["review"], abs=1e-3) == -1.6

    best_act, best_eu, regret, all_eus = matrix.select_optimal_action(probs)
    # Review is the highest expected utility!
    assert best_act == "review"
    assert pytest.approx(best_eu, abs=1e-3) == -1.6
    assert pytest.approx(regret, abs=1e-3) == (-1.6 - (-12.0))


def test_engine_decide_with_actions_and_utility():
    engine = DecisionEngine(model="mock")
    q = Question.choice("Classify transaction", ["fraud", "legitimate"])

    # High cost for false negative fraud
    action_costs = {
        "approve": 0.0,
        "reject": -10.0,
        "human_review": -2.0,
    }
    res = engine.decide(q, actions=action_costs)

    assert res.selected_action in ("approve", "reject", "human_review")
    assert len(res.expected_utilities) == 3
    assert res.optimal_action_utility is not None


def test_engine_decide_with_escalation():
    engine = DecisionEngine(model="mock")
    q = Question.choice("Customer refund status", ["eligible", "ineligible"])

    # Custom matrix where ambiguous probability triggers human escalation
    matrix = UtilityMatrix(
        actions=["auto_refund", "deny", "human_review"],
        states=["eligible", "ineligible"],
        matrix={
            "auto_refund": {"eligible": 5.0, "ineligible": -80.0},
            "deny": {"eligible": -60.0, "ineligible": 5.0},
            "human_review": {"eligible": -1.0, "ineligible": -1.0},
        },
    )
    res = engine.decide(q, utility_matrix=matrix)
    assert res.selected_action is not None
    assert isinstance(res.escalated, bool)


def test_decision_compiler_plan_explain():
    engine = DecisionEngine(model="mock")
    q = Question.choice(
        "Which support team?",
        ["billing", "technical", "sales"],
    )
    plan = engine.compile(q)

    assert plan.question_id == q.id
    assert plan.num_candidate_options == 3
    assert plan.total_forward_passes >= 1
    assert "COMPILED DECISION EXECUTION PLAN" in plan.explain()
