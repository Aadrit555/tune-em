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


def test_from_action_costs_explicit_mapping_no_name_guessing():
    # fraud -> fraud_block must NOT be inferred from names; explicit mapping required.
    costs = {"fraud_block": 0.0, "approve": 0.0, "manual_review": -2.0}
    states = ["fraud", "legitimate"]
    default = UtilityMatrix.from_action_costs(costs, states)
    # Without explicit mapping, fraud_block does not match "fraud" (exact equality only).
    assert default.matrix["fraud_block"]["fraud"] == pytest.approx(-20.0)
    explicit = UtilityMatrix.from_action_costs(
        costs, states, correct_action_for_state={"fraud": "fraud_block", "legitimate": "approve"}
    )
    assert explicit.matrix["fraud_block"]["fraud"] == pytest.approx(10.0)
    assert explicit.matrix["approve"]["legitimate"] == pytest.approx(10.0)
    assert explicit.matrix["fraud_block"]["legitimate"] == pytest.approx(-20.0)
    # Safe fallback keeps constant cost regardless of state.
    assert explicit.matrix["manual_review"]["fraud"] == pytest.approx(-2.0)
    assert explicit.matrix["manual_review"]["legitimate"] == pytest.approx(-2.0)
    # Case-insensitive guessing is never performed.
    case_costs = {"FRAUD": 0.0}
    case_mat = UtilityMatrix.from_action_costs(case_costs, ["fraud"])
    assert case_mat.matrix["FRAUD"]["fraud"] == pytest.approx(-20.0)


def test_standard_classification_costs_permutation_invariance():
    classes_a = ["fraud", "legitimate"]
    classes_b = ["legitimate", "fraud"]
    m_a = UtilityMatrix.standard_classification_costs(
        classes_a, cost_false_positive=5.0, cost_false_negative=20.0, positive_class="fraud"
    )
    m_b = UtilityMatrix.standard_classification_costs(
        classes_b, cost_false_positive=5.0, cost_false_negative=20.0, positive_class="fraud"
    )
    # Missing the positive class costs FN regardless of ordering.
    assert m_a.matrix["act_legitimate"]["fraud"] == pytest.approx(-20.0)
    assert m_b.matrix["act_legitimate"]["fraud"] == pytest.approx(-20.0)
    assert m_a.matrix["act_fraud"]["legitimate"] == pytest.approx(-5.0)
    assert m_b.matrix["act_fraud"]["legitimate"] == pytest.approx(-5.0)
    # Symmetric default: no ordering dependence.
    s_a = UtilityMatrix.standard_classification_costs(classes_a)
    s_b = UtilityMatrix.standard_classification_costs(classes_b)
    assert s_a.matrix["act_legitimate"]["fraud"] == pytest.approx(
        s_b.matrix["act_legitimate"]["fraud"]
    )
    # Invalid positive class fails loudly.
    with pytest.raises(ValueError):
        UtilityMatrix.standard_classification_costs(classes_a, positive_class="unknown")
