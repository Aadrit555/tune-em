"""Decision-theoretic utility matrices, action costs, and expected utility optimization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field


class UtilityMatrix(BaseModel):
    """Cost/Utility matrix governing rational decision-making under uncertainty.

    Instead of assuming every probability corresponds to an autonomous action,
    rational agents select actions that maximize Expected Utility:
        EU(a) = sum_{y} P(y | x) * U(a, y)
    """
    actions: List[str] = Field(description="List of available operational actions (e.g. approve, reject, escalate).")
    states: List[str] = Field(description="List of possible true world states / question outcomes.")
    matrix: Dict[str, Dict[str, float]] = Field(
        description="Nested mapping: action -> state -> utility value (higher is better)."
    )

    @classmethod
    def from_action_costs(
        cls,
        action_costs: Dict[str, float],
        states: List[str],
        correct_action_for_state: Optional[Dict[str, str]] = None,
        correct_reward: float = 10.0,
        incorrect_penalty: float = -20.0,
        safe_actions: Tuple[str, ...] = ("human_review", "escalate", "abstain", "manual_review"),
    ) -> UtilityMatrix:
        """Create utility matrix with an explicit action<->state mapping.

        No semantic relationship is inferred from action/state names.  The
        caller must pass ``correct_action_for_state`` (state -> correct
        action) for non-trivial mappings such as ``{"fraud": "fraud_block"}``.
        When it is omitted, only exact string equality (``act == state``)
        counts as a match; case-insensitive / substring guessing is never
        performed.  Safe fallback actions receive ``base_cost`` in every
        state; all other mismatches receive ``incorrect_penalty + base_cost``.
        """
        actions = list(action_costs.keys())
        matrix: Dict[str, Dict[str, float]] = {}

        for act in actions:
            matrix[act] = {}
            base_cost = action_costs[act]
            for state in states:
                if act in safe_actions:
                    # Safe fallbacks take the constant cost regardless of state.
                    matrix[act][state] = base_cost
                    continue
                if correct_action_for_state is not None:
                    is_correct = correct_action_for_state.get(state) == act
                else:
                    # Explicit default: exact equality only, never fuzzy matching.
                    is_correct = (act == state)
                if is_correct:
                    matrix[act][state] = correct_reward + base_cost
                else:
                    matrix[act][state] = incorrect_penalty + base_cost

        return cls(actions=actions, states=states, matrix=matrix)

    @classmethod
    def standard_classification_costs(
        cls,
        classes: List[str],
        cost_false_positive: float = 5.0,
        cost_false_negative: float = 20.0,
        human_review_cost: float = 2.0,
        positive_class: Optional[str] = None,
    ) -> UtilityMatrix:
        """Construct standard asymmetric cost matrix with human-in-the-loop fallback.

        ``positive_class`` names the state for which a miss counts as a false
        negative (``-cost_false_negative``); misses on any other state cost
        ``-cost_false_positive``.  Cost semantics are therefore invariant to
        the ordering of ``classes``.  If ``positive_class`` is None, all
        misclassifications cost ``-cost_false_positive`` (symmetric) so that
        no ordering-dependent assumption is made silently.  ``positive_class``
        must be a member of ``classes`` when given.
        """
        if positive_class is not None and positive_class not in classes:
            raise ValueError(
                f"positive_class={positive_class!r} must be one of classes={classes}."
            )
        actions = [f"act_{c}" for c in classes] + ["human_review"]
        matrix: Dict[str, Dict[str, float]] = {}

        for c_act in classes:
            act_name = f"act_{c_act}"
            matrix[act_name] = {}
            for c_state in classes:
                if c_act == c_state:
                    matrix[act_name][c_state] = 0.0  # Zero cost for correct classification
                else:
                    if positive_class is None:
                        matrix[act_name][c_state] = -cost_false_positive
                    else:
                        matrix[act_name][c_state] = (
                            -cost_false_negative
                            if c_state == positive_class
                            else -cost_false_positive
                        )

        # Human review cost
        matrix["human_review"] = {c_state: -human_review_cost for c_state in classes}

        return cls(actions=actions, states=classes, matrix=matrix)

    def compute_expected_utilities(
        self,
        probabilities: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute expected utility for every defined action given state probability distribution.

        EU(a) = sum_{y} P(y | x) * U(a, y)
        """
        expected_utilities: Dict[str, float] = {}

        for act in self.actions:
            eu = 0.0
            act_utilities = self.matrix.get(act, {})
            for state, prob in probabilities.items():
                util = act_utilities.get(state, 0.0)
                eu += prob * util
            expected_utilities[act] = float(eu)

        return expected_utilities

    def select_optimal_action(
        self,
        probabilities: Dict[str, float],
    ) -> Tuple[str, float, float, Dict[str, float]]:
        """Select action maximizing expected utility.

        Returns:
            Tuple of (optimal_action, max_utility, regret, all_expected_utilities)
        """
        eus = self.compute_expected_utilities(probabilities)
        sorted_actions = sorted(eus.items(), key=lambda kv: kv[1], reverse=True)
        best_act, best_eu = sorted_actions[0]
        second_best_eu = sorted_actions[1][1] if len(sorted_actions) > 1 else best_eu
        regret = float(best_eu - second_best_eu)

        return best_act, best_eu, regret, eus
