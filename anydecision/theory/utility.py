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
    ) -> UtilityMatrix:
        """Create utility matrix where actions have intrinsic costs/utilities relative to matched state."""
        actions = list(action_costs.keys())
        matrix: Dict[str, Dict[str, float]] = {}

        for act in actions:
            matrix[act] = {}
            base_cost = action_costs[act]
            for state in states:
                # If action name matches state (e.g. state="fraud", act="fraud_block"), reward = 0, penalty if wrong
                if act.lower() == state.lower():
                    matrix[act][state] = 10.0 + base_cost
                elif act in ("human_review", "escalate", "abstain"):
                    # Safe fallbacks take the constant cost regardless of state
                    matrix[act][state] = base_cost
                else:
                    # Incorrect action penalty
                    matrix[act][state] = -20.0 + base_cost

        return cls(actions=actions, states=states, matrix=matrix)

    @classmethod
    def standard_classification_costs(
        cls,
        classes: List[str],
        cost_false_positive: float = 5.0,
        cost_false_negative: float = 20.0,
        human_review_cost: float = 2.0,
    ) -> UtilityMatrix:
        """Construct standard asymmetric cost matrix with automated human-in-the-loop fallback."""
        actions = [f"act_{c}" for c in classes] + ["human_review"]
        matrix: Dict[str, Dict[str, float]] = {}

        for c_act in classes:
            act_name = f"act_{c_act}"
            matrix[act_name] = {}
            for c_state in classes:
                if c_act == c_state:
                    matrix[act_name][c_state] = 0.0  # Zero cost for correct classification
                else:
                    matrix[act_name][c_state] = -cost_false_negative if c_state == classes[0] else -cost_false_positive

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
