"""Unit tests for agent loop evaluation workflow."""

import pytest
from anydecision import DecisionEngine
from anydecision.evaluation.agent_loop import AgentWorkflowEvaluator
from anydecision.evaluation.datasets import create_agent_safety_tasks


def test_agent_workflow_evaluator():
    engine = DecisionEngine(model="mock")
    evaluator = AgentWorkflowEvaluator(engine)
    tasks = create_agent_safety_tasks(n_tasks=6)

    report_l0 = evaluator.evaluate_strategy(tasks, strategy="l0_raw")
    assert report_l0.total_tasks == 6
    assert 0.0 <= report_l0.task_success_rate <= 1.0

    report_abstain = evaluator.evaluate_strategy(tasks, strategy="l2_with_abstention")
    assert report_abstain.total_tasks == 6
    assert report_abstain.abstained_actions >= 0

