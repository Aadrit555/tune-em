"""Agent-loop evaluation: Comparing plain LLM vs L0 vs L1 vs L2 vs Abstention."""

from anydecision import DecisionEngine
from anydecision.evaluation.agent_loop import AgentWorkflowEvaluator
from anydecision.evaluation.datasets import create_agent_safety_tasks

engine = DecisionEngine(model="mock")
evaluator = AgentWorkflowEvaluator(engine)

tasks = create_agent_safety_tasks(n_tasks=25)

print("=" * 70)
print("     AUTONOMOUS AGENT DECISION LOOP BENCHMARK")
print("=" * 70)

strategies = [
    "plain_text_greedy",
    "l0_raw",
    "l1_zero_label",
    "l2_with_abstention",
]

for strat in strategies:
    report = evaluator.evaluate_strategy(tasks, strategy=strat)
    print(f"\nStrategy: [{strat.upper()}]")
    print(f"  Task Success Rate:         {report.task_success_rate * 100:.1f}%")
    print(f"  Catastrophic Failure Rate: {report.catastrophic_failure_rate * 100:.1f}%")
    print(f"  Abstentions:               {report.abstained_actions}/{report.total_tasks}")
    print(f"  Mean Latency:              {report.mean_decision_latency_ms:.2f} ms")
    print(f"  Simulated Token Cost:      {report.simulated_token_cost} tokens")

print("=" * 70)

