"""Run autonomous agent loop evaluation across decision paradigms."""

import json
from pathlib import Path
from anydecision import DecisionEngine
from anydecision.evaluation.agent_loop import AgentWorkflowEvaluator
from anydecision.evaluation.datasets import create_agent_safety_tasks


def main():
    print("=" * 65)
    print("     STARTING AGENT-LOOP SAFETY & RELIABILITY BENCHMARK")
    print("=" * 65)

    engine = DecisionEngine(model="mock")
    evaluator = AgentWorkflowEvaluator(engine)
    tasks = create_agent_safety_tasks(n_tasks=50)

    strategies = [
        "plain_text_greedy",
        "l0_raw",
        "l1_zero_label",
        "l2_with_abstention",
    ]

    reports = {}
    for s in strategies:
        rep = evaluator.evaluate_strategy(tasks, strategy=s)
        reports[s] = rep.model_dump()
        print(f"\nStrategy: [{s}]")
        print(f"  Task Success Rate:         {rep.task_success_rate * 100:.1f}%")
        print(f"  Catastrophic Failure Rate: {rep.catastrophic_failure_rate * 100:.1f}%")
        print(f"  Abstentions:               {rep.abstained_actions}/{rep.total_tasks}")
        print(f"  Mean Latency:              {rep.mean_decision_latency_ms:.2f} ms")

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "agent_eval_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print(f"\n[Saved Agent Workflow Evaluation Report to: {out_file}]")
    print("=" * 65)


if __name__ == "__main__":
    main()
