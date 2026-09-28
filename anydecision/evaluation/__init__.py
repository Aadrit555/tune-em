"""Evaluation benchmarks, agent loop simulation, and visualizer."""

from anydecision.evaluation.agent_loop import (
    AgentTask,
    AgentWorkflowEvaluator,
    AgentWorkflowReport,
)
from anydecision.evaluation.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSample,
)
from anydecision.evaluation.datasets import (
    create_agent_safety_tasks,
    create_customer_escalation_benchmark,
    create_topic_categorization_benchmark,
)
from anydecision.evaluation.economics import (
    DecisionEconomicsEvaluator,
    DecisionEconomicsReport,
    EconomicsConfig,
    LevelComparisonRecord,
)

def __getattr__(name: str):  # type: ignore[no-redef]
    """Lazily load the matplotlib-dependent visualizer (research extra)."""
    if name == "ResearchVisualizer":
        try:
            from anydecision.evaluation.visualizer import ResearchVisualizer as _RV

            return _RV
        except ImportError as e:
            raise ImportError(
                "ResearchVisualizer requires the 'research' extra (matplotlib). "
                "Install it via: pip install 'anydecision[research]'"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentTask",
    "AgentWorkflowEvaluator",
    "AgentWorkflowReport",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSample",
    "DecisionEconomicsEvaluator",
    "DecisionEconomicsReport",
    "EconomicsConfig",
    "LevelComparisonRecord",
    "ResearchVisualizer",
    "create_agent_safety_tasks",
    "create_customer_escalation_benchmark",
    "create_topic_categorization_benchmark",
]


