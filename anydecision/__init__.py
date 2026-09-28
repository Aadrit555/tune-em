"""anydecision: A typed, uncertainty-aware decision runtime for open-weight LLMs."""

from __future__ import annotations

from typing import Any, Sequence

from anydecision._version import __version__
from anydecision.adaptive.router import AdaptiveComputeConfig, AdaptiveComputeRouter
from anydecision.artifacts.saver import (
    load_calibration_artifact,
    save_calibration_artifact,
)
from anydecision.backends.base import BaseBackend, ModelMetadata
from anydecision.bias.templates import PromptTemplate
from anydecision.calibration.active import (
    ActiveCalibrationBenchmark,
    ActiveCalibrationReport,
    ActiveCalibrator,
    ActiveSelectionCriterion,
)
from anydecision.calibration.base import BaseCalibrator
from anydecision.calibration.conformal import ConformalPredictor
from anydecision.calibration.drift import (
    CalibrationDriftMonitor,
    DistributionShiftEvaluator,
    DistributionShiftReport,
    DistributionShiftType,
    DriftAlert,
)
from anydecision.calibration.hierarchical import (
    HierarchicalCalibrator,
    HierarchicalDiagnostics,
)
from anydecision.calibration.metrics import (
    compute_adaptive_ece,
    compute_brier_score,
    compute_ece,
    compute_nll,
)
from anydecision.calibration.report import CalibrationReport
from anydecision.calibration.selective_conformal import (
    SelectiveConformalPredictor,
    SelectiveConformalResult,
)
from anydecision.calibration.temperature import TemperatureScaling
from anydecision.calibration.vector import VectorScaling
from anydecision.core.decision import Decision
from anydecision.core.engine import DecisionEngine
from anydecision.core.policies import AbstentionPolicy, DecisionPolicy
from anydecision.core.question import Question
from anydecision.core.types import (
    AnswerType,
    DecisionLevel,
    DecisionTrace,
    Diagnostics,
    OptionDefinition,
    ReadoutStrategy,
)

from anydecision.backends.consistency import CrossBackendConsistencyChecker
from anydecision.core.context import DecisionContext, InjectionResistantContext
from anydecision.core.semantic_option import SemanticOptionRegistry
from anydecision.ensemble.model_disagreement import (
    MultiModelDecisionResult,
    MultiModelEnsemble,
    MultiModelEnsembleDisagreement,
)
from anydecision.games.doom import (
    DoomActionOutcome,
    DoomCombatBenchmarkRunner,
    DoomEnemy,
    DoomGameState,
    DoomScenarioEnvironment,
    DoomTacticalAgent,
)
from anydecision.models.decision_head import (
    FastOptionScorer,
    NonAutoregressiveDecisionHead,
)
from anydecision.representations.fusion import (
    FusionComparisonExperiment,
    FusionExperimentReport,
    FusionModelCard,
    FusionStrategy,
    MultiLayerFusionHead,
)
from anydecision.representations.trajectory import (
    LayerCheckpoint,
    LayerTrajectoryAnalyzer,
    LayerTrajectoryResult,
)
from anydecision.theory.compiler import CompiledExecutionPlan, DecisionCompiler
from anydecision.theory.utility import UtilityMatrix
from anydecision.tui.app import TerminalUI, run_tui

__all__ = [
    "__version__",
    "AbstentionPolicy",
    "ActiveCalibrationBenchmark",
    "ActiveCalibrationReport",
    "ActiveCalibrator",
    "ActiveSelectionCriterion",
    "AdaptiveComputeConfig",
    "AdaptiveComputeRouter",
    "AnswerType",
    "BaseBackend",
    "BaseCalibrator",
    "CalibrationDriftMonitor",
    "CalibrationReport",
    "CompiledExecutionPlan",
    "ConformalPredictor",
    "CrossBackendConsistencyChecker",
    "DistributionShiftEvaluator",
    "DistributionShiftReport",
    "DistributionShiftType",
    "DoomActionOutcome",
    "DoomCombatBenchmarkRunner",
    "DoomEnemy",
    "DoomGameState",
    "DoomScenarioEnvironment",
    "DoomTacticalAgent",
    "DriftAlert",
    "Decision",
    "DecisionCompiler",
    "DecisionContext",
    "DecisionEngine",
    "DecisionLevel",
    "DecisionPolicy",
    "DecisionTrace",
    "Diagnostics",
    "FastOptionScorer",
    "FusionComparisonExperiment",
    "FusionExperimentReport",
    "FusionModelCard",
    "FusionStrategy",
    "HierarchicalCalibrator",
    "HierarchicalDiagnostics",
    "InjectionResistantContext",
    "LayerCheckpoint",
    "LayerTrajectoryAnalyzer",
    "LayerTrajectoryResult",
    "ModelMetadata",
    "MultiLayerFusionHead",
    "MultiModelDecisionResult",
    "MultiModelEnsemble",
    "MultiModelEnsembleDisagreement",
    "NonAutoregressiveDecisionHead",
    "OptionDefinition",
    "PromptTemplate",
    "Question",
    "ReadoutStrategy",
    "SelectiveConformalPredictor",
    "SelectiveConformalResult",
    "SemanticOptionRegistry",
    "TemperatureScaling",
    "TerminalUI",
    "UtilityMatrix",
    "VectorScaling",
    "choose",
    "compute_adaptive_ece",
    "compute_brier_score",
    "compute_ece",
    "compute_nll",
    "load_calibration_artifact",
    "run_tui",
    "save_calibration_artifact",
]

_GLOBAL_ENGINE = None


def choose(
    prompt: str,
    choices: Sequence[str],
    model: str = "mock",
    level: str = "L0",
    **kwargs: Any,
) -> Decision:
    """Universal top-level decision function (analogous to von.choose).

    Evaluates any arbitrary prompt and candidate options directly from model distributions,
    returning a strongly typed, calibrated Decision object.
    """
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None or _GLOBAL_ENGINE.metadata.model_name != model:
        _GLOBAL_ENGINE = DecisionEngine(model=model)
    return _GLOBAL_ENGINE.choose(prompt, choices, level=level, **kwargs)


