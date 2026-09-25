"""anydecision: A typed, uncertainty-aware decision runtime for open-weight LLMs."""

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
    "DistributionShiftEvaluator",
    "DistributionShiftReport",
    "DistributionShiftType",
    "DriftAlert",
    "Decision",
    "DecisionCompiler",
    "DecisionEngine",
    "DecisionLevel",
    "DecisionPolicy",
    "DecisionTrace",
    "Diagnostics",
    "FusionComparisonExperiment",
    "FusionExperimentReport",
    "FusionModelCard",
    "FusionStrategy",
    "HierarchicalCalibrator",
    "HierarchicalDiagnostics",
    "LayerCheckpoint",
    "LayerTrajectoryAnalyzer",
    "LayerTrajectoryResult",
    "ModelMetadata",
    "MultiLayerFusionHead",
    "OptionDefinition",
    "PromptTemplate",
    "Question",
    "ReadoutStrategy",
    "SelectiveConformalPredictor",
    "SelectiveConformalResult",
    "TemperatureScaling",
    "UtilityMatrix",
    "VectorScaling",
    "compute_adaptive_ece",
    "compute_brier_score",
    "compute_ece",
    "compute_nll",
    "load_calibration_artifact",
    "save_calibration_artifact",
]

