"""anydecision: A typed, uncertainty-aware decision runtime for open-weight LLMs."""

from anydecision._version import __version__
from anydecision.artifacts.saver import (
    load_calibration_artifact,
    save_calibration_artifact,
)
from anydecision.backends.base import BaseBackend, ModelMetadata
from anydecision.bias.templates import PromptTemplate
from anydecision.calibration.base import BaseCalibrator
from anydecision.calibration.conformal import ConformalPredictor
from anydecision.calibration.metrics import (
    compute_adaptive_ece,
    compute_brier_score,
    compute_ece,
    compute_nll,
)
from anydecision.calibration.report import CalibrationReport
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

from anydecision.theory.compiler import CompiledExecutionPlan, DecisionCompiler
from anydecision.theory.utility import UtilityMatrix

__all__ = [
    "__version__",
    "AbstentionPolicy",
    "AnswerType",
    "BaseBackend",
    "BaseCalibrator",
    "CalibrationReport",
    "CompiledExecutionPlan",
    "ConformalPredictor",
    "Decision",
    "DecisionCompiler",
    "DecisionEngine",
    "DecisionLevel",
    "DecisionPolicy",
    "DecisionTrace",
    "Diagnostics",
    "ModelMetadata",
    "OptionDefinition",
    "PromptTemplate",
    "Question",
    "ReadoutStrategy",
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

