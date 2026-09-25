"""Statistical calibration layers, conformal prediction, and evaluation metrics."""

from anydecision.calibration.base import BaseCalibrator
from anydecision.calibration.conformal import ConformalPredictor
from anydecision.calibration.isotonic import IsotonicCalibration
from anydecision.calibration.metrics import (
    compute_adaptive_ece,
    compute_brier_score,
    compute_ece,
    compute_nll,
    compute_reliability_diagram_data,
    compute_selective_prediction_curve,
)
from anydecision.calibration.platt import PlattScaling
from anydecision.calibration.report import CalibrationReport
from anydecision.calibration.temperature import TemperatureScaling
from anydecision.calibration.vector import VectorScaling

__all__ = [
    "BaseCalibrator",
    "CalibrationReport",
    "ConformalPredictor",
    "IsotonicCalibration",
    "PlattScaling",
    "TemperatureScaling",
    "VectorScaling",
    "compute_adaptive_ece",
    "compute_brier_score",
    "compute_ece",
    "compute_nll",
    "compute_reliability_diagram_data",
    "compute_selective_prediction_curve",
]

