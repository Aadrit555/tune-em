"""Unit tests for calibration methods, conformal prediction, and calibration metrics."""

import numpy as np
import pytest
from anydecision.calibration.conformal import ConformalPredictor
from anydecision.calibration.isotonic import IsotonicCalibration
from anydecision.calibration.metrics import (
    compute_adaptive_ece,
    compute_brier_score,
    compute_ece,
    compute_nll,
)
from anydecision.calibration.platt import PlattScaling
from anydecision.calibration.report import CalibrationReport
from anydecision.calibration.temperature import TemperatureScaling
from anydecision.calibration.vector import VectorScaling


def test_temperature_scaling_fit_and_calibrate():
    # Overconfident synthetic predictions
    probs = np.array([
        [0.95, 0.05],
        [0.90, 0.10],
        [0.20, 0.80],
        [0.30, 0.70],
    ])
    labels = np.array([0, 1, 1, 0])

    ts = TemperatureScaling()
    ts.fit(probs, labels)
    assert ts.fitted is True
    assert ts.temperature > 0.0

    calibrated = ts.calibrate(probs)
    assert calibrated.shape == probs.shape
    assert np.allclose(np.sum(calibrated, axis=-1), 1.0)


def test_vector_scaling():
    probs = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.3, 0.7],
    ])
    labels = np.array([0, 0, 1, 1])
    vs = VectorScaling()
    vs.fit(probs, labels)
    assert vs.fitted is True
    cal = vs.calibrate(probs)
    assert cal.shape == probs.shape
    assert np.allclose(np.sum(cal, axis=-1), 1.0)


def test_isotonic_calibration():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.3, 0.7],
    ])
    labels = np.array([0, 0, 1, 1])
    iso = IsotonicCalibration()
    iso.fit(probs, labels)
    assert iso.fitted is True
    cal = iso.calibrate(probs)
    assert np.allclose(np.sum(cal, axis=-1), 1.0)


def test_platt_scaling():
    probs = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.3, 0.7],
    ])
    labels = np.array([0, 0, 1, 1])
    platt = PlattScaling()
    platt.fit(probs, labels)
    assert platt.fitted is True
    cal = platt.calibrate(probs)
    assert np.allclose(np.sum(cal, axis=-1), 1.0)


def test_conformal_prediction_set():
    probs = np.array([
        [0.85, 0.15],
        [0.75, 0.25],
        [0.90, 0.10],
        [0.10, 0.90],
    ])
    labels = np.array([0, 0, 0, 1])
    cp = ConformalPredictor(alpha=0.10)
    cp.fit(probs, labels)
    assert cp.fitted is True

    # Test set prediction
    test_p = np.array([0.55, 0.45])
    pset = cp.predict_set(test_p, ["yes", "no"])
    assert len(pset) >= 1


def test_calibration_metrics_and_report():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.2, 0.8],
    ])
    labels = np.array([0, 0, 1, 1])
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)
    accs = (preds == labels).astype(np.float64)

    ece = compute_ece(confs, accs, num_bins=5)
    aece = compute_adaptive_ece(confs, accs, num_bins=2)
    brier = compute_brier_score(probs, labels)
    nll = compute_nll(probs, labels)

    assert 0.0 <= ece <= 1.0
    assert 0.0 <= aece <= 1.0
    assert brier >= 0.0
    assert nll >= 0.0

    report = CalibrationReport.evaluate(probs, labels)
    assert report.accuracy == 1.0
    assert "STATISTICAL CALIBRATION REPORT" in report.summary()

