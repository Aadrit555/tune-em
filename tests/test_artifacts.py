"""Unit tests for calibration artifact serialization, schema validation, and integrity."""

from pathlib import Path
import pytest
from anydecision.artifacts.saver import (
    IncompatibleArtifactError,
    IntegrityError,
    load_calibration_artifact,
    save_calibration_artifact,
)
from anydecision.calibration.temperature import TemperatureScaling


def test_save_and_load_artifact(tmp_path: Path):
    filepath = tmp_path / "test_head.json"
    ts = TemperatureScaling(temperature=1.65)
    ts.fitted = True

    manifest = save_calibration_artifact(
        calibrator=ts,
        filepath=filepath,
        model="test-model",
        model_revision="v1",
        tokenizer_revision="v1",
        backend="mock",
    )

    assert filepath.exists()
    assert manifest.model == "test-model"
    assert manifest.integrity.sha256 != ""

    # Reload
    loaded_cal, loaded_manifest = load_calibration_artifact(
        filepath, expected_model="test-model", expected_revision="v1"
    )
    assert isinstance(loaded_cal, TemperatureScaling)
    assert pytest.approx(loaded_cal.temperature, abs=1e-4) == 1.65


def test_artifact_model_mismatch_prevention(tmp_path: Path):
    filepath = tmp_path / "mismatch_head.json"
    ts = TemperatureScaling(temperature=1.2)
    save_calibration_artifact(
        calibrator=ts,
        filepath=filepath,
        model="llama-3-8b",
        model_revision="main",
    )

    with pytest.raises(IncompatibleArtifactError):
        load_calibration_artifact(filepath, expected_model="mistral-7b")


def test_artifact_tamper_detection(tmp_path: Path):
    filepath = tmp_path / "tampered_head.json"
    ts = TemperatureScaling(temperature=1.2)
    save_calibration_artifact(
        calibrator=ts,
        filepath=filepath,
        model="test-model",
    )

    # Tamper with the file contents
    content = filepath.read_text(encoding="utf-8")
    tampered = content.replace("1.2", "9.99")
    filepath.write_text(tampered, encoding="utf-8")

    with pytest.raises(IntegrityError):
        load_calibration_artifact(filepath, expected_model="test-model")

