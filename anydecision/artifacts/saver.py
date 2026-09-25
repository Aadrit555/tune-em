"""Artifact serialization, safe loading, and cryptographic verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from anydecision.artifacts.schema import ArtifactManifest, IntegrityBlock
from anydecision.calibration.base import BaseCalibrator
from anydecision.calibration.conformal import ConformalPredictor
from anydecision.calibration.isotonic import IsotonicCalibration
from anydecision.calibration.platt import PlattScaling
from anydecision.calibration.temperature import TemperatureScaling
from anydecision.calibration.vector import VectorScaling


class IncompatibleArtifactError(ValueError):
    """Raised when an artifact cannot be safely loaded due to model or version mismatch."""
    pass


class IntegrityError(ValueError):
    """Raised when an artifact checksum does not match its payload."""
    pass


def _compute_payload_hash(payload_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of dict payload."""
    canonical_json = json.dumps(payload_dict, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def save_calibration_artifact(
    calibrator: BaseCalibrator,
    filepath: Union[str, Path],
    model: str,
    model_revision: str = "main",
    tokenizer_revision: str = "main",
    backend: str = "transformers",
    question_schema: str = "generic",
    training_data_hash: str = "0000000000000000",
) -> ArtifactManifest:
    """Safely serialize a trained calibration head to a versioned JSON artifact."""
    cal_dict = calibrator.to_dict()
    head_type = cal_dict.get("type", "unknown")

    # Construct payload without integrity block to compute hash
    payload = {
        "model": model,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "question_schema": question_schema,
        "backend": backend,
        "head": head_type,
        "calibration": cal_dict,
        "training_data_hash": training_data_hash,
    }
    sha256 = _compute_payload_hash(payload)

    manifest = ArtifactManifest(
        model=model,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        question_schema=question_schema,
        backend=backend,
        head=head_type,
        calibration=cal_dict,
        training_data_hash=training_data_hash,
        integrity=IntegrityBlock(sha256=sha256),
    )

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    return manifest


def load_calibration_artifact(
    filepath: Union[str, Path],
    expected_model: Optional[str] = None,
    expected_revision: Optional[str] = None,
    verify_integrity: bool = True,
) -> Tuple[BaseCalibrator, ArtifactManifest]:
    """Safely load and validate a calibration artifact.

    Validates schema, cryptographic hash, and prevents model mismatch.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Calibration artifact not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = ArtifactManifest.model_validate(data)

    # 1. Cryptographic integrity check
    if verify_integrity:
        payload = {
            "model": manifest.model,
            "model_revision": manifest.model_revision,
            "tokenizer_revision": manifest.tokenizer_revision,
            "question_schema": manifest.question_schema,
            "backend": manifest.backend,
            "head": manifest.head,
            "calibration": manifest.calibration,
            "training_data_hash": manifest.training_data_hash,
        }
        computed_sha = _compute_payload_hash(payload)
        if computed_sha != manifest.integrity.sha256:
            raise IntegrityError(
                f"Artifact integrity compromised! Expected {manifest.integrity.sha256} but computed {computed_sha}"
            )

    # 2. Model compatibility check
    if expected_model and manifest.model != expected_model:
        raise IncompatibleArtifactError(
            f"Model mismatch! Artifact was fitted for '{manifest.model}', but current model is '{expected_model}'."
        )

    if expected_revision and manifest.model_revision != expected_revision:
        raise IncompatibleArtifactError(
            f"Revision mismatch! Artifact was fitted for revision '{manifest.model_revision}', "
            f"but current model revision is '{expected_revision}'."
        )

    # 3. Instantiate calibrator
    cal_dict = manifest.calibration
    head_type = manifest.head

    if head_type == "temperature_scaling":
        calibrator: BaseCalibrator = TemperatureScaling.from_dict(cal_dict)
    elif head_type == "vector_scaling":
        calibrator = VectorScaling.from_dict(cal_dict)
    elif head_type == "isotonic_calibration":
        calibrator = IsotonicCalibration.from_dict(cal_dict)
    elif head_type == "platt_scaling":
        calibrator = PlattScaling.from_dict(cal_dict)
    else:
        raise ValueError(f"Unknown calibration head type: {head_type}")

    return calibrator, manifest
