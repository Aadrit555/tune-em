"""Versioned calibration artifact schemas and integrity specifications."""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from anydecision._version import __version__


class IntegrityBlock(BaseModel):
    sha256: str = Field(description="Cryptographic SHA-256 hash of the artifact payload.")


class ArtifactManifest(BaseModel):
    """Manifest describing a saved calibration head and its compatibility guarantees."""
    schema_version: int = Field(default=1, description="Artifact layout specification version.")
    anydecision_version: str = Field(default=__version__, description="Library version.")
    model: str = Field(description="Compatible model name or identifier.")
    model_revision: str = Field(default="main", description="Git commit hash or tag of the model.")
    tokenizer_revision: str = Field(default="main", description="Revision tag of the tokenizer.")
    question_schema: str = Field(default="generic", description="Question category or schema signature.")
    backend: str = Field(default="transformers", description="Inference backend used during fitting.")
    head: str = Field(description="Calibration head type (e.g. 'temperature_scaling', 'vector_scaling').")
    calibration: Dict[str, Any] = Field(description="Trained parameters of the calibrator.")
    training_data_hash: str = Field(description="SHA-256 hash of the training/calibration observations.")
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp."
    )
    integrity: IntegrityBlock = Field(description="SHA-256 checksum block.")

