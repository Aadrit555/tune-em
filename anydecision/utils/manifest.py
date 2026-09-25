"""Reproducibility manifest generator and system compatibility checks."""

from __future__ import annotations

import datetime
import hashlib
import platform
import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from anydecision._version import __version__
from anydecision.backends.base import ModelMetadata


class ReproducibilityManifest(BaseModel):
    """Manifest recording all factors influencing decision generation."""
    anydecision_version: str = Field(default=__version__)
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    random_seed: int = 42
    model_metadata: ModelMetadata
    prompt_templates: List[str] = Field(default_factory=list)
    calibration_config: Dict[str, Any] = Field(default_factory=dict)
    run_hash: str = Field(default="")

    @classmethod
    def generate(
        cls,
        model_metadata: ModelMetadata,
        templates: List[str],
        seed: int = 42,
        calibration_config: Optional[Dict[str, Any]] = None,
    ) -> ReproducibilityManifest:
        config = calibration_config or {}
        raw_str = f"{__version__}:{model_metadata.fingerprint()}:{seed}:{','.join(templates)}"
        run_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

        return cls(
            model_metadata=model_metadata,
            prompt_templates=templates,
            random_seed=seed,
            calibration_config=config,
            run_hash=run_hash,
        )
