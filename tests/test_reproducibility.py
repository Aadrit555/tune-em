"""Unit tests for reproducibility manifest and deterministic output."""

import pytest
from anydecision.backends.mock import MockBackend
from anydecision.utils.manifest import ReproducibilityManifest


def test_reproducibility_manifest():
    backend = MockBackend()
    meta = backend.get_metadata()
    manifest = ReproducibilityManifest.generate(
        model_metadata=meta,
        templates=["minimal", "structured"],
        seed=42,
    )
    assert manifest.run_hash != ""
    assert manifest.random_seed == 42
    assert manifest.platform != ""
    assert manifest.python_version != ""
