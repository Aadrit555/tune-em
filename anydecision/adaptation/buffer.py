"""Observation buffer and train/eval leakage prevention."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, NamedTuple, Optional
import numpy as np


class LabeledObservation(NamedTuple):
    question_id: str
    probabilities: Dict[str, float]
    prediction: str
    label: str
    metadata: Dict[str, Any]


class ObservationBuffer:
    """Bounded observation buffer with strict dataset partition guards to prevent test leakage."""

    def __init__(self, max_capacity: int = 5000) -> None:
        self.max_capacity = max_capacity
        self.observations: List[LabeledObservation] = []
        self._forbidden_eval_ids: set[str] = set()

    def register_evaluation_split(self, question_ids: List[str]) -> None:
        """Register question IDs designated for evaluation/test to permanently forbid accidental leakage."""
        self._forbidden_eval_ids.update(question_ids)

    def add(
        self,
        question_id: str,
        probabilities: Dict[str, float],
        prediction: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an observation, enforcing split integrity."""
        if question_id in self._forbidden_eval_ids:
            raise ValueError(
                f"Data Leakage Violation! Question '{question_id}' belongs to the protected "
                f"evaluation/test partition and cannot be used for online adaptation."
            )

        obs = LabeledObservation(
            question_id=question_id,
            probabilities=probabilities,
            prediction=prediction,
            label=label,
            metadata=metadata or {},
        )
        self.observations.append(obs)
        if len(self.observations) > self.max_capacity:
            self.observations.pop(0)

    def clear(self) -> None:
        """Clear all stored adaptation observations."""
        self.observations.clear()

    def count(self) -> int:
        return len(self.observations)

    def compute_data_hash(self) -> str:
        """Compute SHA256 integrity hash of all buffered observations."""
        h = hashlib.sha256()
        for obs in self.observations:
            h.update(f"{obs.question_id}:{obs.label}".encode("utf-8"))
        return h.hexdigest()

