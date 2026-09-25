"""Online lightweight adaptation and incremental calibration updates."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import numpy as np

from anydecision.adaptation.buffer import ObservationBuffer
from anydecision.calibration.temperature import TemperatureScaling
from anydecision.calibration.vector import VectorScaling


class OnlineAdapter:
    """Manages lightweight online adaptation without fine-tuning LLM weights."""

    def __init__(
        self,
        refit_interval: int = 10,
        adaptation_method: str = "temperature",
        buffer_capacity: int = 2000,
    ) -> None:
        self.refit_interval = refit_interval
        self.adaptation_method = adaptation_method
        self.buffer = ObservationBuffer(max_capacity=buffer_capacity)
        self.calibrator = (
            TemperatureScaling()
            if adaptation_method == "temperature"
            else VectorScaling()
        )
        self.updates_count = 0
        self.is_active = True

    def observe(
        self,
        question_id: str,
        probabilities: Dict[str, float],
        prediction: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record labeled feedback and trigger incremental refit if threshold reached."""
        if not self.is_active:
            return False

        self.buffer.add(
            question_id=question_id,
            probabilities=probabilities,
            prediction=prediction,
            label=label,
            metadata=metadata,
        )

        if self.buffer.count() >= 5 and (self.buffer.count() % self.refit_interval == 0):
            self._refit()
            self.updates_count += 1
            return True
        return False

    def _refit(self) -> None:
        """Refit lightweight calibration head on accumulated online observations."""
        observations = self.buffer.observations
        if not observations:
            return

        # Build matrix
        keys = sorted(list(observations[0].probabilities.keys()))
        key_to_idx = {k: i for i, k in enumerate(keys)}

        n = len(observations)
        k = len(keys)
        prob_matrix = np.zeros((n, k), dtype=np.float64)
        labels = []

        for i, obs in enumerate(observations):
            for opt, p in obs.probabilities.items():
                if opt in key_to_idx:
                    prob_matrix[i, key_to_idx[opt]] = p
            lbl_idx = key_to_idx.get(obs.label, 0)
            labels.append(lbl_idx)

        # Normalize matrix rows
        row_sums = np.sum(prob_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        prob_matrix = prob_matrix / row_sums

        self.calibrator.fit(prob_matrix, np.array(labels, dtype=np.int64), option_keys=keys)

    def calibrate(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Calibrate a probability dictionary using the online adapted head."""
        if not self.calibrator.fitted:
            return probabilities
        return self.calibrator.calibrate_dict(probabilities)

    def reset(self) -> None:
        """Reset online adaptation buffer and revert calibrator."""
        self.buffer.clear()
        self.calibrator = (
            TemperatureScaling()
            if self.adaptation_method == "temperature"
            else VectorScaling()
        )
        self.updates_count = 0

    def state_dict(self) -> Dict[str, Any]:
        """Export serialized adaptation state."""
        return {
            "adaptation_method": self.adaptation_method,
            "refit_interval": self.refit_interval,
            "updates_count": self.updates_count,
            "buffer_count": self.buffer.count(),
            "data_hash": self.buffer.compute_data_hash(),
            "calibrator": self.calibrator.to_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore online adaptation state."""
        self.adaptation_method = state.get("adaptation_method", "temperature")
        self.refit_interval = state.get("refit_interval", 10)
        self.updates_count = state.get("updates_count", 0)
        cal_data = state.get("calibrator", {})
        if cal_data.get("type") == "temperature_scaling":
            self.calibrator = TemperatureScaling.from_dict(cal_data)
        elif cal_data.get("type") == "vector_scaling":
            self.calibrator = VectorScaling.from_dict(cal_data)


def create_calibration_splits(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Reproducibly split dataset into disjoint partitions: train/calibration, validation, test."""
    import random
    rng = random.Random(seed)
    shuffled = dataset.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train_calibration": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

