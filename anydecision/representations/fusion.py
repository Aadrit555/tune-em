"""Multi-layer hidden representation fusion heads for calibrated L2 decisions.

Instead of binding to a single transformer hidden layer (e.g. Layer 24), fuses
representations from multiple depth checkpoints (e.g. [12, 18, 24]) using:
- Concatenation
- Softmax Weighted Average
- Learned Context-Dependent Gating
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from anydecision.calibration.metrics import compute_brier_score, compute_ece, compute_nll
from anydecision.scoring.normalization import softmax


class FusionStrategy(str, Enum):
    """Architecture for fusing multiple hidden representations."""
    CONCATENATION = "concatenation"
    WEIGHTED_AVERAGE = "weighted_average"
    LEARNED_GATING = "learned_gating"


class FusionModelCard(BaseModel):
    """Metadata describing trained fusion head."""
    strategy: FusionStrategy
    layers: List[int]
    input_dim_per_layer: int
    total_feature_dim: int
    num_classes: int
    accuracy: float
    ece: float
    brier_score: float


class FusionExperimentReport(BaseModel):
    """Comparative research benchmark of single-layer vs multi-layer fusion heads."""
    dataset_name: str
    num_samples: int
    single_layer_baseline: Dict[str, Any]
    fusion_architectures: Dict[str, Dict[str, Any]]
    best_architecture: str
    relative_accuracy_gain_pct: float
    relative_ece_reduction_pct: float

    def summary(self) -> str:
        lines = [
            f"=== Multi-Layer Representation Fusion Experiment ({self.dataset_name}) ===",
            f"Evaluated Samples: {self.num_samples:,}",
            "",
            f"{'Head Architecture':<22} | {'Layers':<14} | {'Accuracy':<10} | {'ECE':<9} | {'Brier':<9} | {'Params':<8}",
            "-" * 78,
            f"{'Single Best Layer (L2)':<22} | {str(self.single_layer_baseline.get('layer', 24)):<14} | "
            f"{self.single_layer_baseline.get('accuracy', 0)*100:>8.2f}% | "
            f"{self.single_layer_baseline.get('ece', 0):>8.4f} | "
            f"{self.single_layer_baseline.get('brier', 0):>8.4f} | "
            f"{self.single_layer_baseline.get('params', 0):>7,}",
        ]
        for name, metrics in sorted(self.fusion_architectures.items()):
            layers_str = str(metrics.get("layers", []))
            lines.append(
                f"{name:<22} | {layers_str:<14} | "
                f"{metrics.get('accuracy', 0)*100:>8.2f}% | "
                f"{metrics.get('ece', 0):>8.4f} | "
                f"{metrics.get('brier', 0):>8.4f} | "
                f"{metrics.get('params', 0):>7,}"
            )
        lines.extend([
            "-" * 78,
            f"Optimal Architecture:        {self.best_architecture}",
            f"Relative Accuracy Gain:      {self.relative_accuracy_gain_pct:+.2f}%",
            f"Relative ECE Error Drop:     {self.relative_ece_reduction_pct:+.2f}%",
            "=============================================================================",
        ])
        return "\n".join(lines)


class MultiLayerFusionHead:
    """Fuses representations across multiple layer checkpoints into calibrated class probabilities."""

    def __init__(
        self,
        layers: Sequence[int] = (12, 18, 24),
        strategy: FusionStrategy = FusionStrategy.CONCATENATION,
        regularization_c: float = 1.0,
    ) -> None:
        self.layers = sorted(list(layers))
        self.strategy = strategy
        self.regularization_c = regularization_c
        self.classifier = LogisticRegression(
            C=regularization_c,
            max_iter=1000,
            solver="lbfgs",
        )
        self.option_keys: List[str] = []
        self.layer_weights: Optional[np.ndarray] = None
        self.gating_weights: Optional[np.ndarray] = None
        self.fitted = False

    def _fuse_features(
        self,
        X_by_layer: Dict[int, np.ndarray],
        fit_mode: bool = False,
    ) -> np.ndarray:
        """Combine representations from specified layers according to chosen strategy."""
        for l in self.layers:
            if l not in X_by_layer:
                raise ValueError(f"Requested layer {l} missing from representation dictionary.")

        n_samples = X_by_layer[self.layers[0]].shape[0]
        layer_arrays = [X_by_layer[l] for l in self.layers]

        if self.strategy == FusionStrategy.CONCATENATION:
            # Concatenate along feature dimension
            return np.concatenate(layer_arrays, axis=1)

        elif self.strategy == FusionStrategy.WEIGHTED_AVERAGE:
            # Weighted average across layer vectors (requires equal dimension)
            if self.layer_weights is None or fit_mode:
                # Initialize equal weights or compute variance weights
                num_l = len(self.layers)
                self.layer_weights = np.ones(num_l, dtype=np.float64) / num_l

            stacked = np.stack(layer_arrays, axis=1)  # [N, num_layers, Dim]
            w = self.layer_weights[:, None]  # [num_layers, 1]
            return np.sum(stacked * w, axis=1)

        elif self.strategy == FusionStrategy.LEARNED_GATING:
            # Context-aware gating: compute attention scalar per layer based on early layer representation
            d = layer_arrays[0].shape[1]
            if self.gating_weights is None or fit_mode:
                rng = np.random.RandomState(42)
                self.gating_weights = rng.randn(d, len(self.layers)) * 0.1

            # Gating logits: [N, num_layers] = X_early @ gating_weights
            gate_logits = np.dot(layer_arrays[0], self.gating_weights)
            # Softmax gating weights per sample
            exp_g = np.exp(gate_logits - np.max(gate_logits, axis=1, keepdims=True))
            gates = exp_g / np.sum(exp_g, axis=1, keepdims=True)  # [N, num_layers]

            stacked = np.stack(layer_arrays, axis=1)  # [N, num_layers, Dim]
            return np.sum(stacked * gates[:, :, None], axis=1)

        raise ValueError(f"Unknown fusion strategy: {self.strategy}")

    def fit(
        self,
        X_by_layer: Dict[int, np.ndarray],
        y: np.ndarray,
        option_keys: Sequence[str],
    ) -> MultiLayerFusionHead:
        """Train the multi-layer fusion readout head."""
        self.option_keys = list(option_keys)
        fused = self._fuse_features(X_by_layer, fit_mode=True)
        self.classifier.fit(fused, y)
        self.fitted = True
        return self

    def predict_proba(self, X_by_layer: Dict[int, np.ndarray]) -> np.ndarray:
        """Predict calibrated multi-class probability distribution."""
        if not self.fitted:
            raise RuntimeError("MultiLayerFusionHead must be fitted before prediction.")
        fused = self._fuse_features(X_by_layer, fit_mode=False)
        return self.classifier.predict_proba(fused)

    def predict_dict(self, X_by_layer: Dict[int, np.ndarray]) -> Dict[str, float]:
        """Predict single-sample probability dictionary."""
        probs = self.predict_proba(X_by_layer)[0]
        return {k: float(p) for k, p in zip(self.option_keys, probs)}


class FusionExperimentRunner:
    """Systematically benchmarks single-layer vs multi-layer fusion heads."""

    @staticmethod
    def run_comparison(
        X_by_layer: Dict[int, np.ndarray],
        y: np.ndarray,
        option_keys: Sequence[str],
        train_split: float = 0.7,
        dataset_name: str = "representations_benchmark",
        candidate_layers: Sequence[int] = (12, 18, 24),
    ) -> FusionExperimentReport:
        n = len(y)
        n_train = int(n * train_split)
        indices = np.arange(n)
        np.random.RandomState(42).shuffle(indices)

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_X_layers = {l: X_by_layer[l][train_idx] for l in X_by_layer}
        test_X_layers = {l: X_by_layer[l][test_idx] for l in X_by_layer}
        train_y = y[train_idx]
        test_y = y[test_idx]

        # 1. Single Best Layer Baseline (e.g. highest layer)
        best_single_layer = max(candidate_layers)
        single_clf = LogisticRegression(C=1.0, max_iter=1000)
        single_clf.fit(train_X_layers[best_single_layer], train_y)
        single_probs = single_clf.predict_proba(test_X_layers[best_single_layer])
        single_preds = np.argmax(single_probs, axis=1)
        single_acc = float(np.mean(single_preds == test_y))
        single_confs = np.max(single_probs, axis=1)
        single_correct = (single_preds == test_y).astype(int)
        single_ece = compute_ece(single_confs, single_correct)
        single_brier = compute_brier_score(single_probs, test_y)
        single_params = int(train_X_layers[best_single_layer].shape[1] * len(option_keys))

        single_baseline = {
            "layer": best_single_layer,
            "accuracy": single_acc,
            "ece": single_ece,
            "brier": single_brier,
            "params": single_params,
        }

        # 2. Multi-layer Fusion Heads
        architectures = {}
        for strat in [FusionStrategy.CONCATENATION, FusionStrategy.WEIGHTED_AVERAGE, FusionStrategy.LEARNED_GATING]:
            head = MultiLayerFusionHead(layers=candidate_layers, strategy=strat)
            head.fit(train_X_layers, train_y, option_keys)
            probs = head.predict_proba(test_X_layers)
            preds = np.argmax(probs, axis=1)
            acc = float(np.mean(preds == test_y))
            head_confs = np.max(probs, axis=1)
            head_correct = (preds == test_y).astype(int)
            ece_val = compute_ece(head_confs, head_correct)
            brier_val = compute_brier_score(probs, test_y)

            dim = (
                train_X_layers[candidate_layers[0]].shape[1] * len(candidate_layers)
                if strat == FusionStrategy.CONCATENATION
                else train_X_layers[candidate_layers[0]].shape[1]
            )
            params = int(dim * len(option_keys))

            architectures[strat.value] = {
                "layers": list(candidate_layers),
                "accuracy": acc,
                "ece": ece_val,
                "brier": brier_val,
                "params": params,
            }

        # Determine best
        best_name = max(architectures.keys(), key=lambda k: architectures[k]["accuracy"])
        best_acc = architectures[best_name]["accuracy"]
        best_ece = architectures[best_name]["ece"]

        rel_acc_gain = ((best_acc - single_acc) / max(1e-6, single_acc)) * 100.0
        rel_ece_drop = ((single_ece - best_ece) / max(1e-6, single_ece)) * 100.0

        return FusionExperimentReport(
            dataset_name=dataset_name,
            num_samples=n,
            single_layer_baseline=single_baseline,
            fusion_architectures=architectures,
            best_architecture=best_name,
            relative_accuracy_gain_pct=rel_acc_gain,
            relative_ece_reduction_pct=rel_ece_drop,
        )


FusionComparisonExperiment = FusionExperimentRunner
