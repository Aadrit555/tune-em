"""Learned ViZDoom state estimator (Option B): behavior cloning + kill attribution.

Trains a compact MLP on REAL logged trajectories: sensor features ->
demonstrator action, with samples upweighted when a kill follows within
`kill_horizon` steps. Split by EPISODE (no same-episode leakage between train
and dev). Saved artifact records features, classes, scaler, weights, metrics,
and data provenance. Clearly labeled 'state estimator', evaluated separately.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


ATTACK_ACTIONS = {
    "PRECISION_ATTACK", "KITE_AND_FIRE", "CIRCLE_STRAFE_LEFT",
    "CIRCLE_STRAFE_RIGHT", "ASSAULT_ADVANCE",
}


def label_trajectories(
    records: Sequence[Dict[str, Any]],
    kill_horizon: int = 10,
) -> List[Dict[str, Any]]:
    """Attach clone labels and kill-attribution flags to step records.

    `kill_ahead`: a kill lands within the next `kill_horizon` steps of the
    same episode. `weight`: 1.0, or 3.0 for attack steps with kill_ahead.
    """
    recs = [dict(r) for r in records]
    by_ep: Dict[Any, List[int]] = {}
    for i, r in enumerate(recs):
        by_ep.setdefault(r.get("episode"), []).append(i)
    for ep, idxs in by_ep.items():
        kills = np.array([int(recs[i].get("kills_total", 0)) for i in idxs])
        future_kills = np.maximum.accumulate(kills[::-1])[::-1]
        for pos, i in enumerate(idxs):
            ahead = bool(future_kills[pos] > kills[pos]) if pos < len(idxs) else False
            # kill within horizon: any increase in the next H steps
            horizon_end = min(len(idxs), pos + 1 + kill_horizon)
            ahead = bool(np.any(kills[pos + 1:horizon_end] > kills[pos])) if horizon_end > pos + 1 else False
            recs[i]["kill_ahead"] = ahead
            recs[i]["weight"] = 3.0 if (recs[i].get("action") in ATTACK_ACTIONS and ahead) else 1.0
    return recs


def _run_key(r: Dict[str, Any]) -> tuple:
    """Independent-trajectory group key (episode ids restart per run)."""
    return (r.get("scenario"), r.get("policy"), r.get("seed"), r.get("episode"))


VISION_FEATURES = [
    "health", "armor", "ammo",
    "visible_hostiles", "crosshair_locked", "target_offset_x",
]


def train_estimator(
    records: Sequence[Dict[str, Any]],
    seed: int = 0,
    dev_episode_fraction: float = 0.25,
    hidden: Tuple[int, ...] = (32, 16),
    alpha: float = 1e-3,
    policies: Optional[Sequence[str]] = ("scripted",),
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train scaler + MLP on train runs; select/measure on held-out runs.

    Split key is (scenario, policy, seed, episode): episode ids restart every
    run, so splitting on episode alone would leak across runs.
    Sample weights combine kill-attribution boost with inverse class frequency.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from collections import Counter

    from anydecision.games.vizdoom_env import FEATURE_ORDER

    feat_names = list(feature_names) if feature_names else list(FEATURE_ORDER)
    feat_idx = [FEATURE_ORDER.index(n) for n in feat_names]
    recs = [r for r in label_trajectories(records)
            if policies is None or r.get("policy") in policies]
    if not recs:
        raise ValueError("no training records for the selected demonstrator policies")
    runs = sorted({_run_key(r) for r in recs})
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(runs))
    n_dev = max(1, int(round(dev_episode_fraction * len(runs))))
    dev_runs = {runs[i] for i in order[:n_dev].tolist()}
    counts = Counter(str(r["action"]) for r in recs)
    n_total = len(recs)
    n_cls = len(counts)
    Xtr, ytr, wtr, Xdv, ydv = [], [], [], [], []
    for r in recs:
        full = [float(v) for v in r["features"]]
        x = [full[i] for i in feat_idx]
        if _run_key(r) in dev_runs:
            Xdv.append(x)
            ydv.append(str(r["action"]))
        else:
            Xtr.append(x)
            ytr.append(str(r["action"]))
            wtr.append(float(r.get("weight", 1.0)) * n_total / (n_cls * counts[str(r["action"])]))
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xdv = np.asarray(Xdv, dtype=np.float64)
    scaler = StandardScaler()
    Xtrs = scaler.fit_transform(Xtr)
    Xdvs = scaler.transform(Xdv)
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
                        max_iter=800, early_stopping=True, n_iter_no_change=25,
                        random_state=seed)
    clf.fit(Xtrs, ytr, sample_weight=np.asarray(wtr))
    dev_acc = float(clf.score(Xdvs, ydv))
    train_acc = float(clf.score(Xtrs, ytr))
    classes = [str(c) for c in clf.classes_]
    info = {
        "train_n": len(ytr), "dev_n": len(ydv),
        "train_runs": sorted([list(k) for k in runs if k not in dev_runs]),
        "dev_runs": sorted([list(k) for k in dev_runs]),
        "classes": classes, "hidden": list(hidden), "seed": seed, "alpha": alpha,
        "train_accuracy": train_acc, "dev_accuracy": dev_acc,
        "n_features": Xtr.shape[1],
        "feature_names": feat_names,
        "policies": list(policies) if policies is not None else None,
    }
    return (scaler, clf), info


def save_estimator(path: str | Path, scaler: Any, clf: Any, info: Dict[str, Any],
                   provenance: Dict[str, Any]) -> Tuple[Path, Path]:
    """Persist scaler + MLP weights (.npz) and metadata sidecar (.json).

    Returns (weights_path, meta_path). `path` must end in .npz.
    """
    out = Path(path)
    if out.suffix != ".npz":
        raise ValueError(f"estimator artifact path must end in .npz, got {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "classes": np.array([str(c) for c in clf.classes_]),
        "n_features": np.array([int(clf.n_features_in_)]),
    }
    for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        payload[f"W{i}"] = np.asarray(w, dtype=np.float64)
        payload[f"b{i}"] = np.asarray(b, dtype=np.float64)
    np.savez_compressed(out, **payload)
    meta = dict(info)
    meta.update(provenance)
    meta["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["kind"] = "doom_state_estimator_mlp"
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out, meta_path


class DoomEstimator:
    """Inference wrapper for a saved state-estimator artifact (numpy only)."""

    def __init__(self, artifact: str | Path) -> None:
        z = np.load(str(artifact), allow_pickle=False)
        self.mean = np.asarray(z["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(z["scaler_scale"], dtype=np.float64)
        self.classes = [str(c) for c in z["classes"]]
        self.coefs: List[np.ndarray] = []
        self.intercepts: List[np.ndarray] = []
        i = 0
        while f"W{i}" in z:
            self.coefs.append(np.asarray(z[f"W{i}"], dtype=np.float64))
            self.intercepts.append(np.asarray(z[f"b{i}"], dtype=np.float64))
            i += 1
        meta_path = Path(str(artifact)).with_suffix(".json")
        self.meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        from anydecision.games.vizdoom_env import FEATURE_ORDER

        names = self.meta.get("feature_names", FEATURE_ORDER)
        self.feature_idx = [FEATURE_ORDER.index(n) for n in names]

    def predict_proba(self, features: Sequence[float]) -> Dict[str, float]:
        full = [float(v) for v in features]
        x = (np.asarray([full[i] for i in self.feature_idx], dtype=np.float64) - self.mean) / np.maximum(self.scale, 1e-12)
        for j, (w, b) in enumerate(zip(self.coefs, self.intercepts)):
            x = x @ w + b
            if j < len(self.coefs) - 1:
                x = np.maximum(x, 0.0)  # relu on hidden layers; logits stay linear
        x = x - x.max()
        e = np.exp(x)
        p = e / e.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}

    def predict(self, features: Sequence[float]) -> str:
        probs = self.predict_proba(features)
        return max(probs, key=probs.get)  # type: ignore[arg-type]
