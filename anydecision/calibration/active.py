"""Active calibration: high-information sample selection for minimal annotation cost.

Rather than randomly requesting 100-500 human labels, identifies the most informative
calibration candidates using:
- Epistemic & Shannon Entropy (Uncertainty)
- Decision Margin Proximity (|P(y_1) - P(y_2)|)
- Template & Layer Disagreement
- Out-of-Distribution (OOD) distance
- Hybrid information score
"""

from __future__ import annotations

from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

from anydecision.scoring.normalization import compute_entropy

if TYPE_CHECKING:
    from anydecision.core.engine import DecisionEngine
    from anydecision.core.question import Question
    from anydecision.evaluation.benchmark import BenchmarkSample


class ActiveSelectionCriterion(str, Enum):
    """Information-theoretic criterion for sample selection."""
    UNCERTAINTY = "uncertainty"
    MARGIN = "margin"
    DISAGREEMENT = "disagreement"
    OOD = "ood"
    HYBRID = "hybrid"


class CandidateSampleScore(BaseModel):
    """Information scoring breakdown for an unlabeled candidate."""
    question_id: str
    information_score: float
    entropy: float
    margin: float
    disagreement: float
    ood_score: float
    selected_rank: Optional[int] = None


class ActiveCalibrationReport(BaseModel):
    """Empirical comparison of active sampling vs random sampling at one label budget.

    ``relative_ece_change_pct`` is the relative ECE difference at this single
    budget point — it is NOT an annotation-cost saving. Label-budget savings
    can only be read off a budget curve (see estimate_label_savings()).
    """
    dataset_name: str
    pool_size: int
    budget: int
    test_size: int = 0
    random_ece: float = 0.0
    active_ece: float = 0.0
    random_accuracy: float = 0.0
    active_accuracy: float = 0.0
    random_nll: float = 0.0
    active_nll: float = 0.0
    random_brier: float = 0.0
    active_brier: float = 0.0
    random_selective_risk: float = 0.0
    active_selective_risk: float = 0.0
    relative_ece_change_pct: float = 0.0
    ece_ci95: List[float] = Field(default_factory=list)
    note: str = (
        "Single-budget ECE delta; not a label-cost saving. "
        "Use estimate_label_savings() over a budget curve for cost claims."
    )
    selected_sample_ids: List[str] = Field(default_factory=list)

    @property
    def calibration_cost_savings_pct(self) -> float:
        """Deprecated alias for relative_ece_change_pct (kept for compatibility)."""
        return self.relative_ece_change_pct

    def summary(self) -> str:
        lines = [
            f"=== Active Calibration Benchmark ({self.dataset_name}) ===",
            f"Candidate Pool Size:         {self.pool_size:,}",
            f"Human Annotation Budget:     {self.budget} samples (test n={self.test_size})",
            "",
            f"{'Method':<20} | {'Budget':<8} | {'ECE':<8} | {'NLL':<8} | {'Brier':<8} | {'Accuracy':<10}",
            "-" * 78,
            f"{'Random Sampling':<20} | {self.budget:<8} | {self.random_ece:>6.4f} | {self.random_nll:>6.4f} | {self.random_brier:>6.4f} | {self.random_accuracy*100:>8.2f}%",
            f"{'Active Calibration':<20} | {self.budget:<8} | {self.active_ece:>6.4f} | {self.active_nll:>6.4f} | {self.active_brier:>6.4f} | {self.active_accuracy*100:>8.2f}%",
            "-" * 78,
            f"Relative ECE change at this budget: {self.relative_ece_change_pct:+.1f}% (positive favors active).",
            "NOTE: this is NOT an annotation-cost saving. Cost savings require a",
            "budget curve via evaluate_budget_curve() + estimate_label_savings().",
            "======================================================================",
        ]
        return "\n".join(lines)


class ActiveCalibrator:
    """Selects highest-information samples from unlabeled pool to minimize labeling cost."""

    def __init__(self, criterion: ActiveSelectionCriterion = ActiveSelectionCriterion.HYBRID) -> None:
        self.criterion = criterion

    def score_question(
        self,
        question: Question,
        engine: DecisionEngine,
    ) -> CandidateSampleScore:
        """Compute multi-factor information score for an unlabeled question."""
        # Run cheap L0 decision to inspect distribution
        dec = engine.decide(question, level="L0")
        probs = list(dec.probabilities.values())
        sorted_probs = sorted(probs, reverse=True)

        # 1. Entropy (high = uncertain)
        entropy_val = compute_entropy(probs)

        # 2. Margin (small difference between top-1 and top-2 = high informativeness)
        p1 = sorted_probs[0] if len(sorted_probs) > 0 else 0.5
        p2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin_val = float(p1 - p2)
        margin_informational = 1.0 - margin_val  # high when near decision boundary

        # 3. Disagreement across permutations or templates
        dec_l1 = engine.decide(question, level="L1")
        disagreement_val = (
            1.0 - dec_l1.diagnostics.template_agreement
            if dec_l1.diagnostics
            else 0.0
        )

        # 4. OOD score
        ood_val = (
            dec.diagnostics.ood_score
            if (dec.diagnostics and dec.diagnostics.ood_score is not None)
            else 0.0
        )

        # Composite Hybrid Score
        if self.criterion == ActiveSelectionCriterion.UNCERTAINTY:
            total_score = entropy_val
        elif self.criterion == ActiveSelectionCriterion.MARGIN:
            total_score = margin_informational
        elif self.criterion == ActiveSelectionCriterion.DISAGREEMENT:
            total_score = disagreement_val
        elif self.criterion == ActiveSelectionCriterion.OOD:
            total_score = ood_val
        else:  # HYBRID
            total_score = (
                0.35 * entropy_val
                + 0.35 * margin_informational
                + 0.20 * disagreement_val
                + 0.10 * ood_val
            )

        return CandidateSampleScore(
            question_id=question.id,
            information_score=float(total_score),
            entropy=float(entropy_val),
            margin=float(margin_val),
            disagreement=float(disagreement_val),
            ood_score=float(ood_val),
        )

    def suggest_calibration_examples(
        self,
        pool: Sequence[Question],
        engine: DecisionEngine,
        budget: int = 25,
    ) -> List[Question]:
        """Rank and return the top `budget` most informative questions to label."""
        if not pool:
            return []
        if budget >= len(pool):
            return list(pool)

        scored: List[tuple[CandidateSampleScore, Question]] = []
        for q in pool:
            score = self.score_question(q, engine)
            scored.append((score, q))

        # Sort by information score descending
        scored.sort(key=lambda x: x[0].information_score, reverse=True)
        return [q for _, q in scored[:budget]]


class ActiveCalibrationBenchmark:
    """Simulates active learning vs random sampling calibration efficiency."""

    @staticmethod
    def run_benchmark(
        samples: Sequence[BenchmarkSample],
        engine_factory: Any,
        budget: int = 25,
        dataset_name: str = "active_calibration_benchmark",
        criterion: ActiveSelectionCriterion = ActiveSelectionCriterion.HYBRID,
    ) -> ActiveCalibrationReport:
        n = len(samples)
        if budget >= n:
            raise ValueError(f"Budget {budget} must be smaller than pool size {n}.")

        selector = ActiveCalibrator(criterion=criterion)
        eval_engine = engine_factory()

        # 1. Active Selection
        scored_pairs = []
        for s in samples:
            sc = selector.score_question(s.question, eval_engine)
            scored_pairs.append((sc.information_score, s))

        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        active_selected = [s for _, s in scored_pairs[:budget]]
        active_ids = [s.question.id for s in active_selected]

        # 2. Random Selection Baseline
        rng = np.random.RandomState(42)
        shuffled = list(samples).copy()
        rng.shuffle(shuffled)
        random_selected = shuffled[:budget]

        # Remaining test set (samples not in either active or random to test generalizability)
        held_out_active = [s for s in samples if s.question.id not in active_ids]
        test_samples = held_out_active if held_out_active else samples

        # Train Active Calibrator
        active_engine = engine_factory()
        from anydecision.calibration.temperature import TemperatureScaling
        active_ts = TemperatureScaling()
        # Collect probabilities on active batch
        active_probs = []
        active_labels = []
        keys = samples[0].question.option_keys()
        k_to_idx = {k: i for i, k in enumerate(keys)}

        for s in active_selected:
            if s.ground_truth not in k_to_idx:
                raise ValueError(
                    f"Sample ground truth {s.ground_truth!r} is not among option keys {keys!r}."
                )
            d = active_engine.decide(s.question, level="L0")
            row = [d.probabilities.get(k, 0.0) for k in keys]
            active_probs.append(row)
            active_labels.append(k_to_idx[s.ground_truth])

        active_ts.fit(np.array(active_probs), np.array(active_labels), keys)
        active_engine.calibrator = active_ts

        # Train Random Calibrator
        random_engine = engine_factory()
        random_ts = TemperatureScaling()
        random_probs = []
        random_labels = []
        for s in random_selected:
            if s.ground_truth not in k_to_idx:
                raise ValueError(
                    f"Sample ground truth {s.ground_truth!r} is not among option keys {keys!r}."
                )
            d = random_engine.decide(s.question, level="L0")
            row = [d.probabilities.get(k, 0.0) for k in keys]
            random_probs.append(row)
            random_labels.append(k_to_idx[s.ground_truth])

        random_ts.fit(np.array(random_probs), np.array(random_labels), keys)
        random_engine.calibrator = random_ts

        # Evaluate both on the held-out test set.
        from anydecision.calibration.metrics import (
            compute_brier_score,
            compute_ece,
            compute_nll,
        )

        def eval_on_test(eng: DecisionEngine) -> Dict[str, Any]:
            decs = [eng.decide(s.question, level="L2") for s in test_samples]
            confs = np.array([d.confidence for d in decs])
            correct = np.array([1 if d.answer == s.ground_truth else 0 for d, s in zip(decs, test_samples)])
            prob_rows = np.array([[d.probabilities.get(k, 0.0) for k in keys] for d in decs])
            label_idx = np.array([k_to_idx[s.ground_truth] for s in test_samples])
            # Selective risk at ~80% coverage (risk among the top-80% most confident).
            order = np.argsort(-confs)
            keep = max(1, int(round(0.8 * len(order))))
            sel = order[:keep]
            selective_risk = float(1.0 - np.mean(correct[sel])) if len(sel) else 1.0
            return {
                "accuracy": float(np.mean(correct)),
                "ece": compute_ece(confs, correct),
                "nll": compute_nll(prob_rows, label_idx),
                "brier": compute_brier_score(prob_rows, label_idx),
                "selective_risk": selective_risk,
                "confs": confs,
                "correct": correct,
            }

        act = eval_on_test(active_engine)
        rnd = eval_on_test(random_engine)

        rel_ece_change = ((rnd["ece"] - act["ece"]) / max(1e-6, rnd["ece"])) * 100.0

        # 95% bootstrap CI (fixed seed) for the ECE difference on the test set.
        ci95: List[float] = []
        if len(test_samples) >= 10:
            from anydecision.calibration.metrics import compute_ece as _ece
            rng = np.random.RandomState(12345)
            diffs = []
            for _ in range(200):
                idx = rng.randint(0, len(test_samples), len(test_samples))
                e_r = _ece(rnd["confs"][idx], rnd["correct"][idx])
                e_a = _ece(act["confs"][idx], act["correct"][idx])
                diffs.append(e_r - e_a)
            lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
            ci95 = [lo, hi]

        return ActiveCalibrationReport(
            dataset_name=dataset_name,
            pool_size=n,
            budget=budget,
            test_size=len(test_samples),
            random_ece=rnd["ece"],
            active_ece=act["ece"],
            random_accuracy=rnd["accuracy"],
            active_accuracy=act["accuracy"],
            random_nll=rnd["nll"],
            active_nll=act["nll"],
            random_brier=rnd["brier"],
            active_brier=act["brier"],
            random_selective_risk=rnd["selective_risk"],
            active_selective_risk=act["selective_risk"],
            relative_ece_change_pct=float(rel_ece_change),
            ece_ci95=ci95,
            selected_sample_ids=active_ids,
        )

    @staticmethod
    def estimate_label_savings(
        active_curve: Sequence[ActiveCalibrationReport],
        random_curve: Sequence[ActiveCalibrationReport],
    ) -> Dict[str, Any]:
        """Estimate annotation savings from budget curves (honest cost comparison).

        For each active budget point, finds the smallest random-sampling budget
        whose ECE is at or below the active ECE (linear interpolation between
        evaluated random budgets). Returns measured savings per point plus an
        explanation. Points where random never reaches the active ECE within
        the evaluated budgets are reported as not-demonstrated, never
        extrapolated.
        """
        rand_pts = sorted((r.budget, r.random_ece) for r in random_curve)
        out: List[Dict[str, Any]] = []
        for a in sorted(active_curve, key=lambda r: r.budget):
            target = a.active_ece
            matched: Optional[float] = None
            for (b0, e0), (b1, e1) in zip(rand_pts, rand_pts[1:]):
                if (e0 - target) * (e1 - target) <= 0 and e0 != e1:
                    frac = (e0 - target) / (e0 - e1)
                    matched = float(b0 + frac * (b1 - b0))
                    break
                if e1 <= target:
                    matched = float(b1)
                    break
            else:
                if rand_pts and rand_pts[0][1] <= target:
                    matched = float(rand_pts[0][0])
            if matched is None or matched <= 0:
                out.append({
                    "active_budget": a.budget, "active_ece": target,
                    "matched_random_budget": None, "estimated_savings_pct": None,
                    "demonstrated": False,
                    "explanation": "Random sampling did not reach this ECE within evaluated budgets.",
                })
            else:
                out.append({
                    "active_budget": a.budget, "active_ece": target,
                    "matched_random_budget": matched,
                    "estimated_savings_pct": float((1.0 - a.budget / matched) * 100.0),
                    "demonstrated": True,
                    "explanation": (
                        f"Random needs ~{matched:.1f} labels to match active ECE "
                        f"{target:.4f} at {a.budget} labels."
                    ),
                })
        return {"points": out}

    @staticmethod
    def evaluate_budget_curve(
        samples: Sequence[BenchmarkSample],
        engine_factory: Any,
        budgets: Sequence[int] = (10, 20, 40),
        dataset_name: str = "budget_curve",
        criterion: ActiveSelectionCriterion = ActiveSelectionCriterion.HYBRID,
    ) -> List[ActiveCalibrationReport]:
        """Evaluate calibration performance across multiple labeling budget points."""
        return [
            ActiveCalibrationBenchmark.run_benchmark(
                samples=samples,
                engine_factory=engine_factory,
                budget=b,
                dataset_name=dataset_name,
                criterion=criterion,
            )
            for b in budgets
            if b < len(samples)
        ]
