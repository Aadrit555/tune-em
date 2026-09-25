"""The core DecisionEngine runtime for direct probability-based typed decisions."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from anydecision.adaptation.online import OnlineAdapter
from anydecision.artifacts.saver import (
    load_calibration_artifact,
    save_calibration_artifact,
)
from anydecision.backends.base import BaseBackend, ModelMetadata
from anydecision.backends.registry import get_backend
from anydecision.bias.aggregation import aggregate_distributions
from anydecision.bias.permutation import (
    compute_permutation_invariance_metrics,
    generate_permutations,
)
from anydecision.bias.templates import DEFAULT_TEMPLATES, PromptTemplate
from anydecision.calibration.base import BaseCalibrator
from anydecision.calibration.conformal import ConformalPredictor
from anydecision.calibration.isotonic import IsotonicCalibration
from anydecision.calibration.platt import PlattScaling
from anydecision.calibration.temperature import TemperatureScaling
from anydecision.calibration.vector import VectorScaling
from anydecision.core.decision import Decision
from anydecision.core.policies import AbstentionPolicy, DecisionPolicy
from anydecision.core.question import Question
from anydecision.core.types import (
    AnswerType,
    DecisionLevel,
    DecisionTrace,
    Diagnostics,
    ReadoutStrategy,
)
from anydecision.scoring.normalization import compute_entropy, normalize_log_probabilities
from anydecision.uncertainty.abstention import AbstentionController
from anydecision.uncertainty.entropy import normalized_entropy
from anydecision.uncertainty.ood import OODDetector


class DecisionEngine:
    """The central decision engine for extracting typed, uncertainty-aware decisions from LLMs.

    Never generates free-form text. Directly evaluates candidate probability distributions,
    measures prompt and order bias, performs statistical calibration, and supports
    rigorous selective prediction and abstention.
    """

    def __init__(
        self,
        model: Union[str, BaseBackend] = "mock",
        default_level: Union[str, DecisionLevel] = DecisionLevel.L0,
        calibrator: Optional[BaseCalibrator] = None,
        conformal_predictor: Optional[ConformalPredictor] = None,
        default_policy: Optional[DecisionPolicy] = None,
        **backend_kwargs: Any,
    ) -> None:
        self.backend: BaseBackend = get_backend(model, **backend_kwargs)
        self.metadata: ModelMetadata = self.backend.get_metadata()
        self.default_level: DecisionLevel = (
            DecisionLevel(default_level)
            if isinstance(default_level, str)
            else default_level
        )
        self.calibrator: Optional[BaseCalibrator] = calibrator
        self.conformal_predictor: Optional[ConformalPredictor] = conformal_predictor
        self.policy: DecisionPolicy = default_policy or DecisionPolicy(level=self.default_level)
        self.abstention_controller = AbstentionController(self.policy.abstention)
        self.ood_detector = OODDetector()
        self.adapter = OnlineAdapter()

    def decide(
        self,
        question: Question,
        level: Optional[Union[str, DecisionLevel]] = None,
        min_confidence: Optional[float] = None,
        target_error: Optional[float] = None,
        allow_abstain: Optional[bool] = None,
        templates: Optional[Sequence[Union[str, PromptTemplate]]] = None,
        num_permutations: Optional[int] = None,
        trace: Optional[bool] = None,
        scoring_method: str = "length_normalized",
    ) -> Decision:
        """Execute a typed decision for a given question.

        Args:
            question: Typed Question definition with candidate options.
            level: 'L0' (raw), 'L1' (zero-label debiasing), 'L2' (calibrated).
            min_confidence: Threshold on top confidence below which engine abstains.
            target_error: Maximum acceptable risk / error rate before abstaining.
            allow_abstain: Flag allowing or forbidding selective abstention.
            templates: List of prompt templates for L1 ensemble.
            num_permutations: Number of option permutations for L1 debiasing.
            trace: If True, attach step-by-step DecisionTrace to output.
            scoring_method: Multi-token sequence scoring algorithm.

        Returns:
            Strongly typed Decision object.
        """
        start_time = time.perf_counter()
        target_level = (
            DecisionLevel(level)
            if isinstance(level, str)
            else (level or self.policy.level)
        )

        enable_trace = trace if trace is not None else self.policy.enable_trace
        decision_trace = (
            DecisionTrace(question_id=question.id) if enable_trace else None
        )

        if decision_trace:
            decision_trace.add_step(
                "question_received",
                "Received typed question for probability evaluation",
                question_text=question.text,
                options=question.option_keys(),
                readout_strategy=question.readout_strategy.value,
            )

        candidate_strings = {opt.key: opt.label for opt in question.options}
        is_multi_token = question.readout_strategy == ReadoutStrategy.MULTI_TOKEN_SEQUENCE

        # Build prompt templates
        template_objs: List[PromptTemplate] = []
        if templates:
            for t in templates:
                if isinstance(t, PromptTemplate):
                    template_objs.append(t)
                else:
                    template_objs.append(DEFAULT_TEMPLATES.get(str(t)))
        else:
            # Default templates based on level
            if target_level == DecisionLevel.L0:
                template_objs = [DEFAULT_TEMPLATES.get("minimal")]
            else:
                template_objs = [
                    DEFAULT_TEMPLATES.get(t_name) for t_name in self.policy.templates
                ]

        # Determine permutations
        if target_level == DecisionLevel.L0:
            orderings = [question.option_keys()]
        else:
            n_perms = num_permutations or self.policy.num_permutations
            orderings = generate_permutations(question.option_keys(), max_permutations=n_perms)

        # Execute evaluation runs across templates and permutations
        run_distributions: List[Dict[str, float]] = []
        backend_calls = 0

        for t_idx, tmpl in enumerate(template_objs):
            for p_idx, order in enumerate(orderings):
                prompt = tmpl.render(question, option_order=order)
                backend_calls += 1

                if is_multi_token:
                    # Multi-token sequence scoring
                    logprobs = self.backend.sequence_logprobs(
                        prompt, candidate_strings, scoring_method=scoring_method
                    )
                else:
                    # Single-token next-token logprobs
                    logprobs = self.backend.next_token_logprobs(prompt, candidate_strings)

                probs = normalize_log_probabilities(logprobs)
                run_distributions.append(probs)

                if decision_trace:
                    decision_trace.add_step(
                        "readout_forward_pass",
                        f"Evaluated template '{tmpl.name}' on permutation {p_idx + 1}",
                        prompt_preview=prompt[:120] + "...",
                        normalized_probs=probs,
                    )

        # Aggregate distributions across runs
        agg_method = self.policy.aggregation_method
        raw_probs = aggregate_distributions(run_distributions, method=agg_method)

        if decision_trace:
            decision_trace.add_step(
                "aggregation",
                f"Aggregated {len(run_distributions)} forward evaluations using '{agg_method}'",
                aggregated_probs=raw_probs,
            )

        # Invariance and bias metrics
        invariance_metrics = compute_permutation_invariance_metrics(
            run_distributions, question.option_keys()
        )

        # Calibrated probabilities (L2)
        calibrated_probs: Optional[Dict[str, float]] = None
        is_calibrated = False
        method_str = "raw" if target_level == DecisionLevel.L0 else "zero_label"

        if target_level == DecisionLevel.L2:
            if self.calibrator is not None:
                calibrated_probs = self.calibrator.calibrate_dict(raw_probs)
                is_calibrated = True
                method_str = getattr(self.calibrator, "type", "calibrated")
            elif self.adapter.calibrator.fitted:
                calibrated_probs = self.adapter.calibrate(raw_probs)
                is_calibrated = True
                method_str = "online_calibrated"
            else:
                # If L2 requested without fitted calibrator, fallback cleanly with notice
                calibrated_probs = dict(raw_probs)
                is_calibrated = False

        effective_probs = calibrated_probs if (is_calibrated and calibrated_probs) else raw_probs

        # Winning answer and confidence
        sorted_candidates = sorted(effective_probs.items(), key=lambda kv: kv[1], reverse=True)
        top_answer, top_confidence = sorted_candidates[0] if sorted_candidates else (None, 0.0)

        # Risk calculation (expected posterior error = 1.0 - confidence)
        posterior_risk = float(1.0 - top_confidence)
        uncertainty = float(1.0 - top_confidence)
        entropy_nats = compute_entropy(list(effective_probs.values()))

        # Evaluate Selective Prediction / Abstention
        active_abstention_policy = self.policy.abstention.model_copy()
        if min_confidence is not None:
            active_abstention_policy.min_confidence = min_confidence
        if target_error is not None:
            active_abstention_policy.target_error = target_error
        if allow_abstain is not None:
            active_abstention_policy.allow_abstain = allow_abstain

        should_abstain, abstain_reason = self.abstention_controller.evaluate(
            confidence=top_confidence,
            probabilities=effective_probs,
            calibrated_risk=posterior_risk,
            policy_override=active_abstention_policy,
        )

        final_answer = None if should_abstain else top_answer

        # Prediction set (conformal or selective)
        prediction_set: Optional[List[str]] = None
        if self.conformal_predictor is not None and self.conformal_predictor.fitted:
            k_list = question.option_keys()
            prob_arr = np.array([effective_probs[k] for k in k_list])
            prediction_set = self.conformal_predictor.predict_set(prob_arr, k_list)

        # Expected value for numeric score or ordinal questions
        expected_val: Optional[float] = None
        if question.answer_type in (AnswerType.NUMERIC_SCORE, AnswerType.ORDINAL):
            val_sum = 0.0
            has_val = False
            for opt in question.options:
                prob = effective_probs.get(opt.key, 0.0)
                if opt.numeric_value is not None:
                    val_sum += prob * opt.numeric_value
                    has_val = True
                elif opt.ordinal_rank is not None:
                    val_sum += prob * float(opt.ordinal_rank)
                    has_val = True
            if has_val:
                expected_val = float(val_sum)

        # OOD Diagnostics
        ood_result = self.ood_detector.diagnose(
            probabilities=effective_probs,
            permutation_distributions=run_distributions,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        diagnostics = Diagnostics(
            backend=self.metadata.backend_name,
            latency_ms=latency_ms,
            model=self.metadata.model_name,
            level=target_level.value,
            raw_probabilities=raw_probs,
            calibrated_probabilities=calibrated_probs,
            entropy=entropy_nats,
            risk=posterior_risk,
            abstention_threshold=active_abstention_policy.min_confidence,
            number_of_permutations=len(orderings),
            template_agreement=invariance_metrics.get("permutation_agreement", 1.0),
            option_order_sensitivity=invariance_metrics.get("position_bias_score", 0.0),
            cache_usage={
                "cache_hits": self.backend.cache_stats.cache_hits,
                "cache_misses": self.backend.cache_stats.cache_misses,
                "hit_rate": self.backend.cache_stats.hit_rate,
                "tokens_reused": self.backend.cache_stats.tokens_reused,
            },
            number_of_backend_calls=backend_calls,
            ood_score=ood_result.ood_score,
            ood_warning=ood_result.warning,
        )

        if decision_trace:
            decision_trace.add_step(
                "decision_finalized",
                "Constructed final typed decision",
                answer=final_answer,
                confidence=top_confidence,
                abstained=should_abstain,
                abstain_reason=abstain_reason,
            )

        return Decision(
            answer=final_answer,
            probabilities=effective_probs,
            confidence=top_confidence,
            uncertainty=uncertainty,
            level=target_level.value,
            method=method_str,
            calibrated=is_calibrated,
            abstained=should_abstain,
            reason=abstain_reason,
            risk=posterior_risk,
            prediction_set=prediction_set,
            expected_value=expected_val,
            diagnostics=diagnostics,
            trace=decision_trace,
        )

    def batch_decide(
        self,
        questions: Sequence[Question],
        level: Optional[Union[str, DecisionLevel]] = None,
        **kwargs: Any,
    ) -> List[Decision]:
        """Perform batched decisions across multiple questions with prefix sharing."""
        # Detect shared prefixes and evaluate efficiently
        results = []
        for q in questions:
            results.append(self.decide(q, level=level, **kwargs))
        return results

    async def async_decide(
        self,
        question: Question,
        level: Optional[Union[str, DecisionLevel]] = None,
        **kwargs: Any,
    ) -> Decision:
        """Asynchronous decision execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.decide(question, level=level, **kwargs))

    async def async_batch_decide(
        self,
        questions: Sequence[Question],
        level: Optional[Union[str, DecisionLevel]] = None,
        **kwargs: Any,
    ) -> List[Decision]:
        """Asynchronous batched decisions."""
        tasks = [self.async_decide(q, level=level, **kwargs) for q in questions]
        return await asyncio.gather(*tasks)

    def calibrate(
        self,
        calibration_dataset: List[Dict[str, Any]],
        method: str = "temperature",
        option_keys: Optional[List[str]] = None,
    ) -> BaseCalibrator:
        """Fit a post-hoc calibration head on labeled validation data.

        Args:
            calibration_dataset: List of dicts, each with {'question': Question, 'label': str}
                or {'probabilities': Dict[str, float], 'label': str}.
            method: 'temperature', 'vector', 'isotonic', 'platt', 'conformal'.
            option_keys: Optional canonical class order.

        Returns:
            Fitted BaseCalibrator instance.
        """
        # Collect probability matrix and integer labels
        raw_probs_list = []
        labels_list = []
        canonical_keys = option_keys

        for item in calibration_dataset:
            if "probabilities" in item:
                probs_dict = item["probabilities"]
            else:
                q = item["question"]
                dec = self.decide(q, level=DecisionLevel.L1)
                probs_dict = dec.probabilities

            if canonical_keys is None:
                canonical_keys = sorted(list(probs_dict.keys()))

            raw_probs_list.append([probs_dict.get(k, 0.0) for k in canonical_keys])
            lbl_str = str(item["label"])
            labels_list.append(canonical_keys.index(lbl_str) if lbl_str in canonical_keys else 0)

        prob_arr = np.array(raw_probs_list, dtype=np.float64)
        lbl_arr = np.array(labels_list, dtype=np.int64)

        if method == "temperature":
            calibrator: BaseCalibrator = TemperatureScaling()
            calibrator.fit(prob_arr, lbl_arr, option_keys=canonical_keys)
            self.calibrator = calibrator
        elif method == "vector":
            calibrator = VectorScaling()
            calibrator.fit(prob_arr, lbl_arr, option_keys=canonical_keys)
            self.calibrator = calibrator
        elif method == "isotonic":
            calibrator = IsotonicCalibration()
            calibrator.fit(prob_arr, lbl_arr, option_keys=canonical_keys)
            self.calibrator = calibrator
        elif method == "platt":
            calibrator = PlattScaling()
            calibrator.fit(prob_arr, lbl_arr, option_keys=canonical_keys)
            self.calibrator = calibrator
        elif method == "conformal":
            cp = ConformalPredictor(alpha=self.policy.conformal_alpha)
            cp.fit(prob_arr, lbl_arr)
            self.conformal_predictor = cp
            calibrator = TemperatureScaling()  # Keep default temperature for point probs
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        return calibrator

    def observe(
        self,
        question: Question,
        prediction: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Online adaptation hook for streaming labeled feedback."""
        # Execute fast L0 or L1 to obtain probability distribution
        dec = self.decide(question, level=DecisionLevel.L0)
        return self.adapter.observe(
            question_id=question.id,
            probabilities=dec.probabilities,
            prediction=prediction,
            label=label,
            metadata=metadata,
        )

    def save_calibration(
        self,
        filepath: str,
        question_schema: str = "generic",
    ) -> None:
        """Save active calibration head to a versioned, cryptographically verified artifact."""
        if self.calibrator is None:
            raise ValueError("Cannot save: No calibration head has been fitted.")
        save_calibration_artifact(
            calibrator=self.calibrator,
            filepath=filepath,
            model=self.metadata.model_name,
            model_revision=self.metadata.model_revision,
            tokenizer_revision=self.metadata.tokenizer_revision,
            backend=self.metadata.backend_name,
            question_schema=question_schema,
        )

    def load_calibration(
        self,
        filepath: str,
        verify_model: bool = True,
    ) -> None:
        """Load and validate a saved calibration artifact."""
        calibrator, _ = load_calibration_artifact(
            filepath=filepath,
            expected_model=self.metadata.model_name if verify_model else None,
            expected_revision=self.metadata.model_revision if verify_model else None,
        )
        self.calibrator = calibrator
