"""vLLM high-throughput serving and inference backend."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from anydecision.backends.base import BaseBackend, ModelMetadata
from anydecision.scoring.candidate_tokenizer import (
    requires_sequence_scoring_for_candidates,
    tokenize_candidate_set,
)
from anydecision.scoring.sequence import (
    SequenceScoreResult,
    SequenceScorer,
    SequenceScoringMethod,
)


class VLLMBackend(BaseBackend):
    """High-throughput inference backend using vLLM's engine.

    Directly inspects prompt logprobs and token logprobs without open-ended generation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        try:
            from vllm import LLM, SamplingParams
            self._SamplingParams = SamplingParams
            self.llm = LLM(
                model=model_name_or_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                **kwargs,
            )
        except ImportError:
            raise ImportError(
                "vLLM is not installed. To use the vLLM backend, install it via: "
                "pip install 'anydecision[vllm]' or pip install vllm"
            )

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.model_name,
            model_revision="main",
            tokenizer_revision="main",
            backend_name="vllm",
            dtype="auto",
            device="cuda",
        )

    @property
    def token_counts_exact(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        """Exact token count using the vLLM-bundled tokenizer."""
        try:
            tokenizer = self.llm.get_tokenizer()
            ids = tokenizer.encode(text)
            return max(1, len(ids))
        except Exception:
            return max(1, len(text.split()))

    def requires_sequence_scoring(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> bool:
        """Check whether any candidate encodes to more than 1 token in the context of the prompt."""
        tokenizer = self.llm.get_tokenizer()
        return requires_sequence_scoring_for_candidates(tokenizer, prompt, candidate_strings)

    def next_token_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, float]:
        if self.requires_sequence_scoring(prompt, candidate_strings):
            return self.sequence_logprobs(prompt, candidate_strings)

        # Sampling params requesting top logprobs at first generated token
        sampling_params = self._SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=len(candidate_strings) * 10,
        )
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        first_token_logprobs = outputs[0].outputs[0].logprobs[0]

        candidate_scores: Dict[str, float] = {}
        for key, text in candidate_strings.items():
            clean_text = text.strip()
            matched_lp = -1e9
            for tid, lp_obj in first_token_logprobs.items():
                if lp_obj.decoded_token and lp_obj.decoded_token.strip() == clean_text:
                    matched_lp = max(matched_lp, float(lp_obj.logprob))
            candidate_scores[key] = matched_lp

        # If any candidate was outside top-k logprobs, fallback to robust sequence scoring
        if any(s <= -1e8 for s in candidate_scores.values()):
            return self.sequence_logprobs(prompt, candidate_strings)

        # Normalize
        keys = list(candidate_scores.keys())
        scores = np.array([candidate_scores[k] for k in keys], dtype=np.float64)
        max_s = np.max(scores)
        log_z = max_s + np.log(np.sum(np.exp(scores - max_s)))
        norm_scores = scores - log_z

        return {k: float(s) for k, s in zip(keys, norm_scores)}

    def sequence_logprobs_detailed(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        scoring_method: str = "length_normalized",
    ) -> Dict[str, Any]:
        """Sequence logprobs evaluating exact token likelihoods via prompt logprobs with detailed breakdown."""
        scorer = SequenceScorer(
            method=SequenceScoringMethod(scoring_method)
            if scoring_method in SequenceScoringMethod._value2member_map_
            else SequenceScoringMethod.LENGTH_NORMALIZED
        )

        tokenizer = self.llm.get_tokenizer()
        tokens_info = tokenize_candidate_set(tokenizer, prompt, candidate_strings)

        candidate_items = list(candidate_strings.items())
        full_prompts = [tokens_info[k].full_text for k, _ in candidate_items]

        sampling_params = self._SamplingParams(
            max_tokens=1,
            prompt_logprobs=20,
        )
        outputs = self.llm.generate(full_prompts, sampling_params, use_tqdm=False)

        score_breakdowns: Dict[str, SequenceScoreResult] = {}
        for (key, _), out in zip(candidate_items, outputs):
            info = tokens_info[key]
            p_logprobs = out.prompt_logprobs
            expected_ids = info.candidate_token_ids

            if not p_logprobs or not expected_ids:
                score_breakdowns[key] = scorer.score_sequence([])
                continue

            token_lps: List[float] = []
            for idx, expected_tid in enumerate(expected_ids):
                pos = info.prompt_token_count + idx
                if pos >= len(p_logprobs) or not p_logprobs[pos]:
                    token_lps.append(-20.0)
                    continue

                lp_dict = p_logprobs[pos]
                if expected_tid in lp_dict:
                    token_lps.append(float(lp_dict[expected_tid].logprob))
                else:
                    # Token outside top-k logprobs:
                    # By definition, P(token) <= min(P(k) in top-k).
                    # We bound this conservative estimate without inventing arbitrary offsets.
                    lowest_lp = min(float(lp.logprob) for lp in lp_dict.values()) if lp_dict else -20.0
                    token_lps.append(min(lowest_lp, -20.0))

            score_breakdowns[key] = scorer.score_sequence(token_lps)

        keys = list(score_breakdowns.keys())
        scores = np.array([score_breakdowns[k].score for k in keys], dtype=np.float64)
        max_s = np.max(scores)
        log_z = max_s + np.log(np.sum(np.exp(scores - max_s)))
        norm_scores = scores - log_z

        return {
            "conditional_logprobs": {k: float(s) for k, s in zip(keys, norm_scores)},
            "candidate_scores": score_breakdowns,
            "joint_logprobs": {k: score_breakdowns[k].joint_logprob for k in keys},
            "mean_logprobs": {k: score_breakdowns[k].mean_logprob for k in keys},
            "length_normalized_scores": {k: score_breakdowns[k].length_normalized_score for k in keys},
            "token_infos": tokens_info,
            "scoring_method": scorer.method.value,
        }

    def sequence_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        scoring_method: str = "length_normalized",
    ) -> Dict[str, float]:
        res = self.sequence_logprobs_detailed(prompt, candidate_strings, scoring_method)
        return res["conditional_logprobs"]
