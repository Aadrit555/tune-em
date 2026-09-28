"""vLLM high-throughput serving and inference backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from anydecision.backends.base import BaseBackend, ModelMetadata
from anydecision.scoring.sequence import SequenceScorer, SequenceScoringMethod


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

    def next_token_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, float]:
        # Sampling params requesting top logprobs at first generated token
        sampling_params = self._SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=len(candidate_strings) * 10,
        )
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        first_token_logprobs = outputs[0].outputs[0].logprobs[0]

        candidate_scores = {}
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

    def sequence_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        scoring_method: str = "length_normalized",
    ) -> Dict[str, float]:
        """Sequence logprobs evaluating exact token likelihoods via prompt logprobs."""
        scorer = SequenceScorer(
            method=SequenceScoringMethod(scoring_method)
            if scoring_method in SequenceScoringMethod._value2member_map_
            else SequenceScoringMethod.LENGTH_NORMALIZED
        )

        tokenizer = self.llm.get_tokenizer()
        prompt_tokens_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        candidate_items = list(candidate_strings.items())
        full_prompts = []
        expected_token_ids_list = []

        for key, text in candidate_items:
            full_text = prompt.rstrip() + " " + text.lstrip()
            full_prompts.append(full_text)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            expected_token_ids_list.append(full_ids[prompt_tokens_len:])

        sampling_params = self._SamplingParams(
            max_tokens=1,
            prompt_logprobs=15,
        )
        outputs = self.llm.generate(full_prompts, sampling_params, use_tqdm=False)

        candidate_scores = {}
        for (key, _), expected_ids, out in zip(candidate_items, expected_token_ids_list, outputs):
            p_logprobs = out.prompt_logprobs
            if not p_logprobs or not expected_ids:
                candidate_scores[key] = -1e9
                continue

            token_lps = []
            for idx, expected_tid in enumerate(expected_ids):
                pos = prompt_tokens_len + idx
                if pos >= len(p_logprobs) or not p_logprobs[pos]:
                    token_lps.append(-15.0)
                    continue

                lp_dict = p_logprobs[pos]
                if expected_tid in lp_dict:
                    token_lps.append(float(lp_dict[expected_tid].logprob))
                else:
                    # Token outside top-k logprobs, bounded by lowest returned logprob
                    lowest_lp = min(float(lp.logprob) for lp in lp_dict.values())
                    token_lps.append(lowest_lp - 1.0)

            candidate_scores[key] = scorer.score_tokens(token_lps)

        # Normalize
        keys = list(candidate_scores.keys())
        scores = np.array([candidate_scores[k] for k in keys], dtype=np.float64)
        max_s = np.max(scores)
        log_z = max_s + np.log(np.sum(np.exp(scores - max_s)))
        norm_scores = scores - log_z

        return {k: float(s) for k, s in zip(keys, norm_scores)}

