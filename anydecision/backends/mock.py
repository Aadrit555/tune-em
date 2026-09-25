"""Mock / Synthetic backend for high-speed offline testing and CI/CD."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional
import numpy as np

from anydecision.backends.base import BaseBackend, ModelMetadata
from anydecision.scoring.normalization import softmax
from anydecision.scoring.sequence import SequenceScorer, SequenceScoringMethod


class MockBackend(BaseBackend):
    """Deterministic, high-speed mock backend for testing and benchmarks without GPU."""

    def __init__(
        self,
        model_name: str = "mock-qwen-2.5-7b",
        vocab_size: int = 32000,
        seed: int = 42,
        simulated_latency_ms: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.seed = seed
        self.simulated_latency_ms = simulated_latency_ms
        self._scorer = SequenceScorer(SequenceScoringMethod.LENGTH_NORMALIZED)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.model_name,
            model_revision="v1.0.0-mock",
            tokenizer_revision="v1.0.0-mock",
            hidden_size=4096,
            num_layers=32,
            vocab_size=self.vocab_size,
            dtype="bfloat16",
            device="cpu",
            backend_name="mock",
            quantization=None,
            context_window=8192,
        )

    def _hash_score(self, text: str, salt: str = "") -> float:
        """Generate deterministic pseudo-logit from text content."""
        content = f"{text}::{salt}::{self.seed}"
        digest = hashlib.sha256(content.encode("utf-8")).digest()
        # Convert first 4 bytes to float in range [-2.0, 5.0]
        val = int.from_bytes(digest[:4], "big") / (2**32 - 1)
        return float((val * 7.0) - 2.0)

    def _check_prefix_cache(self, prompt: str) -> None:
        """Simulate prefix caching hit/miss."""
        prefix = prompt[:50]
        if prefix in self._prefix_cache:
            self.cache_stats.cache_hits += 1
            self.cache_stats.tokens_reused += 20
            self.cache_stats.estimated_latency_saved_ms += 1.5
        else:
            self.cache_stats.cache_misses += 1
            self._prefix_cache[prefix] = True

    def next_token_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, float]:
        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        self._check_prefix_cache(prompt)

        # Generate deterministic logits for each candidate
        logits = {}
        for key, text in candidate_strings.items():
            # Add semantic bias if candidate text appears in prompt context
            bonus = 2.5 if text.lower() in prompt.lower() else 0.0
            logits[key] = self._hash_score(prompt + "->" + text) + bonus

        # Convert to log-probabilities via log-softmax
        keys = list(logits.keys())
        logit_arr = np.array([logits[k] for k in keys], dtype=np.float64)
        max_l = np.max(logit_arr)
        log_z = max_l + np.log(np.sum(np.exp(logit_arr - max_l)))
        log_probs = log_arr = logit_arr - log_z

        return {k: float(lp) for k, lp in zip(keys, log_probs)}

    def sequence_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        scoring_method: str = "length_normalized",
    ) -> Dict[str, float]:
        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        self._check_prefix_cache(prompt)
        scorer = SequenceScorer(
            method=SequenceScoringMethod(scoring_method)
            if scoring_method in SequenceScoringMethod._value2member_map_
            else SequenceScoringMethod.LENGTH_NORMALIZED
        )

        res = {}
        for key, text in candidate_strings.items():
            # Tokenize into whitespace words
            tokens = text.split()
            # Generate simulated per-token log probabilities
            token_lps = []
            for t_idx, tok in enumerate(tokens):
                score = self._hash_score(f"{prompt}>>{tok}_{t_idx}")
                # Convert to negative logprob [-0.05, -3.5]
                lp = -abs(score) - 0.2
                if tok.lower() in prompt.lower():
                    lp = max(-0.05, lp + 1.5)
                token_lps.append(lp)

            res[key] = scorer.score_tokens(token_lps)

        # Log-softmax over candidates
        keys = list(res.keys())
        scores = np.array([res[k] for k in keys], dtype=np.float64)
        max_s = np.max(scores)
        log_z = max_s + np.log(np.sum(np.exp(scores - max_s)))
        norm_log_scores = scores - log_z

        return {k: float(lp) for k, lp in zip(keys, norm_log_scores)}

    def batch_next_token_logprobs(
        self,
        prompts: List[str],
        candidate_strings_list: List[Dict[str, str]],
    ) -> List[Dict[str, float]]:
        return [
            self.next_token_logprobs(p, c)
            for p, c in zip(prompts, candidate_strings_list)
        ]

    def batch_sequence_logprobs(
        self,
        prompts: List[str],
        candidate_strings_list: List[Dict[str, str]],
        scoring_method: str = "length_normalized",
    ) -> List[Dict[str, float]]:
        return [
            self.sequence_logprobs(p, c, scoring_method=scoring_method)
            for p, c in zip(prompts, candidate_strings_list)
        ]

