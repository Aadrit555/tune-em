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
        log_probs = logit_arr - log_z

        self._last_raw_vocab_lps = {
            k: float(-abs(self._hash_score(prompt, f"vocab::{k}")) - 1.0)
            for k in candidate_strings
        }

        return {k: float(lp) for k, lp in zip(keys, log_probs)}

    @property
    def token_counts_exact(self) -> bool:
        return False

    def count_tokens(self, text: str) -> int:
        """Whitespace estimate only. Always label results as estimated."""
        return max(1, len(text.split()))

    def requires_sequence_scoring(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> bool:
        """Mock check: multi-token if contains space, symbols (+, -, #, /), or longer word."""
        for text in candidate_strings.values():
            clean = text.strip()
            if " " in clean or any(c in clean for c in "+-#/._") or len(clean) > 8:
                return True
        return False

    def next_token_logprobs_detailed(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, Any]:
        cond = self.next_token_logprobs(prompt, candidate_strings)
        raw = self._last_raw_vocab_lps
        return {
            "conditional_logprobs": cond,
            "raw_vocab_logprobs": raw,
        }

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

        res = self.sequence_logprobs_detailed(prompt, candidate_strings, scoring_method)
        return res["conditional_logprobs"]

    def sequence_logprobs_detailed(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        scoring_method: str = "length_normalized",
    ) -> Dict[str, Any]:
        import re
        from anydecision.scoring.sequence import SequenceScoreResult
        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        self._check_prefix_cache(prompt)
        scorer = SequenceScorer(
            method=SequenceScoringMethod(scoring_method)
            if scoring_method in SequenceScoringMethod._value2member_map_
            else SequenceScoringMethod.LENGTH_NORMALIZED
        )

        score_breakdowns: Dict[str, SequenceScoreResult] = {}
        for key, text in candidate_strings.items():
            # Sub-tokenization splitting words and punctuation
            tokens = re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)
            if not tokens:
                score_breakdowns[key] = scorer.score_sequence([])
                continue

            token_lps = []
            for t_idx, tok in enumerate(tokens):
                score = self._hash_score(f"{prompt}>>{tok}_{t_idx}")
                lp = -abs(score) - 0.2
                if tok.lower() in prompt.lower():
                    lp = max(-0.05, lp + 1.5)
                token_lps.append(lp)

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
            "scoring_method": scorer.method.value,
        }

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

    def get_layer_hidden_states(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        target_layers = layers or [4, 8, 12, 16, 20, 24, 28, 32]
        # Seed deterministic base vector from prompt
        h = int(hashlib.sha256(f"{prompt}::base_rep".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(h % (2**31 - 1))
        dim = 128
        base_vec = rng.randn(dim).astype(np.float32)
        base_vec /= (np.linalg.norm(base_vec) + 1e-8)

        layer_states = {}
        for l in target_layers:
            # Noise decreases as layer depth increases
            noise_scale = 1.0 / np.sqrt(max(1, l))
            noise = rng.randn(dim).astype(np.float32) * noise_scale
            rep = base_vec + noise
            rep /= (np.linalg.norm(rep) + 1e-8)
            layer_states[l] = rep

        return layer_states

    def get_layer_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        target_layers = layers or [4, 8, 12, 16, 20, 24, 28, 32]
        final_logprobs = self.next_token_logprobs(prompt, candidate_strings)
        keys = list(candidate_strings.keys())
        final_arr = np.array([final_logprobs[k] for k in keys], dtype=np.float64)

        layer_results = {}
        for l in sorted(target_layers):
            # Scale temperature and noise based on depth: early layers are high entropy, later layers sharp
            alpha = min(1.0, float(l) / 28.0)  # reaches full emergence around layer 24-28
            # Uniform prior mixture for early layers
            uniform_logits = np.zeros_like(final_arr)
            interpolated = (1.0 - alpha) * uniform_logits + alpha * final_arr
            # Add slight perturbation
            h_layer = int(hashlib.sha256(f"{prompt}::layer_{l}".encode("utf-8")).hexdigest()[:6], 16)
            layer_rng = np.random.RandomState(h_layer % (2**31 - 1))
            jitter = layer_rng.randn(len(keys)) * (0.3 * (1.0 - alpha))
            noisy_logits = interpolated + jitter

            # Softmax to log-probs
            max_val = np.max(noisy_logits)
            lse = max_val + np.log(np.sum(np.exp(noisy_logits - max_val)))
            lps = noisy_logits - lse
            layer_results[l] = {k: float(lp) for k, lp in zip(keys, lps)}

        return layer_results

