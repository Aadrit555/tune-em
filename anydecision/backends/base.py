"""Abstract backend interface and model metadata specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Observable architecture and deployment metadata for the underlying model."""
    model_name: str
    model_revision: str = "main"
    tokenizer_revision: str = "main"
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    vocab_size: Optional[int] = None
    dtype: str = "float32"
    device: str = "cpu"
    backend_name: str = "base"
    quantization: Optional[str] = None
    context_window: int = 4096

    def fingerprint(self) -> str:
        """Construct fingerprint string for artifact and cache validation."""
        return (
            f"{self.model_name}:{self.model_revision}:"
            f"{self.vocab_size}:{self.hidden_size}:{self.num_layers}"
        )


class PrefixCacheStats(BaseModel):
    """Statistics for prefix and KV reuse."""
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_reused: int = 0
    estimated_latency_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return float(self.cache_hits / total) if total > 0 else 0.0


class BaseBackend(ABC):
    """Abstract interface defining required probability readout operations.

    NEVER generates free-form text. All methods extract direct log-probabilities
    or logits from the model's output distribution.
    """

    def __init__(self) -> None:
        self.cache_stats = PrefixCacheStats()
        self._prefix_cache: Dict[str, Any] = {}

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return standardized metadata regarding the loaded model and backend."""
        pass

    @abstractmethod
    def next_token_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, float]:
        """Query direct next-token log-probabilities for candidate single-token strings.

        Args:
            prompt: Context or query text.
            candidate_strings: Dict of {key: verbatim_token_string}.

        Returns:
            Dict of {key: log_probability}.
        """
        pass

    @abstractmethod
    def sequence_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        scoring_method: str = "length_normalized",
    ) -> Dict[str, float]:
        """Query multi-token sequence log-probabilities for arbitrary-length candidate answers.

        Args:
            prompt: Preceding prompt text.
            candidate_strings: Dict of {key: verbatim_string} (e.g. {'tech': 'technical support'}).
            scoring_method: 'sum', 'mean', 'length_normalized', or 'conditional'.

        Returns:
            Dict of {key: sequence_log_score}.
        """
        pass

    def batch_next_token_logprobs(
        self,
        prompts: List[str],
        candidate_strings_list: List[Dict[str, str]],
    ) -> List[Dict[str, float]]:
        """Vectorized / batched next-token logprob queries."""
        # Default fallback if backend does not optimize batching natively
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
        """Vectorized / batched multi-token sequence queries."""
        return [
            self.sequence_logprobs(p, c, scoring_method=scoring_method)
            for p, c in zip(prompts, candidate_strings_list)
        ]

    async def async_next_token_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, float]:
        """Async variant for web servers and concurrent agent workflows."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.next_token_logprobs, prompt, candidate_strings)

    async def async_sequence_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        scoring_method: str = "length_normalized",
    ) -> Dict[str, float]:
        """Async variant for multi-token sequences."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.sequence_logprobs, prompt, candidate_strings, scoring_method
        )
