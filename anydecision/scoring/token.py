"""Single-token extraction and logprob mapping."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from anydecision.scoring.normalization import softmax


def extract_candidate_logprobs_from_logits(
    next_token_logits: np.ndarray,
    candidate_token_ids: Dict[str, int],
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Extract candidate log-probabilities from next-token vocabulary logits.

    Args:
        next_token_logits: Shape [vocab_size] logit array.
        candidate_token_ids: Mapping candidate_key -> single token_id.
        temperature: Temperature scaling factor.

    Returns:
        Mapping candidate_key -> unnormalized log-probability.
    """
    scaled = next_token_logits / max(temperature, 1e-8)
    # Log-softmax over full vocab
    max_l = np.max(scaled)
    log_z = max_l + np.log(np.sum(np.exp(scaled - max_l)))
    vocab_logprobs = scaled - log_z

    return {
        key: float(vocab_logprobs[token_id])
        for key, token_id in candidate_token_ids.items()
    }
