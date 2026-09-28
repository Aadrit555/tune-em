"""Shared tokenizer-aware candidate analysis and token alignment utility.

Handles:
- Single-token vs multi-token candidate distinction without string heuristics
- Prompt-candidate boundary whitespace merging (GPT-2, LLaMA, Mistral, Qwen, Gemma)
- Tokenizers that prepend BOS or alter boundary tokens
- Empty candidates, whitespace-sensitive candidates, punctuation, and Unicode
- Prefix token alignment verification
"""

from __future__ import annotations

import string
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, Field


class CandidateTokenInfo(BaseModel):
    """Structured tokenization and alignment details for a candidate continuation."""

    key: str = Field(description="Unique candidate identifier or option key.")
    text: str = Field(description="Raw candidate text.")
    full_text: str = Field(description="Full text evaluated (prompt + boundary + candidate).")
    prompt_token_count: int = Field(
        description="Index in full_token_ids where candidate tokens begin."
    )
    candidate_token_ids: List[int] = Field(
        default_factory=list,
        description="Exact token IDs corresponding to the candidate span."
    )
    full_token_ids: List[int] = Field(
        default_factory=list,
        description="Full token IDs for the concatenated prompt and candidate."
    )
    is_multi_token: bool = Field(
        description="True if candidate consists of more than 1 token in prompt context."
    )
    is_empty: bool = Field(
        description="True if candidate text or candidate token sequence is empty."
    )
    is_whitespace: bool = Field(
        description="True if candidate string is solely whitespace."
    )
    is_punctuation: bool = Field(
        description="True if candidate text consists entirely of punctuation characters."
    )
    is_unicode: bool = Field(
        description="True if candidate text contains non-ASCII Unicode characters."
    )
    alignment_status: str = Field(
        description="Alignment verification outcome ('exact_prefix', 'boundary_whitespace_merged', 'boundary_decoded_match', 'fallback_direct_encode', 'empty_candidate')."
    )


def format_prompt_candidate_text(prompt: str, candidate_text: str) -> str:
    """Format prompt and candidate text respecting leading/trailing whitespace boundaries."""
    if not candidate_text:
        return prompt

    if prompt.endswith(" ") or prompt.endswith("\t") or prompt.endswith("\n"):
        if candidate_text.startswith(" ") or candidate_text.startswith("\t") or candidate_text.startswith("\n"):
            return prompt + candidate_text.lstrip(" \t\n")
        return prompt + candidate_text
    elif candidate_text.startswith(" ") or candidate_text.startswith("\t") or candidate_text.startswith("\n"):
        return prompt + candidate_text
    else:
        # Neither has whitespace boundary.
        # If candidate starts with attached punctuation like '.', ',', '!', ')', join directly.
        if candidate_text and candidate_text[0] in ".,!?;:)]}":
            return prompt + candidate_text
        return prompt + " " + candidate_text


def tokenize_candidate(
    tokenizer: Any,
    prompt: str,
    candidate_text: str,
    candidate_key: str = "",
) -> CandidateTokenInfo:
    """Extract tokenizer-aware candidate token IDs with prefix alignment verification.

    Args:
        tokenizer: Hugging Face AutoTokenizer or any object with .encode() and .decode().
        prompt: Context prompt string.
        candidate_text: Candidate completion string.
        candidate_key: Identifier key for candidate.

    Returns:
        CandidateTokenInfo detailing token IDs, multi-token status, and alignment status.
    """
    is_empty = len(candidate_text.strip()) == 0
    is_whitespace = bool(candidate_text and candidate_text.isspace())
    clean_text = candidate_text.strip()
    is_punctuation = bool(clean_text and all(c in string.punctuation for c in clean_text))
    is_unicode = any(ord(c) > 127 for c in candidate_text)

    if not candidate_text:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if hasattr(tokenizer, "encode") else []
        return CandidateTokenInfo(
            key=candidate_key,
            text=candidate_text,
            full_text=prompt,
            prompt_token_count=len(prompt_ids),
            candidate_token_ids=[],
            full_token_ids=prompt_ids,
            is_multi_token=False,
            is_empty=True,
            is_whitespace=is_whitespace,
            is_punctuation=False,
            is_unicode=False,
            alignment_status="empty_candidate",
        )

    full_text = format_prompt_candidate_text(prompt, candidate_text)

    # If tokenizer has encode method
    if hasattr(tokenizer, "encode"):
        full_ids = list(tokenizer.encode(full_text, add_special_tokens=False))
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        p_len = len(prompt_ids)

        cand_ids: List[int] = []
        alignment_status: str

        # 1. Exact prompt prefix match
        if full_ids[:p_len] == prompt_ids:
            cand_ids = full_ids[p_len:]
            cand_start = p_len
            alignment_status = "exact_prefix"
        else:
            # 2. Whitespace absorption at boundary (e.g. trailing space absorbed into candidate token)
            p_strip_ids = list(tokenizer.encode(prompt.rstrip(), add_special_tokens=False))
            p_strip_len = len(p_strip_ids)
            if full_ids[:p_strip_len] == p_strip_ids:
                cand_ids = full_ids[p_strip_len:]
                cand_start = p_strip_len
                alignment_status = "boundary_whitespace_merged"
            else:
                # 3. Decoded boundary scan
                p_strip = prompt.rstrip()
                matched_idx = None
                if hasattr(tokenizer, "decode"):
                    for i in range(len(full_ids)):
                        decoded_prefix = tokenizer.decode(full_ids[:i], skip_special_tokens=True).rstrip()
                        if decoded_prefix == p_strip:
                            matched_idx = i
                            break

                if matched_idx is not None:
                    cand_ids = full_ids[matched_idx:]
                    cand_start = matched_idx
                    alignment_status = "boundary_decoded_match"
                else:
                    # 4. Fallback direct encode with leading whitespace if needed
                    cand_prep = (" " + candidate_text) if not candidate_text.startswith(" ") else candidate_text
                    cand_ids = list(tokenizer.encode(cand_prep, add_special_tokens=False))
                    cand_start = max(0, len(full_ids) - len(cand_ids))
                    alignment_status = "fallback_direct_encode"

        is_multi = len(cand_ids) > 1
        return CandidateTokenInfo(
            key=candidate_key,
            text=candidate_text,
            full_text=full_text,
            prompt_token_count=cand_start,
            candidate_token_ids=cand_ids,
            full_token_ids=full_ids,
            is_multi_token=is_multi,
            is_empty=(len(cand_ids) == 0),
            is_whitespace=is_whitespace,
            is_punctuation=is_punctuation,
            is_unicode=is_unicode,
            alignment_status=alignment_status,
        )
    else:
        # Synthetic / mock tokenizer fallback
        tokens = candidate_text.strip().split()
        cand_ids = [hash(t) % 32000 for t in tokens] if tokens else []
        return CandidateTokenInfo(
            key=candidate_key,
            text=candidate_text,
            full_text=full_text,
            prompt_token_count=len(prompt.split()),
            candidate_token_ids=cand_ids,
            full_token_ids=[1] * len(prompt.split()) + cand_ids,
            is_multi_token=(len(cand_ids) > 1),
            is_empty=(len(cand_ids) == 0),
            is_whitespace=is_whitespace,
            is_punctuation=is_punctuation,
            is_unicode=is_unicode,
            alignment_status="mock_fallback",
        )


def tokenize_candidate_set(
    tokenizer: Any,
    prompt: str,
    candidate_strings: Dict[str, str],
) -> Dict[str, CandidateTokenInfo]:
    """Tokenize and align all candidates for a given prompt."""
    return {
        key: tokenize_candidate(tokenizer, prompt, text, candidate_key=key)
        for key, text in candidate_strings.items()
    }


def requires_sequence_scoring_for_candidates(
    tokenizer: Any,
    prompt: str,
    candidate_strings: Dict[str, str],
) -> bool:
    """Check whether any candidate in the set requires multi-token sequence scoring."""
    for key, text in candidate_strings.items():
        info = tokenize_candidate(tokenizer, prompt, text, candidate_key=key)
        if info.is_multi_token:
            return True
    return False


def get_shared_prefix_token_ids(
    token_infos: Sequence[CandidateTokenInfo],
) -> List[int]:
    """Find shared prefix token IDs among candidates, if any."""
    if not token_infos:
        return []
    min_len = min(len(info.candidate_token_ids) for info in token_infos)
    if min_len == 0:
        return []
    shared = []
    first_ids = token_infos[0].candidate_token_ids
    for idx in range(min_len):
        tid = first_ids[idx]
        if all(info.candidate_token_ids[idx] == tid for info in token_infos):
            shared.append(tid)
        else:
            break
    return shared
