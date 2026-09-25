"""Hugging Face Transformers local causal LM backend."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from anydecision.backends.base import BaseBackend, ModelMetadata
from anydecision.scoring.sequence import SequenceScorer, SequenceScoringMethod


class TransformersBackend(BaseBackend):
    """Local inference backend using Hugging Face Transformers causal models.

    Performs direct forward passes to extract next-token and multi-token log probabilities.
    Never executes iterative text generation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

    def get_metadata(self) -> ModelMetadata:
        config = getattr(self.model, "config", None)
        return ModelMetadata(
            model_name=self.model_name,
            model_revision=getattr(config, "_commit_hash", "main") or "main",
            tokenizer_revision="main",
            hidden_size=getattr(config, "hidden_size", None),
            num_layers=getattr(config, "num_hidden_layers", None),
            vocab_size=getattr(config, "vocab_size", len(self.tokenizer)),
            dtype=str(self.dtype),
            device=str(self.device),
            backend_name="transformers",
            quantization=None,
            context_window=getattr(config, "max_position_embeddings", 4096),
        )

    def next_token_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, float]:
        start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Logits of last prompt token: [vocab_size]
            next_logits = outputs.logits[0, -1, :].to(torch.float32)

        # Log softmax across vocab
        vocab_logprobs = torch.log_softmax(next_logits, dim=-1).cpu().numpy()

        candidate_scores = {}
        for key, text in candidate_strings.items():
            # Check with and without leading space to support BPE tokenizers
            tokens_space = self.tokenizer.encode(" " + text.strip(), add_special_tokens=False)
            tokens_raw = self.tokenizer.encode(text.strip(), add_special_tokens=False)
            # Choose primary single token
            token_id = tokens_space[0] if tokens_space else (tokens_raw[0] if tokens_raw else 0)
            candidate_scores[key] = float(vocab_logprobs[token_id])

        # Normalize across candidates
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
        scorer = SequenceScorer(
            method=SequenceScoringMethod(scoring_method)
            if scoring_method in SequenceScoringMethod._value2member_map_
            else SequenceScoringMethod.LENGTH_NORMALIZED
        )

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        len_prompt = len(prompt_ids)

        candidate_scores = {}

        with torch.no_grad():
            for key, text in candidate_strings.items():
                full_text = prompt.rstrip() + " " + text.lstrip()
                full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
                candidate_token_ids = full_ids[len_prompt:]

                if not candidate_token_ids:
                    candidate_scores[key] = -1e9
                    continue

                input_tensor = torch.tensor([full_ids], device=self.device)
                outputs = self.model(input_tensor)
                logits = outputs.logits[0]  # [seq_len, vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1)

                # For token at pos t in candidate, log P(tok_t | tok_{<t}) is at logits index (t-1)
                token_lps = []
                for idx, tok_id in enumerate(candidate_token_ids):
                    logit_pos = len_prompt + idx - 1
                    if logit_pos >= 0 and logit_pos < logits.shape[0]:
                        lp = float(log_probs[logit_pos, tok_id].item())
                        token_lps.append(lp)

                candidate_scores[key] = scorer.score_tokens(token_lps)

        # Normalize over candidates
        keys = list(candidate_scores.keys())
        scores = np.array([candidate_scores[k] for k in keys], dtype=np.float64)
        max_s = np.max(scores)
        log_z = max_s + np.log(np.sum(np.exp(scores - max_s)))
        norm_scores = scores - log_z

        return {k: float(s) for k, s in zip(keys, norm_scores)}

