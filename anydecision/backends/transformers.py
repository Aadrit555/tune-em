"""Hugging Face Transformers local causal LM backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch

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

    @property
    def token_counts_exact(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        """Exact token count using the loaded HF tokenizer (no special tokens added)."""
        return max(1, len(self.tokenizer.encode(text, add_special_tokens=False)))

    def requires_sequence_scoring(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> bool:
        """Check whether any candidate encodes to more than 1 token in the context of the prompt."""
        return requires_sequence_scoring_for_candidates(self.tokenizer, prompt, candidate_strings)

    def next_token_logprobs_detailed(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, Any]:
        """Return both raw full-vocabulary logprobs and candidate-conditional logprobs."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            next_logits = outputs.logits[0, -1, :].to(torch.float32)

        vocab_logprobs = torch.log_softmax(next_logits, dim=-1).cpu().numpy()

        candidate_infos = tokenize_candidate_set(self.tokenizer, prompt, candidate_strings)
        raw_vocab_lps: Dict[str, float] = {}

        for key, info in candidate_infos.items():
            if info.candidate_token_ids:
                token_id = info.candidate_token_ids[0]
                if 0 <= token_id < len(vocab_logprobs):
                    raw_vocab_lps[key] = float(vocab_logprobs[token_id])
                else:
                    raw_vocab_lps[key] = -1e9
            else:
                raw_vocab_lps[key] = -1e9

        keys = list(raw_vocab_lps.keys())
        scores = np.array([raw_vocab_lps[k] for k in keys], dtype=np.float64)
        max_s = np.max(scores)
        log_z = max_s + np.log(np.sum(np.exp(scores - max_s)))
        norm_scores = scores - log_z

        return {
            "conditional_logprobs": {k: float(s) for k, s in zip(keys, norm_scores)},
            "raw_vocab_logprobs": raw_vocab_lps,
            "candidate_token_infos": candidate_infos,
        }

    def next_token_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
    ) -> Dict[str, float]:
        if self.requires_sequence_scoring(prompt, candidate_strings):
            return self.sequence_logprobs(prompt, candidate_strings)

        res = self.next_token_logprobs_detailed(prompt, candidate_strings)
        return res["conditional_logprobs"]

    def sequence_logprobs_detailed(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        scoring_method: str = "length_normalized",
    ) -> Dict[str, Any]:
        """Compute exact joint log-probabilities and length-penalized sequence scores."""
        scorer = SequenceScorer(
            method=SequenceScoringMethod(scoring_method)
            if scoring_method in SequenceScoringMethod._value2member_map_
            else SequenceScoringMethod.LENGTH_NORMALIZED
        )

        tokens_info = tokenize_candidate_set(self.tokenizer, prompt, candidate_strings)
        score_breakdowns: Dict[str, SequenceScoreResult] = {}

        with torch.no_grad():
            for key, info in tokens_info.items():
                if not info.candidate_token_ids:
                    score_breakdowns[key] = scorer.score_sequence([])
                    continue

                input_tensor = torch.tensor([info.full_token_ids], device=self.device)
                outputs = self.model(input_tensor)
                logits = outputs.logits[0].to(torch.float32)  # [seq_len, vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1)

                token_lps: List[float] = []
                for idx, tok_id in enumerate(info.candidate_token_ids):
                    # For token at pos in sequence, log P(tok | tok_{<pos}) is located at logits[pos - 1]
                    pos = info.prompt_token_count + idx
                    logit_pos = pos - 1
                    if 0 <= logit_pos < logits.shape[0]:
                        lp = float(log_probs[logit_pos, tok_id].item())
                        token_lps.append(lp)
                    else:
                        token_lps.append(-1e9)

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

    def _final_norm(self, hidden: "torch.Tensor") -> "torch.Tensor":
        """Apply the model's final layer-norm if the architecture exposes one."""
        model = self.model
        for attr_path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
            obj: object = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                norm = obj
            except AttributeError:
                continue
            if hasattr(norm, "__call__"):
                weight = getattr(norm, "weight", None)
                probe = hidden.to(dtype=weight.dtype) if weight is not None else hidden
                try:
                    return norm(probe).to(torch.float32)  # type: ignore[operator]
                except (AttributeError, RuntimeError, TypeError):
                    continue
        return hidden.to(torch.float32)

    def get_layer_hidden_states(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """Real per-layer last-token hidden states (1-based layer indices)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states
        if not hidden:
            return {}
        num_layers = len(hidden) - 1
        wanted = layers or list(range(1, num_layers + 1))
        states: Dict[int, np.ndarray] = {}
        for l in wanted:
            if 1 <= l <= num_layers:
                vec = hidden[l][0, -1, :].to(torch.float32).cpu().numpy()
                states[int(l)] = vec
        return states

    def get_layer_logprobs(
        self,
        prompt: str,
        candidate_strings: Dict[str, str],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Logit-lens candidate logprobs per layer.

        Each requested layer's last-token hidden state is projected through the
        model's final norm (where exposed) and lm_head. The final layer equals
        the standard next-token readout; earlier layers are the standard
        logit-lens approximation and are documented as such (not exact
        early-exited predictions).
        """
        candidate_infos = tokenize_candidate_set(self.tokenizer, prompt, candidate_strings)
        first_ids = {
            key: (info.candidate_token_ids[0] if info.candidate_token_ids else None)
            for key, info in candidate_infos.items()
        }
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states
        if not hidden:
            return {}
        num_layers = len(hidden) - 1
        wanted = layers or list(range(1, num_layers + 1))
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            return {}
        results: Dict[int, Dict[str, float]] = {}
        with torch.no_grad():
            for l in wanted:
                if not (1 <= l <= num_layers):
                    continue
                h = hidden[l][0, -1, :]
                head_weight = getattr(lm_head, "weight", None)
                head_dtype = head_weight.dtype if head_weight is not None else torch.float32
                h_in = h.to(dtype=head_dtype).unsqueeze(0)
                try:
                    logits = lm_head(self._final_norm(h_in)).to(torch.float32)[0]
                except RuntimeError:
                    logits = lm_head(h_in).to(torch.float32)[0]
                vocab_lps = torch.log_softmax(logits, dim=-1).cpu().numpy()
                raw = {
                    key: (float(vocab_lps[tid]) if tid is not None and 0 <= tid < len(vocab_lps) else -1e9)
                    for key, tid in first_ids.items()
                }
                keys = list(raw.keys())
                scores = np.array([raw[k] for k in keys], dtype=np.float64)
                max_s = np.max(scores)
                normed = scores - (max_s + np.log(np.sum(np.exp(scores - max_s))))
                results[int(l)] = {k: float(v) for k, v in zip(keys, normed)}
        return results

    def batch_next_token_logprobs(
        self,
        prompts: List[str],
        candidate_strings_list: List[Dict[str, str]],
    ) -> List[Dict[str, float]]:
        """Vectorized batched forward pass with left-padding for causal LM."""
        if not prompts:
            return []

        orig_pad_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        self.tokenizer.padding_side = orig_pad_side

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_logits = outputs.logits[:, -1, :].to(torch.float32)

        vocab_logprobs = torch.log_softmax(last_logits, dim=-1).cpu().numpy()

        results = []
        for b_idx, (p_text, candidate_strings) in enumerate(zip(prompts, candidate_strings_list)):
            cand_infos = tokenize_candidate_set(self.tokenizer, p_text, candidate_strings)
            cand_scores: Dict[str, float] = {}
            for key, info in cand_infos.items():
                if info.candidate_token_ids:
                    token_id = info.candidate_token_ids[0]
                    if 0 <= token_id < vocab_logprobs.shape[1]:
                        cand_scores[key] = float(vocab_logprobs[b_idx, token_id])
                    else:
                        cand_scores[key] = -1e9
                else:
                    cand_scores[key] = -1e9

            keys = list(cand_scores.keys())
            scores = np.array([cand_scores[k] for k in keys], dtype=np.float64)
            max_s = np.max(scores)
            log_z = max_s + np.log(np.sum(np.exp(scores - max_s)))
            norm_scores = scores - log_z
            results.append({k: float(s) for k, s in zip(keys, norm_scores)})

        return results
