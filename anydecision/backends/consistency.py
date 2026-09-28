"""Cross-backend numerical consistency test suite.

Compares probability vectors, candidate log-probabilities, and decisions across
different serving runtimes (e.g. Hugging Face vs vLLM vs Mock) to verify numerical
and behavioral stability prior to production deployment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from anydecision.backends.base import BaseBackend
    from anydecision.core.question import Question


class BackendPairComparison(BaseModel):
    """Pairwise numerical delta between two backends on an identical question."""
    backend_a: str
    backend_b: str
    decision_match: bool
    answer_a: str
    answer_b: str
    probability_tv_distance: float = Field(
        description="Total Variation distance: 0.5 * sum |P_A(y) - P_B(y)|."
    )
    max_logprob_difference: float = Field(
        description="Maximum absolute log-probability difference across all candidates."
    )
    kl_divergence: float = Field(
        description="Kullback-Leibler divergence D_KL(P_A || P_B)."
    )
    mean_abs_error: float = Field(
        description="Mean absolute error between probability distributions."
    )


class BackendConsistencyReport(BaseModel):
    """Observable report measuring numerical consistency across multiple backends."""
    question_id: str
    num_backends: int
    all_decisions_agree: bool
    max_tv_distance: float
    max_logprob_delta: float
    pairwise_comparisons: List[BackendPairComparison]
    status: str = Field(description="'PASS', 'WARNING', or 'FAIL'")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Cross-Backend Consistency Report (Question: {self.question_id}) ===",
            f"Evaluated Backends:          {self.num_backends}",
            f"All Decisions Agree:         {self.all_decisions_agree}",
            f"Max Probability TV Distance: {self.max_tv_distance:.6f}",
            f"Max Logprob Delta:           {self.max_logprob_delta:.6f}",
            f"Consistency Verification:    [{self.status}]",
            "",
            f"{'Backend Pair':<32} | {'Agreement':<10} | {'TV Distance':<12} | {'Max Delta':<10}",
            "-" * 72,
        ]
        for pair in self.pairwise_comparisons:
            p_name = f"{pair.backend_a} vs {pair.backend_b}"
            lines.append(
                f"{p_name:<32} | {str(pair.decision_match):<10} | "
                f"{pair.probability_tv_distance:>11.6f} | {pair.max_logprob_difference:>9.6f}"
            )
        lines.append("========================================================================")
        return "\n".join(lines)


class CrossBackendConsistencyChecker:
    """Verifies that decisions and probabilities remain invariant across serving runtimes."""

    def __init__(
        self,
        max_acceptable_tv: float = 0.05,
        max_acceptable_logprob_delta: float = 0.25,
    ) -> None:
        self.max_acceptable_tv = max_acceptable_tv
        self.max_acceptable_logprob_delta = max_acceptable_logprob_delta

    def compare_backends(
        self,
        question: Question,
        backends: Sequence[BaseBackend],
    ) -> BackendConsistencyReport:
        """Run identical prompt and candidate evaluation across all backends and measure delta."""
        if len(backends) < 2:
            raise ValueError("CrossBackendConsistencyChecker requires at least two backends to compare.")

        from anydecision.bias.templates import DEFAULT_TEMPLATES
        template = DEFAULT_TEMPLATES.get("structured")
        prompt = template.render(question)
        candidate_strings = {opt.key: opt.label for opt in question.options}
        keys = list(candidate_strings.keys())

        # Query all backends
        distributions: Dict[str, Dict[str, float]] = {}
        logprobs: Dict[str, Dict[str, float]] = {}
        top_answers: Dict[str, str] = {}

        for b in backends:
            meta = b.get_metadata()
            name = meta.backend_name
            if name in distributions:
                name = f"{name}_{id(b)}"

            lp = b.next_token_logprobs(prompt, candidate_strings)
            logprobs[name] = lp

            # Softmax to normalized probabilities
            lp_arr = np.array([lp[k] for k in keys], dtype=np.float64)
            p_arr = np.exp(lp_arr - np.max(lp_arr))
            p_arr /= np.sum(p_arr)
            prob_dict = {k: float(p) for k, p in zip(keys, p_arr)}
            distributions[name] = prob_dict

            top_ans = max(prob_dict.items(), key=lambda x: x[1])[0]
            top_answers[name] = top_ans

        # Pairwise comparisons
        b_names = list(distributions.keys())
        pairwise: List[BackendPairComparison] = []
        max_tv = 0.0
        max_delta = 0.0
        all_match = True

        for i in range(len(b_names)):
            for j in range(i + 1, len(b_names)):
                name_a, name_b = b_names[i], b_names[j]
                p_a = np.array([distributions[name_a][k] for k in keys], dtype=np.float64)
                p_b = np.array([distributions[name_b][k] for k in keys], dtype=np.float64)
                lp_a = np.array([logprobs[name_a][k] for k in keys], dtype=np.float64)
                lp_b = np.array([logprobs[name_b][k] for k in keys], dtype=np.float64)

                tv = float(0.5 * np.sum(np.abs(p_a - p_b)))
                delta = float(np.max(np.abs(lp_a - lp_b)))
                mae = float(np.mean(np.abs(p_a - p_b)))

                # KL divergence D_KL(P_A || P_B)
                kl = float(np.sum(p_a * np.log(np.clip(p_a, 1e-12, 1.0) / np.clip(p_b, 1e-12, 1.0))))

                ans_a = top_answers[name_a]
                ans_b = top_answers[name_b]
                match = (ans_a == ans_b)

                if tv > max_tv:
                    max_tv = tv
                if delta > max_delta:
                    max_delta = delta
                if not match:
                    all_match = False

                pairwise.append(
                    BackendPairComparison(
                        backend_a=name_a,
                        backend_b=name_b,
                        decision_match=match,
                        answer_a=ans_a,
                        answer_b=ans_b,
                        probability_tv_distance=tv,
                        max_logprob_difference=delta,
                        kl_divergence=max(0.0, kl),
                        mean_abs_error=mae,
                    )
                )

        if not all_match or max_tv > self.max_acceptable_tv or max_delta > self.max_acceptable_logprob_delta:
            status = "FAIL" if not all_match else "WARNING"
        else:
            status = "PASS"

        return BackendConsistencyReport(
            question_id=question.id,
            num_backends=len(backends),
            all_decisions_agree=all_match,
            max_tv_distance=max_tv,
            max_logprob_delta=max_delta,
            pairwise_comparisons=pairwise,
            status=status,
        )
