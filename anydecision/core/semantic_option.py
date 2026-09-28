"""Semantic Option definitions and alias probability aggregation.

Enables candidate options with rich semantic properties:
- Multiple aliases / synonyms ('invoice', 'refund', 'charge' -> 'billing')
- Prior probabilities for Bayes prior correction
- Direct operational action mapping
- Alias-sum probability aggregation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

from anydecision.core.types import OptionDefinition


class SemanticOption(BaseModel):
    """Enriched option definition supporting aliases, priors, and action mappings."""
    id: str = Field(description="Unique stable key for the candidate option.")
    label: str = Field(description="Primary descriptive label presented to the model.")
    aliases: List[str] = Field(
        default_factory=list,
        description="Synonyms, sub-terms, or alternative wordings for multi-surface scoring."
    )
    prior_probability: Optional[float] = Field(
        default=None,
        description="Empirical or domain prior P(y) for Bayesian prior correction."
    )
    action_mapping: Optional[str] = Field(
        default=None,
        description="Associated operational system action (e.g. 'route_billing_desk')."
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed contextual description."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_option_definition(self) -> OptionDefinition:
        """Convert to basic engine OptionDefinition."""
        return OptionDefinition(
            key=self.id,
            label=self.label,
            description=self.description,
            metadata={
                **self.metadata,
                "aliases": self.aliases,
                "action_mapping": self.action_mapping,
                "prior_probability": self.prior_probability,
            }
        )


class SemanticOptionRegistry:
    """Manages alias resolution and probability aggregation across synonyms."""

    def __init__(self, options: Sequence[SemanticOption]) -> None:
        self.options: Dict[str, SemanticOption] = {opt.id: opt for opt in options}

    def expand_candidate_strings(self) -> Dict[str, str]:
        """Generate flat mapping of candidate tokens/strings including all aliases.

        Returns mapping in the format: {option_id::alias_idx: candidate_text}
        """
        candidates: Dict[str, str] = {}
        for opt in self.options.values():
            # Primary label
            candidates[f"{opt.id}::__primary__"] = opt.label
            # Aliases
            for a_idx, alias in enumerate(opt.aliases):
                candidates[f"{opt.id}::alias_{a_idx}"] = alias
        return candidates

    def aggregate_alias_probabilities(
        self,
        surface_probabilities: Dict[str, float],
        method: str = "sum",
    ) -> Dict[str, float]:
        """Aggregate probabilities across all surface forms/aliases into canonical option IDs.

        Args:
            surface_probabilities: Map of surface_id -> probability.
            method: 'sum' (sum of alias probabilities) or 'max' (highest alias probability).

        Returns:
            Map of canonical option_id -> normalized probability.
        """
        aggregated: Dict[str, float] = {opt_id: 0.0 for opt_id in self.options}

        for surface_key, prob in surface_probabilities.items():
            opt_id = surface_key.split("::")[0]
            if opt_id in aggregated:
                if method == "max":
                    aggregated[opt_id] = max(aggregated[opt_id], prob)
                else:  # sum
                    aggregated[opt_id] += prob

        # Re-normalize across canonical options
        total = sum(aggregated.values())
        if total > 0:
            return {k: float(v / total) for k, v in aggregated.items()}
        return aggregated

    def apply_prior_correction(
        self,
        probabilities: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply Bayesian prior correction P(y|x) / P(y) if priors are specified."""
        has_priors = any(opt.prior_probability is not None for opt in self.options.values())
        if not has_priors:
            return probabilities

        corrected = {}
        for opt_id, p in probabilities.items():
            prior = self.options[opt_id].prior_probability or (1.0 / len(self.options))
            # Bayes update with uniform target: P_corr ~ P(y|x) / P(y)
            corrected[opt_id] = p / max(1e-6, prior)

        total = sum(corrected.values())
        if total > 0:
            return {k: float(v / total) for k, v in corrected.items()}
        return corrected
