"""Bias mitigation, option permutation, prompt ensembles, and debiasing."""

from anydecision.bias.aggregation import aggregate_distributions
from anydecision.bias.permutation import (
    compute_permutation_invariance_metrics,
    generate_permutations,
)
from anydecision.bias.templates import DEFAULT_TEMPLATES, PromptTemplate, TemplateRegistry

__all__ = [
    "DEFAULT_TEMPLATES",
    "PromptTemplate",
    "TemplateRegistry",
    "aggregate_distributions",
    "compute_permutation_invariance_metrics",
    "generate_permutations",
]

