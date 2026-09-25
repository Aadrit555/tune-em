"""Demonstration of multi-token candidate answer scoring and length normalization."""

from anydecision import DecisionEngine, Question
from anydecision.scoring.sequence import SequenceScoringMethod

engine = DecisionEngine(model="mock")

# Multi-token phrases that span multiple BPE tokens
choices = [
    "urgent technical support",
    "routine billing inquiry",
    "account recovery assistance",
    "security and compliance review",
]

question = Question.choice(
    text="Customer request: 'I am locked out of our root AWS console and suspect credentials were leaked.'\nWhich department should take action?",
    choices=choices,
)

print("=" * 60)
print("     MULTI-TOKEN SEQUENCE PROBABILITY READOUT")
print("=" * 60)

for method in [
    SequenceScoringMethod.SUM,
    SequenceScoringMethod.MEAN,
    SequenceScoringMethod.LENGTH_NORMALIZED,
]:
    res = engine.decide(question, scoring_method=method.value)
    print(f"\nScoring Method: [{method.value.upper()}]")
    print(f"Top Choice:     {res.answer} (Confidence: {res.confidence:.2%})")
    for opt, prob in res.probabilities.items():
        print(f"  - {opt:35s}: {prob * 100:.2f}%")
print("=" * 60)
