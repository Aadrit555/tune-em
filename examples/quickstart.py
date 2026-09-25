"""Quickstart: Instant typed decision extraction directly from model probabilities.

Run:
    python examples/quickstart.py
"""

from anydecision import DecisionEngine, Question

# Initialize decision engine (uses high-speed deterministic mock by default, or any HuggingFace model)
engine = DecisionEngine(model="mock")

# Define a typed categorical question
question = Question.choice(
    "Should this customer escalation request be approved?",
    ["yes", "no"]
)

# Extract model decision directly from output probability distribution
result = engine.decide(question)

print("=" * 45)
print("             ANYDECISION QUICKSTART")
print("=" * 45)
print(f"Decision:         {result.answer}")
print(f"Probabilities:    {result.probabilities}")
print(f"Confidence:       {result.confidence:.2%}")
print(f"Uncertainty:      {result.uncertainty:.4f}")
print(f"Level:            {result.level}")
print(f"Abstained:        {result.abstained}")
print(f"Posterior Risk:   {result.risk:.4f}")
print(f"Method:           {result.method}")
if result.diagnostics:
    print(f"Latency:          {result.diagnostics.latency_ms:.2f} ms")
print("=" * 45)

