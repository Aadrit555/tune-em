"""L1 Zero-Label Invariance: Comparing raw L0 against permutation debiasing."""

from anydecision import DecisionEngine, DecisionLevel, Question

engine = DecisionEngine(model="mock")

question = Question.choice(
    text="Which data structure provides average O(1) key-value lookup?",
    choices=["hash table", "binary search tree", "linked list", "skip list"],
)

# L0: Raw model probabilities with single fixed prompt
res_l0 = engine.decide(question, level=DecisionLevel.L0)

# L1: Option permutations + prompt template ensemble debiasing
res_l1 = engine.decide(
    question,
    level=DecisionLevel.L1,
    num_permutations=4,
    templates=["minimal", "structured", "qa"],
    trace=True,
)

print("=" * 65)
print("       L0 (RAW) vs L1 (ZERO-LABEL DEBIASED) READOUT")
print("=" * 65)

print("\n[L0 - Raw Single Pass]")
print(f"Decision:     {res_l0.answer}")
print(f"Confidence:   {res_l0.confidence:.2%}")
print(f"Probabilities:{res_l0.probabilities}")

print("\n[L1 - Debiased Ensemble across 4 Permutations & 3 Templates]")
print(f"Decision:     {res_l1.answer}")
print(f"Confidence:   {res_l1.confidence:.2%}")
print(f"Probabilities:{res_l1.probabilities}")

if res_l1.diagnostics:
    diag = res_l1.diagnostics
    print("\n[Invariance Diagnostics]")
    print(f"Backend Forward Passes:   {diag.number_of_backend_calls}")
    print(f"Permutations Tested:      {diag.number_of_permutations}")
    print(f"Permutation Agreement:    {diag.template_agreement * 100:.1f}%")
    print(f"Option-Order Sensitivity: {diag.option_order_sensitivity:.4f}")
    print(f"Entropy:                  {diag.entropy:.4f} nats")
print("=" * 65)
