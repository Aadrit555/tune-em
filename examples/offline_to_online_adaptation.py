"""Streaming online adaptation: Updating calibration layer from user feedback."""

from anydecision import DecisionEngine, DecisionLevel, Question

engine = DecisionEngine(model="mock")

print("=" * 65)
print("     ONLINE ADAPTATION VIA ENGINE.OBSERVE()")
print("=" * 65)

# Simulate streaming operational feedback loop
categories = ["billing", "technical", "sales"]
feedbacks = [
    ("Payment declined error 4002", "billing"),
    ("Kernel panic during cuda initialization", "technical"),
    ("Quote request for 100 enterprise seats", "sales"),
    ("Invoice shows extra tax line item", "billing"),
    ("Connection refused to port 5432", "technical"),
    ("Discount negotiation for multi-year contract", "sales"),
    ("Credit card expired notification", "billing"),
    ("Segmentation fault in C++ extension", "technical"),
    ("RFP submission guidelines", "sales"),
    ("Double charge on monthly subscription", "billing"),
]

for idx, (text, true_label) in enumerate(feedbacks):
    q = Question.choice(text=text, choices=categories, question_id=f"stream_{idx}")
    initial_dec = engine.decide(q, level=DecisionLevel.L0)

    # Observe feedback and trigger lightweight online calibration update
    was_refit = engine.observe(
        question=q,
        prediction=initial_dec.answer or "billing",
        label=true_label,
    )

    status = "[ADAPTATION REFIT TRIGGERED]" if was_refit else "[BUFFERED]"
    print(f"Sample {idx + 1:02d}: Pred='{initial_dec.answer}' | Label='{true_label}' -> {status}")

print(f"\nFinal Online Adaptation Updates Count: {engine.adapter.updates_count}")
print(f"Buffered Observations in Memory:     {engine.adapter.buffer.count()}")
print("=" * 65)

