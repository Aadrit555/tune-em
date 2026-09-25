"""Selective prediction, abstention on risk/entropy, and conformal prediction sets."""

from anydecision import DecisionEngine, DecisionLevel, Question
from anydecision.calibration.conformal import ConformalPredictor
import numpy as np

engine = DecisionEngine(model="mock")

# Ambiguous or borderline question
question = Question.choice(
    text="Ambiguous query: 'Please reset my permissions or downgrade my organization tier.'\nPrimary category?",
    choices=["billing", "technical_support", "sales_inquiry", "other"],
)

print("=" * 65)
print("     SELECTIVE PREDICTION & ABSTENTION ON EXCESSIVE RISK")
print("=" * 65)

# 1. Unconstrained decision (forced answer)
forced_dec = engine.decide(question, level=DecisionLevel.L0, allow_abstain=False)
print(f"Forced Guess:  {forced_dec.answer} (Confidence: {forced_dec.confidence:.2%}, Risk: {forced_dec.risk:.3f})")

# 2. Selective decision with target risk threshold
selective_dec = engine.decide(
    question,
    level=DecisionLevel.L0,
    min_confidence=0.85,
    allow_abstain=True,
)
print(f"Selective:     Abstained={selective_dec.abstained}, Answer={selective_dec.answer}")
print(f"Reason:        {selective_dec.reason}")

# 3. Conformal Prediction Set (guaranteed coverage set instead of single forced answer)
mock_calib_probs = np.array([
    [0.70, 0.15, 0.10, 0.05],
    [0.85, 0.05, 0.05, 0.05],
    [0.60, 0.20, 0.10, 0.10],
    [0.90, 0.04, 0.03, 0.03],
])
mock_calib_labels = np.array([0, 0, 1, 0])
cp = ConformalPredictor(alpha=0.10).fit(mock_calib_probs, mock_calib_labels)
engine.conformal_predictor = cp

conformal_dec = engine.decide(question, level=DecisionLevel.L0)
print(f"\nConformal Prediction Set (90% finite-sample coverage guarantee):")
print(f"Set:           {conformal_dec.prediction_set}")
print("=" * 65)
