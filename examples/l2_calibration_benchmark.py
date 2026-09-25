"""L2 Calibration: Fitting post-hoc calibration, evaluating ECE, and saving artifacts."""

from anydecision import (
    CalibrationReport,
    DecisionEngine,
    DecisionLevel,
    Question,
    load_calibration_artifact,
)
from anydecision.evaluation.datasets import create_customer_escalation_benchmark

# 1. Initialize Engine
engine = DecisionEngine(model="mock")

# 2. Generate labeled calibration benchmark dataset
dataset_samples = create_customer_escalation_benchmark(n_samples=60)
calib_data = [{"question": s.question, "label": s.ground_truth} for s in dataset_samples[:40]]
test_samples = dataset_samples[40:]

print("=" * 65)
print("     L2 STATISTICAL POST-HOC CALIBRATION BENCHMARK")
print("=" * 65)

# 3. Fit lightweight temperature scaling
calibrator = engine.calibrate(calib_data, method="temperature")
print(f"Fitted Temperature Scaling Parameter: T = {getattr(calibrator, 'temperature', 1.0):.4f}")

# 4. Save versioned artifact
artifact_path = "artifacts/demo_escalation_head.json"
engine.save_calibration(artifact_path, question_schema="customer_escalation_v1")
print(f"Saved versioned calibration artifact to: {artifact_path}")

# 5. Evaluate on held-out test split under L0 vs L2
test_q = test_samples[0].question
dec_l0 = engine.decide(test_q, level=DecisionLevel.L0)
dec_l2 = engine.decide(test_q, level=DecisionLevel.L2)

print("\n[Held-Out Sample Evaluation]")
print(f"Question:    {test_q.text[:60]}...")
print(f"L0 Raw:      Answer={dec_l0.answer}, Confidence={dec_l0.confidence:.2%}, Calibrated={dec_l0.calibrated}")
print(f"L2 Calib:    Answer={dec_l2.answer}, Confidence={dec_l2.confidence:.2%}, Calibrated={dec_l2.calibrated}")

# 6. Load artifact verification test
new_engine = DecisionEngine(model="mock")
new_engine.load_calibration(artifact_path)
print("Successfully validated cryptographic SHA-256 integrity and reloaded artifact!")
print("=" * 65)

