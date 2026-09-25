"""Run comprehensive calibration, bias, and readout benchmarks across L0, L1, and L2."""

from pathlib import Path
import json
from anydecision import DecisionEngine, DecisionLevel
from anydecision.evaluation.benchmark import BenchmarkRunner
from anydecision.evaluation.datasets import create_customer_escalation_benchmark
from anydecision.evaluation.visualizer import ResearchVisualizer
from anydecision.calibration.report import CalibrationReport
import numpy as np

def main():
    print("=" * 65)
    print("      STARTING PRODUCTION RESEARCH BENCHMARK SUITE")
    print("=" * 65)

    engine = DecisionEngine(model="mock")
    runner = BenchmarkRunner(engine)

    # 1. Dataset split
    dataset = create_customer_escalation_benchmark(n_samples=80)
    train_calib = dataset[:40]
    test_eval = dataset[40:]

    # 2. Benchmark L0
    print("\n[Evaluating Level L0: Raw Model Readout]")
    res_l0 = runner.run(test_eval, name="Customer Escalation (L0)", level=DecisionLevel.L0)
    print(res_l0.summary())

    # 3. Benchmark L1
    print("\n[Evaluating Level L1: Zero-Label Debiasing & Ensembles]")
    res_l1 = runner.run(test_eval, name="Customer Escalation (L1)", level=DecisionLevel.L1)
    print(res_l1.summary())

    # 4. Fit L2 Calibration on train_calib
    print("\n[Fitting Level L2 Calibration Layer]")
    calib_items = [{"question": s.question, "label": s.ground_truth} for s in train_calib]
    engine.calibrate(calib_items, method="temperature")

    print("\n[Evaluating Level L2: Statistically Calibrated]")
    res_l2 = runner.run(test_eval, name="Customer Escalation (L2)", level=DecisionLevel.L2)
    print(res_l2.summary())

    # 5. Export JSON benchmark metrics
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_export = {
        "L0_raw": res_l0.model_dump(),
        "L1_zero_label": res_l1.model_dump(),
        "L2_calibrated": res_l2.model_dump(),
    }
    json_path = output_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_export, f, indent=2)
    print(f"\n[Saved JSON Benchmark Report to: {json_path}]")

    # 6. Generate Publication Figures
    calib_report = CalibrationReport.evaluate(
        probabilities=np.array([[d["yes"], d["no"]] for d in [engine.decide(s.question, level=DecisionLevel.L2).probabilities for s in test_eval]]),
        labels=np.array([1 if s.ground_truth == "no" else 0 for s in test_eval])
    )
    rel_path = output_dir / "reliability_diagram.png"
    ResearchVisualizer.plot_reliability_diagram(calib_report, save_path=rel_path)
    print(f"[Generated Reliability Diagram figure: {rel_path}]")

    risk_path = output_dir / "risk_coverage.png"
    ResearchVisualizer.plot_risk_coverage(calib_report, save_path=risk_path)
    print(f"[Generated Risk vs Coverage figure: {risk_path}]")
    print("=" * 65)

if __name__ == "__main__":
    main()

