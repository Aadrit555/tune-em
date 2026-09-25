# Decision Levels: L0 vs L1 vs L2

`anydecision` implements a formal 3-tier hierarchy for extracting decisions from language models:

## Level L0: Raw Model Readout

- **Input Requirements**: Zero labeled data. Single prompt forward pass.
- **Mechanism**: Evaluates next-token or sequence log-probabilities for candidates against a single prompt template.
- **Strengths**: Lowest latency (sub-millisecond on mock/GPU), minimal compute overhead.
- **Weaknesses**: Subject to option-order bias, prompt wording fragility, and standard LLM overconfidence.

```python
decision = engine.decide(question, level="L0")
```

---

## Level L1: Zero-Label Debiasing & Ensembles

- **Input Requirements**: Zero human-labeled data.
- **Mechanism**:
  1. Systematically permutes the presentation order of candidate choices.
  2. Renders multiple diverse prompt templates (e.g. `minimal`, `structured`, `qa`).
  3. Maps positional outputs back to canonical semantic keys.
  4. Aggregates distributions across runs via probability mean, logit mean, or harmonic mean.
- **Observable Diagnostics**:
  - `number_of_permutations`
  - `template_agreement`
  - `option_order_sensitivity` (Total Variation distance)
  - `entropy`
- **Strengths**: Significantly reduces prompt formatting sensitivity and position bias without requiring any annotated datasets.

```python
decision = engine.decide(
    question,
    level="L1",
    num_permutations=4,
    templates=["minimal", "structured", "qa"],
)
```

---

## Level L2: Statistically Calibrated

- **Input Requirements**: Small labeled calibration set (e.g. 20–100 examples).
- **Mechanism**: Learns a lightweight post-hoc mapping (e.g. Temperature Scaling, Vector Scaling, Platt Scaling) that transforms predicted probabilities so that confidence aligns with empirical correctness.
- **Strengths**: Trustworthy probabilities for critical decision automation, selective risk control, and conformal prediction set guarantees.

```python
# 1. Fit once on validation data
engine.calibrate(calibration_dataset, method="temperature")

# 2. Evaluate with calibrated probabilities
decision = engine.decide(question, level="L2", target_error=0.05)
```
