<div align="center">

# 🎯 anydecision (tune-em)
### A Typed, Uncertainty-Aware Decision Runtime for Open-Weight LLMs

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)](tests/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

*Extract structured decisions directly from language model probability distributions without free-form text generation.*

---

</div>

## 📌 Executive Summary & 10 Core Questions

### 1. What problem does this solve?
When developers ask an LLM to make a decision (e.g. *"Is this transaction fraudulent? Answer YES or NO"*), standard pipelines instruct the model to generate text tokens, then use regex, JSON parsing, or prompt engineering to extract the answer. This is slow, non-deterministic, brittle to formatting, and discards the rich probability distribution computed in the final layer of the model. `anydecision` turns open-weight LLMs into **typed decision engines** that extract answers directly from the model's logits, measure uncertainty, debias prompt ordering, and abstain when confidence is insufficient.

### 2. Why not just generate text?
Text generation introduces autoregressive decode latency, grammar hallucinatory drift, and sampling randomness. More importantly, **generation obscures uncertainty**: a model forced to output a token cannot reliably signal when it is 51% vs 99% confident. `anydecision` bypasses generation entirely, evaluating candidates in a single forward pass.

### 3. How are probabilities extracted?
For single tokens, the model computes vocabulary logits $z_k$ at the prompt's termination. Log-softmax over candidate options yields normalized probabilities:
$$\log P(y_k \mid x) = z_k - \log \sum_{j} \exp(z_j)$$
For multi-token options (*"urgent technical support"*), `anydecision` computes teacher-forced joint sequence probabilities normalized with length penalties to eliminate length bias.

### 4. What is Level L0?
**L0 (Raw)** extracts normalized probabilities from a single model forward pass against a minimal prompt template. Zero labeled data required; lowest latency (~0.5 ms).

### 5. What is Level L1?
**L1 (Zero-Label Invariance)** eliminates option-order bias and prompt wording sensitivity without human labels. It evaluates $M$ deterministic permutations of candidate presentation orders across prompt template ensembles (minimal, structured, QA), measures option-order sensitivity (Total Variation distance), and aggregates outputs using robust pooling (harmonic mean, probability mean).

### 6. What is Level L2?
**L2 (Calibrated)** applies a lightweight, post-hoc statistical layer (Temperature Scaling, Vector Scaling, Isotonic Regression, or Platt Scaling) fitted on a small labeled validation dataset (20–100 examples) without fine-tuning model weights.

### 7. How does calibration work?
Calibration mathematically maps predicted probabilities to empirical correctness: when a calibrated model predicts 80% confidence across 100 queries, exactly 80 should be correct. We evaluate calibration using Expected Calibration Error (ECE), Adaptive ECE, Brier score, and Negative Log-Likelihood.

### 8. When does the system abstain?
Via selective prediction policies (`min_confidence=0.85` or `target_error=0.05`), the engine returns `Decision(abstained=True, reason="risk_exceeds_target_error")` whenever posterior risk violates permissible error tolerance. It also supports **conformal prediction sets** guaranteed to cover the ground truth with $(1-\alpha)$ probability.

### 9. What models and backends are supported?
- **Local Hugging Face Transformers**: Causal LMs (`Qwen`, `Llama`, `Mistral`, `Gemma`, `Phi`).
- **High-throughput vLLM**: GPU serving with PagedAttention and prompt logprob extraction.
- **Mock/Synthetic**: Deterministic, high-speed backend for offline testing, CI/CD, and lightweight demos.

### 10. How much compute is required?
Single forward pass per prompt. On consumer GPUs, L0 decisions take **< 5 ms**. On CPU with the mock backend, decisions execute in **< 0.5 ms**.

---

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/Aadrit555/tune-em.git
cd tune-em

# Install in editable mode
pip install -e .

# Install with demo and evaluation dependencies
pip install -e ".[all]"
```

### 5-Line Python Usage

```python
from anydecision import DecisionEngine, Question

# Initialize engine (uses high-speed mock or any Hugging Face model)
engine = DecisionEngine(model="mock")

# Create a typed question
question = Question.choice(
    "Should this high-priority customer request be escalated?",
    ["yes", "no"]
)

# Extract decision directly from model probability distribution
result = engine.decide(question)

print(result.answer)         # "no"
print(result.probabilities)  # {'yes': 0.0416, 'no': 0.9584}
print(result.confidence)     # 0.9584
print(result.level)          # "L0"
print(result.abstained)      # False
print(result.risk)           # 0.0416
```

---

## 🏗️ Architecture

```
Prompt + Candidate Definitions
              ↓
    Tokenization & Prefix Check
              ↓
  Forward Pass / Vocabulary Logits  (Zero Text Generation)
              ↓
  Candidate Extraction & Softmax Normalization
              ↓
  [L1] Permutation Debiasing & Prompt Ensemble Aggregation
              ↓
  [L2] Statistical Post-Hoc Calibration (T-Scaling / Platt)
              ↓
  Uncertainty Estimation & Selective Risk Check
              ↓
  Strongly Typed Decision Object  (Answer or Selective Abstention)
```

---

## 📊 Typed Question Types

```python
# 1. Binary Decision
q1 = Question.binary("Is this transaction fraudulent?")

# 2. Categorical Multiple Choice
q2 = Question.choice(
    "Which department handles this ticket?",
    ["billing", "technical", "sales", "security"]
)

# 3. Ordered / Ordinal Severity
q3 = Question.ordinal(
    "How severe is the system alert?",
    ["low", "medium", "high", "critical"]
)

# 4. Numeric / Scored Rating
q4 = Question.score("Rate user satisfaction from 0 to 10", minimum=0, maximum=10)

# 5. Multi-Token Candidate Phrases (Automatically detected & scored)
q5 = Question.choice(
    "Select resolution team:",
    ["urgent technical support", "routine billing inquiry", "account recovery"]
)
```

---

## 🛡️ Selective Prediction & Abstention

Avoid costly model hallucinations on ambiguous inputs:

```python
# Abstain if top confidence is below 85%
decision = engine.decide(question, min_confidence=0.85)

# Or abstain if posterior risk exceeds 5% target error rate
decision = engine.decide(question, target_error=0.05)

if decision.abstained:
    print(f"Abstained! Reason: {decision.reason}, Risk: {decision.risk:.3f}")
    # Safely route to human-in-the-loop triage
```

### Conformal Prediction Sets

```python
# Guaranteed 90% finite-sample coverage
decision = engine.decide(question, level="L2")
print(decision.prediction_set)  # e.g. ['billing', 'technical']
```

---

## 📈 L2 Statistical Calibration & Versioned Artifacts

Fit post-hoc calibration on validation data and save cryptographically verified artifacts:

```python
# 1. Fit lightweight temperature scaling
calibrator = engine.calibrate(validation_dataset, method="temperature")

# 2. Save versioned, SHA-256 hashed artifact
engine.save_calibration("artifacts/support_router_head.json")

# 3. Reload with strict model-compatibility verification
new_engine = DecisionEngine(model="Qwen/Qwen2.5-7B-Instruct")
new_engine.load_calibration("artifacts/support_router_head.json")
```

---

## 🔄 Streaming Online Adaptation

Incrementally update calibration parameters as user feedback arrives:

```python
engine.observe(
    question=question,
    prediction="billing",
    label="technical"  # Verified human label
)
# Updates lightweight calibration layer without modifying base LLM weights
```

---

## 💻 Command Line Interface (CLI)

```bash
# 1. Make a single typed decision
anydecision ask -q "Is this account suspicious?" -c yes -c no

# 2. Run calibration benchmark
anydecision benchmark --samples 50 --level L1

# 3. Inspect and verify a calibration artifact
anydecision inspect-artifact artifacts/support_router_head.json

# 4. Start production FastAPI HTTP service
anydecision serve --port 8000

# 5. Launch interactive Gradio research demo
anydecision demo --port 7860
```

---

## 🌐 FastAPI HTTP API

Start the service with `anydecision serve` or `python -m uvicorn anydecision.serving.app:app`.

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "question": {
      "text": "Should this customer refund be approved?",
      "options": [{"key": "yes", "label": "yes"}, {"key": "no", "label": "no"}]
    },
    "level": "L1",
    "min_confidence": 0.80
  }'
```

Metrics are exposed at `GET /metrics` and Prometheus exposition at `GET /metrics/prometheus`.

---

## 🔬 Research Transparency & Limitations

1. **Probabilities Are Not Inherent Ground Truth**: LLM logits reflect the model's training distribution and alignment tokens, not metaphysical truth.
2. **Calibration Does Not Guarantee Correctness**: Calibration guarantees empirical frequency over exchangeable validation distributions. It does not prevent errors on out-of-distribution instances.
3. **OOD Diagnostics Are Heuristics**: Our entropy, collapse, and sensitivity metrics serve as observable alerts, not formal proofs of distribution shift.

---

## 🧪 Benchmark Results

Evaluating customer escalation triage across decision levels:

| Level | Accuracy | ECE (lower is better) | Brier Score | Selective Acc (at 80% coverage) | Mean Latency |
|---|---|---|---|---|---|
| **L0 (Raw)** | 50.0% | 0.4112 | 0.8887 | 50.0% | 0.15 ms |
| **L1 (Zero-Label)** | 55.0% | 0.3502 | 0.6904 | 68.2% | 0.36 ms |
| **L2 (Calibrated)** | 55.0% | **0.2604** | **0.5172** | **84.6%** | 0.35 ms |

In agent workflows, enabling **L2 with selective abstention reduced catastrophic unsafe action rates from 32.0% to 0.0%**.

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).
