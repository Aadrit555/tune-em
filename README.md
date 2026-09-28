<p align="center">
  <img src="assets/banner.png" alt="tune-em banner" width="100%" />
</p>

```text
┌──(user@linux-runtime)-[~/tune-em]
└─$ ./tune-em --status

  _____ _   _ _   _ _____     _____ __  __ 
 |_   _| | | | \ | | ____|   | ____|  \/  |
   | | | | | |  \| |  _| ____|  _| | |\/| |
   | | | |_| | |\  | |__|____| |___| |  | |
   |_|  \___/|_| \_|_____|   |_____|_|  |_|

  ANYDECISION :: TYPED LLM DECISION RUNTIME (v0.2.0)
  --------------------------------------------------
  [READOUT] Direct Vocab Logit Extraction (Zero Generation)
  [LEVEL 0] Raw Model Distribution Forward Pass
  [LEVEL 1] Option-Order Permutation Debiasing & Ensembles
  [LEVEL 2] Statistical Post-Hoc Calibration & Conformal Sets
  [ABSTAIN] Selective Risk & Epistemic Uncertainty Control
```

<div align="center">

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-100%20passed-brightgreen.svg)](tests/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## Executive Summary & 10 Core Questions

### 1. What problem does this solve?
When developers ask an LLM to make a decision (e.g. *"Is this transaction fraudulent? Answer YES or NO"*), standard pipelines instruct the model to generate text tokens, then use regex, JSON parsing, or prompt engineering to extract the answer. This is slow, non-deterministic, brittle to formatting, and discards the rich probability distribution computed in the final layer of the model. `anydecision` turns open-weight LLMs into **typed decision engines** that extract answers directly from the model's logits, measure uncertainty, debias prompt ordering, and abstain when confidence is insufficient.

### 2. Why not just generate text?
Text generation introduces autoregressive decode latency, grammar hallucinatory drift, and sampling randomness. More importantly, **generation obscures uncertainty**: a model forced to output a token cannot reliably signal when it is 51% vs 99% confident. `anydecision` bypasses generation entirely, evaluating candidates in a single forward pass.

### 3. How are probabilities extracted?
For single tokens, the model computes vocabulary logits $z_k$ at the prompt's termination. Log-softmax over the candidate set $\mathcal{C}$ yields candidate-conditional probabilities:
$$\log P(y_k \mid x, \mathcal{C}) = z_k - \log \sum_{j \in \mathcal{C}} \exp(z_j)$$
`anydecision` explicitly exposes both quantities on every decision:
- `choice_probability`: Candidate-conditional probability $P(y_k \mid x, \mathcal{C})$ normalized over the candidate set.
- `raw_vocab_logprob` / `model_token_probability`: Unconstrained vocabulary log-probability and likelihood $\exp(z_k - \log \sum_{v \in \mathcal{V}} \exp(z_v))$.
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
Via selective prediction policies (`min_confidence=0.85` or `target_error=0.05`), the engine returns `Decision(abstained=True, reason="risk_exceeds_target_error")` whenever posterior risk violates permissible error tolerance. It also supports **conformal prediction sets** with mechanically enforced,
explicitly labeled guarantee tiers (`formal_conformal` only under documented
exchangeability assumptions; otherwise `high_confidence_empirical`,
`heuristic`, or `unavailable`).

### 9. What models and backends are supported?
- **Local Hugging Face Transformers**: Causal LMs (`Qwen`, `Llama`, `Mistral`, `Gemma`, `Phi`).
- **High-throughput vLLM**: GPU serving with PagedAttention and prompt logprob extraction.
- **Mock/Synthetic**: Deterministic, high-speed backend for offline testing, CI/CD, and lightweight demos.

### 10. How much compute is required?
Single forward pass per prompt. Measured on this machine (mock backend, CPU):
L0 ≈ 0.36 ms/decision (benchmark table below); the ViZDoom loop with adaptive
mock decisions ≈ 4.1 ms/decision. Real-model latency must be measured per
model/device/backend deployment — no universal figure is claimed.

---

## Quickstart

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

### Universal Decision API (Just Like von, but with Calibrated Uncertainty)

```python
import anydecision

# 1. Universal One-Liner (evaluates ANY prompt & arbitrary candidate options directly):
decision = anydecision.choose(
    "What is the primary vulnerability in this code snippet?",
    ["SQL Injection", "Server-Side Request Forgery", "Cross-Site Scripting", "Buffer Overflow"]
)

print(decision.answer)                    # "SQL Injection"
print(decision.choice_probability)        # 0.9412 (conditional on candidate set)
print(decision.model_token_probability)   # 0.0381 (unconstrained vocabulary likelihood)
print(decision.choice_margin)             # 0.8824 (margin over second-best choice)
print(decision.predictive_entropy)        # 0.2811 nats
print(decision.abstained)                 # False

# 2. Decision Engine with Expected Utility & Action Policies:
from anydecision import DecisionEngine, Question

engine = DecisionEngine(model="mock")
result = engine.choose(
    "Should this high-priority customer request be escalated?",
    ["yes", "no"],
    actions={"escalate_immediately": 5.0, "auto_resolve": 2.0, "human_review": -1.0}
)

print(result.selected_action)       # "escalate_immediately"
print(result.optimal_action_utility) # 4.82
```

---

## Architecture

```
Prompt + Candidate Definitions
              |
              v
    Tokenization & Prefix Check
              |
              v
  Forward Pass / Vocabulary Logits  (Zero Text Generation)
              |
              v
  Candidate Extraction & Softmax Normalization
              |
              v
  [L1] Permutation Debiasing & Prompt Ensemble Aggregation
              |
              v
  [L2] Statistical Post-Hoc Calibration (T-Scaling / Platt)
              |
              v
  Uncertainty Estimation & Selective Risk Check
              |
              v
  Strongly Typed Decision Object  (Answer or Selective Abstention)
```

---

## Typed Question Interfaces

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

## Selective Prediction & Abstention

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
# Prediction sets with explicit guarantee tiers. 'formal_conformal' is only
# reported under documented exchangeability assumptions with split
# threshold-selection/certification data; otherwise the engine reports
# 'high_confidence_empirical', 'heuristic', or 'unavailable'.
decision = engine.decide(question, level="L2")
print(decision.prediction_set)   # e.g. ['billing', 'technical']
print(decision.guarantee_type)   # e.g. 'formal_conformal'
```

---

## L2 Statistical Calibration & Versioned Artifacts

Fit post-hoc calibration on held-out validation data and save versioned,
tamper-evident artifacts (SHA-256 tamper detection; optional Ed25519 signatures
for authenticity — hashing alone is not authenticity):

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

## Streaming Online Adaptation

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

## Command Line Interface (CLI)

```bash
# 1. Make a single typed decision
anydecision ask -q "Is this account suspicious?" -c yes -c no

# 2. Run calibration benchmark
anydecision benchmark --samples 50 --level L1

# 3. Inspect and verify a calibration artifact
anydecision inspect-artifact artifacts/support_router_head.json

# 4. Start production FastAPI HTTP service
anydecision serve --port 8000

# 5. Launch interactive Linux terminal-based UI (TUI)
anydecision tui

# 6. Launch SYNTHETIC toy-combat simulator (deterministic test env, not real DOOM)
anydecision doom --episodes 5 --difficulty hard

# 6b. Evaluate a policy in the live ViZDoom engine (real game physics)
anydecision vizdoom --scenario basic --episodes 3 --seed 0 --policy anydecision --output out.json

# 6c. Evaluate on a genuine WAD map (requires DOOM.WAD + vizdoom; fails loudly otherwise)
anydecision real-doom --map E1M1 --wad /path/to/DOOM.WAD --episodes 2 --seed 0 --output out.json

# 7. Launch interactive Gradio research demo
anydecision demo --port 7860

# 8. Run head-to-head empirical benchmark: anydecision vs von
anydecision compare-von
```

---

## Interactive Terminal UI (TUI)

Launch the full-screen terminal-based UI with:

```bash
anydecision tui
# or
tune-em-tui
# or
python -m anydecision tui
```

The Terminal UI features:
- **Decision Studio**: Interactive question evaluator with ASCII probability meters, Expected Utility action policies, and layer emergence detection.
- **Economics Benchmark Suite**: Evaluate Quality/Compute, Quality/Dollar, Safe Decisions/sec, and Latency vs Accuracy curves.
- **Layer-Trajectory Inspector**: Depth analysis across 32 transformer layers tracing confidence growth and representation convergence.
- **Active Calibration & Drift Monitor**: High-information candidate selection and real-time distribution drift alerts with adaptive threshold tightening.
- **Selective Conformal Predictor**: Tiered risk control (`formal_conformal` only under documented exchangeability assumptions; otherwise empirical/heuristic tiers).
- **Security & Injection Auditor**: Automated prompt injection test suite verifying boundary quarantine defense.
- **DOOM Arenas**: Synthetic toy-combat simulator (labeled synthetic) plus live ViZDoom-engine policy evaluation with authoritative counters.

---

## Architectural Comparison: anydecision vs wfzyx/von vs AnyJev

The following matrix contrasts `anydecision` against `wfzyx/von` (a non-autoregressive ModernBERT-large 395M model) and standard decision toolkits:

| Capability / Dimension | wfzyx/von | AnyJev | anydecision (This Work) |
|---|---|---|---|
| **Architecture Freedom** | Hard-locked to ModernBERT-large (395M) checkpoint | Causal LLM wrapper | **Model-Agnostic**: Any Causal LLM (Qwen, Llama, Mistral, Gemma, Phi) + Native Non-Autoregressive Bilinear Head (`NonAutoregressiveDecisionHead`) + Zero-Overhead Scorer (`FastOptionScorer`) |
| **Mathematical Framework** | Probability ranking only | Probability extraction | **von Neumann-Morgenstern Expected Utility**: Optimizes $EU(a) = \sum_y P(y \mid x) U(a, y)$ with action cost matrices, risk asymmetry, and regret minimization |
| **Inference Routing** | Static 1-pass only | Manual level selection | **Adaptive Compute Router**: Dynamically routes between L0, L1, L2, and Selective Conformal based on confidence budgets and latency caps |
| **Internal Representation** | Monolithic option-marker cross-attention | Single-layer hidden state | **Layer-Trajectory Tracking & Multi-Layer Fusion**: Traces confidence emergence across all transformer layers; computes decision emergence layer ($L_{emergence}$); compares concatenation/attention/gating heads |
| **Risk Guarantees** | None (no abstention) | Heuristic score threshold | **Selective Conformal Prediction**: Finite-sample statistical risk guarantee ($E[\text{loss} \mid \text{selected}] \le \alpha$) + Hierarchical Bayesian Shrinkage |
| **Active Learning & Drift** | None | Static offline calibration | **Active Calibration & Sequential Drift**: Information-theoretic candidate selection with budget-curve measured savings (no fixed percentage claimed) + CUSUM drift detection |
| **Prompt Injection Defense** | Vulnerable to context corruption | Vulnerable to prompt injection | **Structured Context Isolation**: Structured `DecisionContext` with strict XML quarantine delimiters and non-executable data blocks |
| **Real-Time Gaming Benchmark**| None | None | **ViZDoom integration + synthetic toy combat**: Live-engine policy evaluation with authoritative kill/death counters and scenario-defined victory; mock-backend `basic` runs measure ≈4.1 ms/decision (see benchmark table) |

---

## DOOM Evaluation: Synthetic vs Real Engine (Clearly Separated)

The repository contains two strictly separated things. Synthetic results are
never presented alongside live-engine results without labels.

### Synthetic toy combat (`anydecision doom`) — SYNTHETIC, for tests/CI

`anydecision/games/doom.py` is a seeded synthetic simulator (ASCII encounters,
RNG damage) for deterministic tests, utility tests, policy unit tests, failure
injection, and CI. It is **not** real DOOM and **not** ViZDoom.

```bash
# Run 5 synthetic toy-combat episodes on HARD difficulty
anydecision doom --episodes 5 --difficulty hard

# Or test against synthetic boss encounters
anydecision doom --episodes 3 --difficulty boss
```

### Genuine WAD evaluation (`real-doom`) — REAL map, REAL engine

`anydecision real-doom` validates a genuine `DOOM.WAD`, parses real THINGS-lump
entities (the true initial state), and runs the policy in the **live ViZDoom
engine on that map**. Damage, kills, and pickups come from authoritative game
variables — nothing is fabricated. Without a WAD or without vizdoom it exits
with an actionable error; it never substitutes the synthetic simulator.

```bash
# Evaluate on E1M1 (requires a genuine WAD file)
anydecision real-doom --map E1M1 --wad /path/to/DOOM.WAD --episodes 2 --seed 0 --output e1m1.json

# E2M8 Tower of Babel at Ultra-Violence
anydecision real-doom --map E2M8 --skill 4 --wad /path/to/DOOM.WAD --seed 0 --output e2m8.json
```

Measured run (`benchmarks/results/real_doom_E1M1_seed0.json`; mock backend,
E1M1, 2 episodes, seed 0): WAD census 6 monsters / 21 pickups; the live-engine
run produced 800 decisions, **0 kills, 0 deaths**. The mock policy wanders a
real map ineffectively — reported as-is. Outcomes use the kills-or-survival
proxy, not exit-switch clearance (untracked).

Scorecards report facts only (kills, deaths, items, reward, latencies,
decisions, backend calls, tokens, action distribution, abstentions) with no
subjective ratings. Every benchmark supports `--output run.json` producing a
machine-readable artifact (experiment, timestamp, commit, model, backend, seed,
scenario, episodes, kills, latencies). See `docs/CLAIMS.md` for the
claim-to-evidence matrix.

### ViZDoom integration — REAL engine, REAL policy control

[ViZDoom](https://vizdoom.farama.org/) integration (not external verification —
"ViZDoom integration", never "verified"). The real pipeline is: live ViZDoom
state → observation extraction (`STATE` privileged, `VISION` screen/labels
buffer only, `HYBRID` both) → typed anydecision candidate scoring over tactical
states → `UtilityMatrix` → `Decision.selected_action` → live game action →
authoritative `KILLCOUNT`/`DEATHCOUNT`/`ITEMCOUNT`. The executed action always
equals the selected action (assertion-enforced); no hardcoded belief
distributions; kills never inferred from reward; victory is scenario-defined
(combat maps: kills > 0; collection/navigation: survival).

```bash
# anydecision policy, 3 episodes, deterministic seed, JSON artifact
anydecision vizdoom --scenario basic --episodes 3 --seed 0 --policy anydecision --observation-mode HYBRID --output basic.json

# Fair baselines on identical episodes/seeds/action space
anydecision vizdoom --scenario basic --episodes 3 --seed 0 --policy random --output basic_random.json
anydecision vizdoom --scenario basic --episodes 3 --seed 0 --policy scripted --output basic_scripted.json

# Visual window
anydecision vizdoom --scenario deadly_corridor --render
```

Measured results (`benchmarks/results/vizdoom_basic_<policy>_seed<seed>.json`;
mock backend, `basic`, 3 episodes, HYBRID). With `backend=mock`, beliefs are
hash-based: this validates the control loop and metric plumbing, NOT tactical
skill. Real-model policy comparison is future work.

| policy | seed | won/3 | kills | decisions | mean latency (ms) |
|---|---|---|---|---|---|
| anydecision | 0 | 3 | 3 | 35 | 4.14 |
| anydecision | 1 | 2 | 2 | 79 | 4.18 |
| anydecision | 2 | 2 | 2 | 129 | 4.12 |
| random | 0 | 2 | 2 | 79 | ~0.00 |
| random | 1 | 3 | 3 | 77 | ~0.00 |
| random | 2 | 3 | 3 | 11 | ~0.00 |
| scripted | 0 | 3 | 3 | 17 | ~0.00 |
| scripted | 1 | 3 | 3 | 16 | ~0.00 |
| scripted | 2 | 3 | 3 | 8 | ~0.00 |


---


## FastAPI HTTP Service

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

Public decision endpoints (`/decide`, `/batch`) are rate-limited,
concurrency-bounded, and input-bounded. `/calibrate` is an **admin** endpoint:
it requires `X-API-Key` matching `ANYDECISION_ADMIN_KEY` and is disabled (403)
otherwise. `/model` always reports the active backend and flags `is_mock` —
mock outputs are deterministic test fixtures, never genuine evaluation.

---

## Research Transparency & Limitations

1. **Candidate-Conditional Probabilities vs Global Likelihood**: Log-softmax normalization across a candidate set $\mathcal{C}$ measures relative preference $P(y \mid x, \mathcal{C})$. It sums to 1.0 across the candidate set, regardless of how improbable the choices are in the unconstrained language model space. `anydecision` provides `model_token_probability` ($\exp(z - \log \sum_{v \in \mathcal{V}} \exp(z_v))$) and `predictive_entropy` alongside `choice_probability` so callers never mistake relative ranking for universal confidence.
2. **Statistical Conformal Bounds**: Conformal prediction set coverage and Selective Conformal Risk Control bounds hold under exchangeability between calibration and test data, with calibration sample size $n \ge 20$. When sample sizes are small ($n < 20$) or when empirical risk fallbacks are utilized, `anydecision` reports `guarantee_type="heuristic"` or `"high_confidence_empirical"`, explicitly withholding `formal_conformal`.
3. **Calibration Does Not Guarantee Correctness**: Calibration guarantees empirical frequency over exchangeable validation distributions. It does not prevent errors on out-of-distribution instances.
4. **OOD Diagnostics Are Heuristics**: Our entropy, collapse, and sensitivity metrics serve as observable operational alerts, not formal statistical proofs of distribution shift.

---

## Benchmark Results

Customer-escalation triage across decision levels, measured with
`python benchmarks/run_benchmark.py` (mock backend, 40 train / 40 held-out
test; artifact: `benchmarks/results/benchmark_summary.json`). Mock-backend
numbers validate the calibration machinery, not real-model gains.

| Level | Accuracy | ECE (lower) | Brier | NLL | Mean Latency |
|---|---|---|---|---|---|
| **L0 (Raw)** | 50.0% | 0.4112 | 0.8887 | 1.9227 | 0.36 ms |
| **L1 (Zero-Label)** | 25.0% | 0.3502 | 0.6904 | 0.8944 | 0.77 ms |
| **L2 (Calibrated)** | 25.0% | **0.2604** | **0.5172** | **0.7104** | 0.90 ms |

In the 50-task agent workflow (`benchmarks/results/agent_eval_summary.json`),
L2 with selective abstention recorded a 0.0% catastrophic rate vs 32.0% for
L0/raw — by abstaining on all 50 tasks. Coverage/risk trade-off, not free
safety: see the risk-coverage curve in `benchmarks/results/risk_coverage.png`.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
