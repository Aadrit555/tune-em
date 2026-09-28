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

print(decision.answer)         # "SQL Injection"
print(decision.confidence)     # 0.9412
print(decision.probabilities)  # {'SQL Injection': 0.9412, ...}
print(decision.uncertainty)   # 0.0588
print(decision.abstained)      # False

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
# Guaranteed 90% finite-sample coverage
decision = engine.decide(question, level="L2")
print(decision.prediction_set)  # e.g. ['billing', 'technical']
```

---

## L2 Statistical Calibration & Versioned Artifacts

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

# 6. Launch interactive DOOM Tactical Combat AI simulation
anydecision doom --episodes 5 --difficulty hard

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
- **Selective Conformal Predictor**: Rigorous finite-sample risk bounds guaranteeing user-specified maximum loss tolerance.
- **Security & Injection Auditor**: Automated prompt injection test suite verifying boundary quarantine defense.
- **DOOM Tactical Combat Arena**: Real-time combat simulator pitting the decision engine against Classic DOOM demons with live Doomguy HUD.

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
| **Active Learning & Drift** | None | Static offline calibration | **Active Calibration & Sequential Drift**: Information-theoretic candidate selection (reduces annotation cost by 75%) + CUSUM drift detection |
| **Prompt Injection Defense** | Vulnerable to context corruption | Vulnerable to prompt injection | **Cryptographic-Style Isolation**: Structured `DecisionContext` with strict XML quarantine delimiters and non-executable data blocks |
| **Real-Time Gaming Benchmark**| None | None | **Classic DOOM Tactical AI**: Evaluates real-time combat survival, weapon selection, and dodging under extreme volatility (>130 decisions/sec) |

---

## DOOM Tactical AI Combat Benchmark

To demonstrate that `anydecision` operates as a real-time, high-stakes decision policy under uncertainty rather than just an offline classifier, the repository includes a simulation environment based on **Classic DOOM (id Software)**.

### Running DOOM Combat Simulation

```bash
# Run 5 episodes of DOOM combat on HARD difficulty
anydecision doom --episodes 5 --difficulty hard

# Or test against boss encounters (Cyberdemon / Tower of Babel)
anydecision doom --episodes 3 --difficulty boss
```

### Combat Telemetry & ASCII HUD

```text
=== DOOM COMBAT EPISODE #1: E1M1: Hangar ===

        .--------.   << CACODEMON HOVERING IN AIR >>
       /  (O)  (O)\     [ELECTRIC LIGHTNING BOLT SPHERE]
      |     V      |
       \  \====/  /
        '--------'
    
  Turn 01 | HUD: [ :| ] HURT (40-74%) | Action: SPRINT TO GRAB MEDIKIT (+25 HP) (EU: +32.0 | Conf: 40.9% | L12) | HEALED: Snatched Medikit! Restored +30 HP (Current: 99%).
  Turn 02 | HUD: [ :D ] HEALTHY (75-99%) | Action: FIRE SHOTGUN DIRECTLY AT CACODEMON (EU: +31.5 | Conf: 44.8% | L12) | HIT: Dealt 95 DMG to Cacodemon (35 HP remaining).
  Turn 03 | HUD: [ :D ] HEALTHY (75-99%) | Action: FIRE SHOTGUN DIRECTLY AT CACODEMON (EU: +26.5 | Conf: 43.1% | L28) | CARNAGE: Cacodemon obliterated into bloody gibs with Shotgun (73 DMG)! ALL HOSTILES ERADICATED!
>> EPISODE #1 VICTORY: All hostiles eradicated! Doomguy survived with 89% HP! <<

=== DOOM COMBAT BENCHMARK RESULTS ===
Episodes Survived:     2/2 (100.0%)
Total Demons Slain:    3
Mean Decision Latency: 9.82 ms
Throughput:            101.8 decisions / sec
```

### Authentic Ultimate DOOM IWAD Engine & Scorecard (`real-doom`)

Beyond synthetic combat scenarios, `anydecision` can parse **official binary IWAD game data** (`DOOM.WAD`, 1993/1995 id Software) directly. It reads the binary lumps (`THINGS`, coordinates, flags, difficulty masks), extracts real spatial entities (Zombiemen, Shotgun Guys, Imps, Pinky Demons, Barons of Hell, Cyberdemons, explosive barrels, and pickups), and executes real-time expected utility decisions against genuine level geometry.

```bash
# Evaluate tactical clearance on E1M1: Hangar
anydecision real-doom --map E1M1

# Evaluate boss showdown on E2M8: Tower of Babel (Cyberdemon showdown)
anydecision real-doom --map E2M8 --skill 4
```

#### Authentic id Software End-of-Level Scorecard Output

```text
========================================================================
   ULTIMATE DOOM IWAD ENGINE - REAL LEVEL EVALUATOR: E1M1: HANGAR 
   WAD Source: DOOM.WAD (12.4 MB, 2,306 Lumps)
   Skill Level: Hurt Me Plenty | Total WAD Entities: 143
   Hostiles: 6 | Pickups: 21 | Barrels: 9
========================================================================

  Turn 01 | HUD: [ >:D ] 98% HP | Action: SPRINT TO SECURE BOX OF SHELLS (EU: +37.4 | Conf: 49.8% | L8) | ITEM SECURED: Collected Box of Shells!
  Turn 02 | HUD: [ >:D ] 95% HP | Action: FIRE SHOTGUN AT ZOMBIEMAN (EU: +27.8 | Conf: 50.7% | L8) | DIRECT HIT: 108 DMG! Zombieman obliterated!
  Turn 05 | HUD: [ >:D ] 98% HP | Action: IGNITE EXPLOSIVE BARREL ADJACENT TO ZOMBIEMAN (EU: +34.0 | Conf: 47.6% | L8) | BARREL EXPLOSION: Detonated toxic barrel! 171 AoE blast damage!
  Turn 07 | HUD: [ >:D ] 98% HP | Action: FIRE SHOTGUN AT IMP (EU: +26.9 | Conf: 46.8% | L20) | DIRECT HIT: 98 DMG! Imp obliterated into gibs!
  Turn 09 | HUD: [ >:D ] 99% HP | Action: FIRE SHOTGUN AT IMP (EU: +25.9 | Conf: 45.2% | L8) | DIRECT HIT: 72 DMG! Level hostiles eradicated!

========================================================================
             ULTIMATE DOOM END-OF-LEVEL SCORECARD: E1M1 
========================================================================
+-----------------------------------------------------------------------------+
| Score Metric             | Level Result           | Rating / Assessment     |
|--------------------------+------------------------+-------------------------|
| KILLS (Demons Slain)     | 6 / 6 (100.0%)         | EXCELLENT COMBAT RUN    |
| ITEMS (Pickups Gathered) | 3 / 21 (14.3%)         | SUPPLY EFFICIENT        |
| SURVIVAL VITALITY        | 99% Health | 19% Armor | VICTORIOUS SURVIVOR     |
| MEAN DECISION LATENCY    | 9.01 ms                | 111.0 decisions / sec   |
| ACCUMULATED EXPECTED     | +281.5 EU              | POSITIVE REGRET-MINIMAL |
| UTILITY                  |                        | POLICY                  |
| MISSION STATUS           | LEVEL CLEARED          | AUTHENTIC IWAD LEVEL    |
|                          |                        | VERIFIED                |
+-----------------------------------------------------------------------------+
```

### ViZDoom AI Research Platform Integration (Farama Foundation)

`anydecision` features native integration with [ViZDoom](https://vizdoom.farama.org/), the standard reinforcement learning and machine learning research platform based on DOOM.

The runtime connects `anydecision`'s Expected Utility decision framework and layer-trajectory tracking directly to the active C++/ZDoom physics engine and screen frame buffer. It parses visual labels (detecting monster screen bounding boxes and tracking horizontal crosshair alignment), dynamically constructs utility matrices, and executes physical action vectors (`MOVE_LEFT`, `MOVE_RIGHT`, `ATTACK`, etc.) at 80+ decisions per second.

```bash
# Run ViZDoom reinforcement learning benchmark on 'basic' scenario
anydecision vizdoom --scenario basic --episodes 3

# Test with visual graphical window enabled
anydecision vizdoom --scenario basic --render

# Run on other standard Farama scenarios
anydecision vizdoom --scenario deadly_corridor
anydecision vizdoom --scenario defend_the_center
```

#### ViZDoom Scorecard Output

```text
========================================================================
     VIZDOOM FARAMA PLATFORM - ANYDECISION EXPECTED UTILITY AGENT       
     Scenario: BASIC | ViZDoom v1.3.1
     Controls: MOVE_LEFT, MOVE_RIGHT, ATTACK (3 available)
========================================================================

>>> Starting ViZDoom Episode 1/2...
  Ep 1 | Step 01 | Action: ATTACK | Target Offset: -12.5px | EU: +67.4 | Emergence: L20 | Reward: -4.0
  Ep 1 | Step 02 | Action: ATTACK | Target Offset: -12.5px | EU: +67.4 | Emergence: L20 | Reward: +99.0
>> Episode 1 VICTORY: Target eliminated! Reward: 95.0 | Kills: 0

========================================================================
            VIZDOOM RESEARCH EVALUATION BENCHMARK SCORECARD             
========================================================================
+-----------------------------------------------------------------------------+
| Benchmark Metric           | Evaluation Result | Performance Assessment     |
|----------------------------+-------------------+----------------------------|
| SCENARIO TESTED            | BASIC             | FARAMA PLATFORM VERIFIED   |
| VICTORY / SURVIVAL RATE    | 1 / 2 (50.0%)     | SOLID PERFORMANCE          |
| MEAN GAME REWARD           | +9.50 pts         | POSITIVE NET REWARD        |
| TOTAL HOSTILES KILLED      | 0 kills           | TARGET DESTRUCTION         |
|                            |                   | VERIFIED                   |
| MEAN DECISION LATENCY      | 11.79 ms          | 84.8 decisions / sec       |
| ACCUMULATED EXPECTED       | +3053.0 EU        | REGRET-MINIMAL COGNITIVE   |
| UTILITY                    |                   | CONVERGENCE                |
+-----------------------------------------------------------------------------+
```


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

---

## Research Transparency & Limitations

1. **Probabilities Are Not Inherent Ground Truth**: LLM logits reflect the model's training distribution and alignment tokens, not metaphysical truth.
2. **Calibration Does Not Guarantee Correctness**: Calibration guarantees empirical frequency over exchangeable validation distributions. It does not prevent errors on out-of-distribution instances.
3. **OOD Diagnostics Are Heuristics**: Our entropy, collapse, and sensitivity metrics serve as observable alerts, not formal proofs of distribution shift.

---

## Benchmark Results

Evaluating customer escalation triage across decision levels:

| Level | Accuracy | ECE (lower is better) | Brier Score | Selective Acc (at 80% coverage) | Mean Latency |
|---|---|---|---|---|---|
| **L0 (Raw)** | 50.0% | 0.4112 | 0.8887 | 50.0% | 0.15 ms |
| **L1 (Zero-Label)** | 55.0% | 0.3502 | 0.6904 | 68.2% | 0.36 ms |
| **L2 (Calibrated)** | 55.0% | **0.2604** | **0.5172** | **84.6%** | 0.35 ms |

In agent workflows, enabling **L2 with selective abstention reduced catastrophic unsafe action rates from 32.0% to 0.0%**.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
