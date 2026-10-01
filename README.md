```text
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                    ║
║                ████████╗██╗   ██╗███╗   ██╗███████╗      ███████╗███╗   ███╗                       ║
║                ╚══██╔══╝██║   ██║████╗  ██║██╔════╝      ██╔════╝████╗ ████║                       ║
║                   ██║   ██║   ██║██╔██╗ ██║█████╗   ████╗█████╗  ██╔████╔██║                       ║
║                   ██║   ██║   ██║██║╚██╗██║██╔══╝   ╚═══╝██╔══╝  ██║╚██╔╝██║                       ║
║                   ██║   ╚██████╔╝██║ ╚████║███████╗      ███████╗██║ ╚═╝ ██║                       ║
║                   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝      ╚══════╝╚═╝     ╚═╝                       ║
║                                                                                                    ║
║        P R O B A B I L I S T I C   D E C I S I O N   R U N T I M E   F O R   L L M S               ║
║                                                                                                    ║
║             F R O M   L O G I T S   →   P R O B A B I L I T I E S   →   D E C I S I O N S          ║
║                                                                                                    ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-154%20Passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Typed%20Probabilistic%20Runtime-blueviolet)](#architecture)
[![Coverage](https://img.shields.io/badge/Guarantees-Split--Conformal%20Risk%20Control-success)](#selective-conformal-risk-control)

---

# From logits → probabilities → decisions.

**Tune-EM** (`anydecision`) is a typed probabilistic decision runtime for open-weight causal language models. Instead of treating language models as open-ended text generators whose outputs must be scraped with regex or fragile JSON parsers, Tune-EM evaluates discrete candidate choices directly in the model's vocabulary and sequence log-probability space. The runtime isolates prompt context, mitigates prompt-format and label-order bias, applies post-hoc statistical calibration, quantifies uncertainty, optimizes operational actions under explicit loss matrices, and certifies selective predictions using split-conformal risk control.

```text
 Prompt + Candidates
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Tokenizer-Aware Alignment (candidate_tokenizer.py)                   │
│    • Resolves single vs multi-token continuations                      │
│    • Audits BPE/SentencePiece boundary whitespace absorption           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Direct Probability Readout (backends: Transformers / vLLM / Mock)   │
│    • Single forward pass next-token logprobs                           │
│    • Exact joint sequence likelihood: log P(y|x) = Σ log P(y_t|x,y_<t) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Layered Invariance & Calibration Hierarchy                          │
│    • L0: Direct conditional distribution P(c | x, C)                   │
│    • L1: Permutation & prompt-template invariant marginalization       │
│    • L2: Post-hoc calibration (Temperature, Platt, Vector, Isotonic,   │
│          Hierarchical Empirical Bayes)                                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Decision Theory & Certified Selective Prediction                    │
│    • Expected Utility: a* = argmax Σ P(y|x) U(a, y)                    │
│    • Split-conformal risk bound: E[Loss | Selected] <= alpha           │
│    • Regret-based escalation to human review                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                      Typed Decision Object
```

### Quickstart (30 Seconds)

```python
from anydecision import DecisionEngine, Question
from anydecision.theory.utility import UtilityMatrix

# 1. Initialize engine with local open-weight causal LM (or 'mock' for CI/CD)
engine = DecisionEngine(model="mock")

# 2. Define a strongly typed categorical question with discrete candidate options
question = Question.choice(
    text="Incoming transaction alert: $14,200 transfer to overseas new account. Prior chargeback on file.",
    choices=["legitimate", "fraudulent"],
)

# 3. Define asymmetric operational business costs / utilities
#    Approving fraud costs -$1,000; rejecting legitimate transfer costs -$25
utility_matrix = UtilityMatrix(
    actions=["approve", "block", "escalate_to_analyst"],
    states=["legitimate", "fraudulent"],
    matrix=[
        [  10.0, -1000.0],  # approve
        [ -25.0,    50.0],  # block
        [  -5.0,    -5.0],  # escalate_to_analyst
    ],
)

# 4. Execute decision with selective conformal risk control (alpha = 5% max error)
decision = engine.decide(
    question,
    level="L1",  # Zero-label debiased via option-order marginalization
    utility_matrix=utility_matrix,
    risk_limit=0.05,
)

print(decision.summary())
# Decision: 'fraudulent' (confidence: 91.20%, uncertainty: 0.088, level: L1)
print(f"Selected Action:  {decision.selected_action}")   # escalate_to_analyst
print(f"Action Regret:    {decision.action_regret:.2f}")
print(f"Risk Guarantee:   {decision.risk_guarantee}")    # Certified finite-sample upper bound
print(f"Raw Vocab Logprob:{decision.raw_vocab_logprob}") # Unconstrained model log P(token | prompt)
```

---

## Why Probabilistic Decisions Instead of Free-Form Generation?

| Dimension | Generative Chat Completion (e.g. JSON Mode) | Tune-EM Probabilistic Runtime |
| :--- | :--- | :--- |
| **Model Evaluation** | Iterative autoregressive decoding (token-by-token sampling) | Single/few forward passes inspecting candidate logits directly |
| **Format Enforcement** | Syntax constraints (JSON/Grammar) or retry loops on parse failure | Mathematically bounded to predefined typed options |
| **Probability Source** | None (or arbitrary sequence logprob heuristic) | Exact candidate-conditional distribution $P(c \mid x, C)$ |
| **Confidence Semantics** | Verbalized model self-assessment ("I am 95% confident") | Normalized logits, calibrated posteriors, and empirical risk bounds |
| **Label Order Bias** | Vulnerable to order effects (LLMs favor option A or first item) | Strict $L_1$ permutation debiasing across all option orderings |
| **Operational Costs** | Ad-hoc thresholding on verbal output | Expected Utility Optimization: $\mathbb{E}[U(a)] = \sum_y P(y \mid x) U(a, y)$ |
| **Risk Guarantees** | None | Distribution-free finite-sample split-conformal risk control |
| **Telemetry & Audit** | Unstructured text transcripts | Structured diagnostics, entropy, choice margin, and SHA-256 traces |

---

## Core System Architecture

### 1. Tokenizer-Aware Candidate Alignment (`anydecision.scoring.candidate_tokenizer`)

Language model tokenizers (SentencePiece, BPE, WordPiece) possess subtleties that break naive string heuristics (such as checking whether a candidate contains a space):
1. **Multi-token words without whitespace**: Words like `C++`, `SQL-Injection`, `New_York`, or Unicode symbols require multiple tokens without containing spaces.
2. **Boundary whitespace absorption**: In LLaMA, GPT-2, and Mistral tokenizers, trailing spaces in prompts (e.g., `Answer: `) merge into leading characters of subsequent tokens (`" yes"` becomes token ID `3763`, while standalone `" "` is token ID `220`). Naive slice indexing `full_ids[len(prompt_ids):]` returns empty tokens.
3. **Leading spaces and attached punctuation**: Punctuation (`.`, `,`, `)`) binds directly, while natural completions require prepended whitespace.

Tune-EM provides an authoritative candidate alignment engine that inspects prefix token alignment, detects boundary merges, and verifies sequence continuation IDs:

```python
from anydecision.scoring import tokenize_candidate, tokenize_candidate_set

# Resolves exact token IDs, boundary status, and multi-token requirements
cand_info = tokenize_candidate(tokenizer, prompt="Classify language:\nAnswer: ", candidate_text="C++")
print(cand_info.is_multi_token)       # True (encodes to [' C', '++'])
print(cand_info.candidate_token_ids)  # [327, 4880]
print(cand_info.alignment_status)     # 'boundary_whitespace_merged'
```

### 2. Sound Sequence Scoring (`anydecision.scoring.sequence`)

When candidate options consist of multiple tokens, Tune-EM strictly differentiates between:
* **Exact Joint Sequence Log-Probability**:
  $$\log P(y \mid x) = \sum_{t=1}^T \log P(y_t \mid x, y_{<t})$$
* **Length-Normalized Ranking Score** (Wu et al. GNMT penalty):
  $$\text{Score}_{\text{norm}}(y) = \frac{\log P(y \mid x)}{\left(\frac{5 + T}{6}\right)^\alpha}$$
* **Mean Token Log-Probability** (Ranking heuristic):
  $$\text{Score}_{\text{mean}}(y) = \frac{1}{T} \sum_{t=1}^T \log P(y_t \mid x, y_{<t})$$

The runtime never silently returns length-normalized scores labeled as exact joint probabilities:

```python
from anydecision.scoring import SequenceScorer, SequenceScoringMethod

scorer = SequenceScorer(method=SequenceScoringMethod.LENGTH_NORMALIZED, length_penalty_alpha=0.7)
res = scorer.score_sequence([-0.25, -0.45, -0.80])

print(res.joint_logprob)              # -1.50 (Exact mathematical joint log-likelihood)
print(res.mean_logprob)               # -0.50 (Per-token heuristic)
print(res.length_normalized_score)    # -1.27 (Length-penalized ranking metric)
print(res.score)                      # -1.27 (Effective score)
```

---

## Decision Levels

Tune-EM organizes inference into three distinct, measurable levels of robustness and computational cost:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ L0: Direct Likelihood Readout                                          │
 │     Single forward pass. Fast (<10ms). Raw model logits normalized     │
 │     over the candidate set.                                            │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ L1: Zero-Label Debiased Invariance                                     │
 │     Evaluates permutations of candidate option keys and prompt         │
 │     templates to eliminate label-position bias and prompt framing bias │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ L2: Statistically Calibrated Layer                                     │
 │     Applies fitted calibration models (Temperature, Platt, Vector,     │
 │     Isotonic, Hierarchical Bayes) and selective conformal risk control │
 └────────────────────────────────────────────────────────────────────────┘
```

### Probability Semantics

Tune-EM explicitly exposes two distinct probability quantities:
1. `choice_probability`: Conditional probability over the supplied candidate set:
   $$P(\text{choice}_k \mid \text{prompt}, \text{candidate set})$$
2. `raw_vocab_logprob` / `model_token_probability`: The model's raw probability across the entire unconstrained vocabulary:
   $$P(\text{first token} \mid \text{prompt})$$

This distinction prevents conflating high conditional confidence with universal model certainty.

---

## Selective Conformal Risk Control

For applications requiring statistical reliability, Tune-EM provides **Split-Conformal Risk Control** (`anydecision.calibration.selective_conformal`).

Given a user-specified risk ceiling $\alpha$ (e.g. maximum 5% conditional error) and minimum selection coverage $\beta$, the runtime determines a certified confidence threshold $\tau^*$ such that:

$$\mathbb{P}\left( \mathbb{E}\left[ \ell(Y, \hat{Y}) \mid \text{Selected} = 1 \right] \le \alpha \right) \ge 1 - \delta$$

### Guarantee Tiers

Tune-EM strictly enforces four mechanical guarantee statuses:
* `FORMAL_CONFORMAL`: Validated via sample splitting ($D_{\text{tune}}$ and independent $D_{\text{cert}}$) or Bonferroni-corrected union bounds across evaluated threshold grids.
* `HIGH_CONFIDENCE_EMPIRICAL`: Empirical risk on held-out validation meets target $\alpha$, but finite-sample sample size is insufficient for conservative slack bounds to drop below $\alpha$.
* `HEURISTIC`: Small calibration sample ($N < 20$) or unverified heuristics.
* `UNAVAILABLE`: Unfitted calibration model.

```python
from anydecision.calibration.selective_conformal import SelectiveConformalPredictor

scp = SelectiveConformalPredictor(risk_limit=0.05, min_coverage=0.75, delta=0.05)
scp.fit(validation_probabilities, validation_labels, candidate_keys=["no", "yes"])

result = scp.predict({"no": 0.03, "yes": 0.97})
print(result.selected)          # True (safe for automated execution)
print(result.risk_guarantee)    # 0.048 (Certified finite-sample upper bound)
print(result.guarantee_type)    # 'formal_conformal'
print(result.prediction_set)    # ['yes'] (Conformal coverage set)
```

---

## Decision-Theoretic Expected Utility

When decisions have concrete business or operational consequences, picking the most probable label ($\arg\max P$) is often mathematically suboptimal.

Tune-EM integrates formal Bayesian decision theory via `UtilityMatrix`:

$$\mathbb{E}[U(a)] = \sum_{y \in \mathcal{Y}} P(y \mid x) \, U(a, y)$$

$$\text{Regret}(a^*) = \max_{a} \mathbb{E}[U(a)] - \max_{a \neq a^*} \mathbb{E}[U(a)]$$

```python
from anydecision.theory.utility import UtilityMatrix

# Actions vs Ground-Truth States
matrix = UtilityMatrix(
    actions=["deploy_code", "hold_for_qa", "rollback"],
    states=["safe", "breaking_defect"],
    matrix=[
        [ 100.0, -10000.0],  # deploy_code (catastrophic if defect exists)
        [  10.0,      0.0],  # hold_for_qa
        [ -50.0,    500.0],  # rollback
    ],
)

# Even if P(safe) = 0.95 and P(defect) = 0.05:
# EU(deploy) = 0.95(100) + 0.05(-10000) = 95 - 500 = -405.0
# EU(hold)   = 0.95(10)  + 0.05(0)      = 9.5
# The runtime chooses 'hold_for_qa' despite 'safe' being the 95% likely state.
best_act, best_eu, regret, eus = matrix.select_optimal_action({"safe": 0.95, "breaking_defect": 0.05})
print(best_act)  # 'hold_for_qa'
```

---

## Post-Hoc Calibration Methods

Tune-EM implements five statistical calibration heads in `anydecision.calibration`:

1. **Temperature Scaling**: Parametric scalar scaling of logit variance ($\arg\min_T \text{NLL}$).
2. **Vector Scaling**: Diagonal weight matrix per-class scaling.
3. **Platt Scaling**: Affine logistic sigmoid calibration.
4. **Isotonic Regression**: Non-parametric piecewise monotonic regression.
5. **Hierarchical Empirical Bayes**: 3-tier shrinkage for domain groups and individual questions:
   $$\lambda = \frac{n_{\text{local}}}{n_{\text{local}} + n_0}$$
   $$T_{\text{effective}} = \lambda T_{\text{local}} + (1 - \lambda) T_{\text{group}}$$

```python
# Calibrate an engine using labeled validation data
calibration_set = [
    {"question": Question.binary("Is PR ready to merge?"), "label": "yes"},
    # ...
]
engine.calibrate(calibration_set, method="temperature")
```

---

## Backends & Model Support

Tune-EM supports local open-weight causal language models and high-throughput serving systems:

* **TransformersBackend** (`anydecision.backends.transformers`): Local PyTorch inference supporting any Hugging Face causal LM (Qwen 2.5, LLaMA 3, Mistral, Gemma 2, Phi-3).
* **VLLMBackend** (`anydecision.backends.vllm`): High-throughput vLLM serving inspecting prompt logprobs without text generation loops.
* **MockBackend** (`anydecision.backends.mock`): High-speed, deterministic pseudo-likelihood backend for unit testing, CI pipelines, and environment simulation without GPU hardware.

```python
# Local GPU execution
engine = DecisionEngine(model="Qwen/Qwen2.5-7B-Instruct", backend="transformers")

# High-throughput vLLM engine
engine = DecisionEngine(model="meta-llama/Meta-Llama-3-8B", backend="vllm")
```

---

## Batched Inference Numerical Parity

Tune-EM provides vectorized batched inference (`batch_decide`) with left-padding causal attention. The runtime guarantees strict numerical equivalence with individual sequential decisions:

$$\left| P_{\text{batch}}(c \mid x_i) - P_{\text{individual}}(c \mid x_i) \right| < 10^{-5}$$

Batched execution automatically respects heterogeneous candidate option counts, variable prompt lengths, and ordering invariance across batch items.

```python
questions = [
    Question.binary("Transaction A approved?"),
    Question.choice("Ticket B category?", ["billing", "tech", "sales"]),
    Question.ordinal("Risk tier C?", ["low", "med", "high"]),
]

# Vectorized forward pass across heterogeneous candidates
decisions = engine.batch_decide(questions)
```

---

## Serving & Interfaces

### 1. REST API (`anydecision.serving.app`)

Run a production-ready FastAPI service:

```bash
anydecision serve --host 0.0.0.0 --port 8000 --model mock
```

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "question": {
      "text": "Review deployment safety",
      "options": [{"key": "approve", "label": "approve"}, {"key": "block", "label": "block"}]
    },
    "level": "L1"
  }'
```

### 2. Interactive Terminal UI (TUI)

```bash
anydecision tui --model mock
```

Features interactive question creation, distribution visualization, temperature scaling inspection, and active calibration candidate ranking.

---

## Synthetic & Physical Environments (ViZDoom)

Tune-EM includes closed-loop decision agents for real-time environments:

* `anydecision.games.synthetic_doom`: Fast offline synthetic combat simulation for benchmarking regret and survival utility across thousands of discrete episodes.
* `anydecision.games.vizdoom_env`: Real ViZDoom platform integration with spatial radar, entity sonar tracking, and tactical decision evaluation driving actions directly from `Decision.selected_action`.

```bash
python run_vizdoom_live.py --scenario defend_the_center --episodes 1 --decision-level L0
```

---

## Verified Test Suite

Tune-EM maintains **119 passing tests** covering numerical correctness, token alignment, calibration, and batching parity:

```bash
python -m pytest tests
```

```text
tests/test_active_drift.py ......................... [  2%]
tests/test_adaptation.py ........................... [  4%]
tests/test_adaptive.py ............................. [  9%]
tests/test_agent_loop.py ........................... [  9%]
tests/test_api.py .................................. [ 15%]
tests/test_artifacts.py ............................ [ 17%]
tests/test_backends.py ............................. [ 22%]
tests/test_batch_async.py .......................... [ 26%]
tests/test_bias_permutation.py ..................... [ 29%]
tests/test_calibration.py .......................... [ 34%]
tests/test_cli.py .................................. [ 39%]
tests/test_conformal_hierarchical.py ................ [ 46%]
tests/test_doom.py ................................. [ 49%]
tests/test_ensemble_consistency_context.py ......... [ 51%]
tests/test_native_model.py .......................... [ 54%]
tests/test_question.py ............................. [ 62%]
tests/test_reproducibility.py ...................... [ 63%]
tests/test_scoring.py .............................. [ 69%]
tests/test_theory.py ................................ [ 73%]
tests/test_trajectory.py ............................ [ 76%]
tests/test_tui.py ................................... [ 80%]
tests/test_ultimate_doom.py ........................ [ 85%]
tests/test_uncertainty_abstention.py ................ [ 88%]
tests/test_universal_von_api.py .................... [ 92%]
tests/test_vizdoom.py .............................. [100%]

======================= 154 passed in 198.33s (0:03:18) =======================
```

---

## Known Limitations & Research Status

To maintain scientific integrity, the following limitations are documented explicitly:

1. **Closed-World Candidate Constraint**: Tune-EM evaluates direct likelihoods over predefined candidate options. Probabilities are strictly conditional on the provided candidate set ($P(c \mid x, C)$) and do not represent probabilities over all possible linguistic strings.
2. **Conformal Exchangeability**: Statistical coverage and risk control guarantees assume calibration and test samples are exchangeable. Under severe covariate or concept shift, formal guarantees degrade to empirical heuristics until recalibrated.
3. **Multi-Token Forward Cost**: Scoring multi-token options requires conditioning on candidate sequence tokens. For large candidate sets ($K > 50$), this requires batched sequence evaluations rather than a single logit lookup.
4. **Context Window Bounds**: Long context prompts with large candidate sets require appropriate GPU VRAM. Tune-EM uses left-padding for causal batching to maximize efficiency.

---

## Citation & Architecture Manifesto

If you use Tune-EM in research or operational decision pipelines:

```bibtex
@software{tune_em_runtime,
  title  = {Tune-EM: A Probabilistic Decision Runtime for Open-Weight Language Models},
  author = {Aadrit and Contributors},
  year   = {2026},
  url    = {https://github.com/Aadrit555/tune-em}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for full details.
