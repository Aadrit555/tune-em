# Repository Feature Inventory and Baseline Audit

**Target Revision**: `db2b5c1`  
**Date**: 2026-09-29  
**Repository**: [Aadrit555/tune-em](https://github.com/Aadrit555/tune-em)  
**Package**: `anydecision` (v0.2.0)

---

## Executive Audit Verdict

`anydecision` presents an ambitious, highly structured vision: a typed, non-autoregressive decision runtime extracting decisions directly from model probability distributions rather than generative decoding. However, a rigorous code, mathematical, and benchmark audit reveals significant disparities between marketing claims and actual source implementations:

1. **ViZDoom & Decision Policy Disconnect**: ViZDoom environment physics and radar were real, but the policy evaluation in `vizdoom_env.py` discarded the engine's `Decision` output and instead selected actions using manually crafted probability vectors.
2. **Tokenization Boundary Assumptions**: Candidate single- vs multi-token classification and token extraction used whitespace heuristics (`text.strip()`) rather than prompt-context tokenizer awareness.
3. **vLLM Top-K Logprob Approximation**: When candidate tokens were missing from the top-k returned by vLLM, the code fell back to synthetic values (`lowest_lp - 1.0`), which was uncalibrated and not exposed as an approximation.
4. **Selective Conformal Guarantees**: `SelectiveConformalPredictor` performed threshold searching over 50 candidate $\tau$ values on the exact same calibration sample without sample splitting or union-bound correction, yet labeled results as `formal_conformal`.
5. **Utility Matrix Semantic Inferences**: `UtilityMatrix.from_action_costs` relied on `act.lower() == state.lower()`, and `standard_classification_costs` hardcoded false-negative costs to `classes[0]`.
6. **Active Calibration Claims**: 75% annotation reduction claims in the README were marketed without empirical label-budget curve validation against random sampling.
7. **FastAPI Hardening**: Serving endpoints lacked input bounds, rate limits, and authentication on the mutable `/calibrate` endpoint, and leaked raw exception strings.
8. **Synthetic vs Real DOOM Distinction**: The synthetic text simulator (`doom.py`) was presented in documentation alongside genuine IWAD binary data and live ViZDoom without clear isolation.

---

## Detailed Feature Matrix

| Module / Feature | Implemented? | Actually Invoked? | Used in Final Decision? | Tested? | Mathematically Correct? | Reproducible? | README Claim Stronger than Evidence? |
|---|---|---|---|---|---|---|---|
| **Core DecisionEngine (`engine.py`)** | Yes | Yes | Yes | Yes | Mostly (requires strict probability distinction) | Yes | Mildly (claimed zero-text overhead universally) |
| **Question Types (`question.py`)** | Yes | Yes | Yes | Yes | Yes (Choice, Ordinal, Numeric, Binary, Multi-Choice) | Yes | No |
| **Multi-Choice Independent Bernoulli (`engine.py:170`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Decision Trace (`types.py`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Transformers Backend (`transformers.py`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **vLLM Backend (`vllm.py`)** | Yes | Optional | Yes | Partial | Partial (`lowest_lp - 1.0` heuristic on top-k miss) | Requires GPU | Yes (claimed exact parity) |
| **Single-Token Logprobs (`token.py`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Multi-Token Sequence Scoring (`sequence.py`)** | Yes | Yes | Yes | Yes | Needs explicit joint vs ranking score distinction | Yes | Mildly (GNMT length penalty is ranking score, not joint probability) |
| **Permutation Debiasing L1 (`bias/permutation.py`)** | Yes | Yes | Yes | Yes | Yes (Total Variation distance metric) | Yes | No |
| **Prompt Ensemble Templates (`bias/templates.py`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Temperature / Platt / Vector / Isotonic (`calibration/`)** | Yes | Yes | Yes | Yes | Yes | Yes | Mildly (evaluation split discipline needed) |
| **Hierarchical Calibration (`calibration/hierarchical.py`)** | Yes | Yes | Yes | Yes | Yes (Empirical Bayes shrinkage) | Yes | No |
| **Conformal Prediction Sets (`calibration/conformal.py`)** | Yes | Yes | Yes | Yes | Yes (Standard split conformal quantile) | Yes | No |
| **Selective Conformal Predictor (`calibration/selective_conformal.py`)** | Yes | Yes | Yes | Yes | Defective (searches thresholds on same data without split or correction) | Yes | **Yes (Claimed formal finite-sample guarantee)** |
| **Active Calibration (`calibration/active.py`)** | Yes | Yes | Yes | Yes | Yes (entropy/margin selection heuristics) | Yes | **Yes (Claimed 75% annotation reduction without budget curve)** |
| **Sequential Drift Detection (`calibration/drift.py`)** | Yes | Yes | Yes | Yes | Yes (Page-Hinkley & CUSUM) | Yes | No |
| **Online Adaptation (`adaptation/online.py`)** | Yes | Yes | Yes | Yes | Yes (Exponential moving average) | Yes | No |
| **Abstention Controller (`uncertainty/abstention.py`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **OOD / Distribution Shift (`uncertainty/ood.py`)** | Yes | Yes | Yes | Yes | Heuristic signals (entropy, collapse, instability) | Yes | **Yes (Marketed as OOD Detector without AUROC/FPR95 benchmarks)** |
| **Utility Matrix (`theory/utility.py`)** | Yes | Yes | Yes | Yes | Flawed (`act.lower() == state.lower()`, `classes[0]` dependence) | Yes | Mildly |
| **Decision Compiler (`theory/compiler.py`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Adaptive Compute Router (`adaptive/router.py`)** | Yes | Yes | Yes | Yes | Flawed (`len(text.split())` token estimation, budget unenforced) | Yes | Mildly |
| **Layer-Trajectory Analysis (`representations/trajectory.py`)** | Yes | Yes | Yes | Yes | Yes (inspects model hidden states and computes emergence) | Requires HF model | Mildly (architecture generalizability) |
| **Multi-Layer Fusion (`representations/fusion.py`)** | Yes | Yes | Yes | Yes | Yes (Gated, Cross-Attention, Concat heads) | Yes | No |
| **Non-Autoregressive Head (`models/decision_head.py`)** | Yes | Yes | Yes | Yes | Yes (PyTorch bilinear decision head) | Yes | No |
| **Synthetic DOOM Simulator (`games/doom.py`)** | Yes | Yes | Yes | Yes | Yes (Deterministic ASCII game model) | Yes | **Yes (Presented as DOOM AI without clarifying synthetic nature)** |
| **Real DOOM WAD Parser (`games/ultimate_doom.py`)** | Yes | Yes | Yes | Yes | Yes (Parses genuine binary lumps) | Yes (requires WAD) | No |
| **ViZDoom Platform (`games/vizdoom_env.py`)** | Yes | Yes | **No (Actions bypassed Decision)** | Yes | Defective (bypassed decision engine, hardcoded probabilities) | Yes | **Yes (Claimed Farama verified & AI control while hardcoding)** |
| **FastAPI Service (`serving/app.py`)** | Yes | Yes | Yes | Yes | Lacks auth, size limits, structured error handling | Yes | Mildly (unbounded production readiness) |
| **Terminal UI (`tui/app.py`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Artifact Saver / Loader (`artifacts/saver.py`)** | Yes | Yes | Yes | Yes | SHA-256 hashing for tamper detection | Yes | **Yes (Marketed as cryptographic authenticity)** |
| **Universal API `anydecision.choose()` (`__init__.py`)** | Yes | Yes | Yes | Yes | Yes | Yes | No |

---

## Systematic Remediation Plan

1. **Phase 1 (This Phase)**: Document inventory, defects, and baseline audit in `docs/AUDIT_INVENTORY.md`.
2. **Phase 2: Core Probability, Tokenizer-Aware Scoring & Multi-Token Semantics**:
   - Create shared tokenizer-aware candidate utility (`scoring/candidate_tokenizer.py`).
   - Expose exact joint sequence log-probabilities alongside length-normalized ranking scores.
   - Cleanly distinguish candidate-conditional probability vs raw vocabulary logprob vs calibrated confidence.
3. **Phase 3: Backend Consistency & Batch Equivalence**:
   - Establish documented numerical equivalence between batch and single inference.
   - Replace synthetic vLLM approximation with explicit bounded contract.
4. **Phase 4: Calibration, Selective Conformal & Risk-Control Rigor**:
   - Split calibration data into threshold tuning and independent certification sets in `SelectiveConformalPredictor`.
   - Mechanically enforce guarantee tiers: `FORMAL_CONFORMAL`, `HIGH_CONFIDENCE_EMPIRICAL`, `HEURISTIC`, `UNAVAILABLE`.
   - Fail loudly on malformed calibration records (no silent class 0 mapping).
   - Evaluate active calibration across budget curves, removing unsupported 75% claims.
5. **Phase 5: Expected Utility, Shift Suspicion & Adaptive Compute**:
   - Explicit action-state semantic mappings (no string matching, no `classes[0]` hardcoding).
   - Rename OOD signals to "Distribution Shift Suspicion / Heuristic Signals".
   - Replace `text.split()` token estimation with exact tokenizer statistics and enforce call limits.
   - Clarify artifact security: SHA-256 tamper detection vs cryptographic digital signatures.
6. **Phase 6: FastAPI Service Hardening & Mock Isolation**:
   - Add auth to admin calibration endpoints, enforce input bounds, and safe structured error responses.
   - Explicitly label `backend="mock"` on mock outputs.
7. **Phase 7: Real DOOM / ViZDoom Architecture & Policy Control**:
   - Ensure ViZDoom action is driven strictly by `Decision.selected_action`.
   - Remove hardcoded probability distributions; evaluate observation states through typed `DecisionEngine` scoring.
   - Support `STATE`, `VISION`, and `HYBRID` observation modes.
   - Authoritative game metrics (kills from game variables, scenario-defined objectives).
   - Clean scorecards (factual metrics only).
   - Distinctly name synthetic simulator `synthetic_doom` / `toy_combat`.
8. **Phase 8: Claim-to-Evidence Matrix & Documentation Alignment**:
   - Create `docs/CLAIMS.md`.
   - Clean `README.md` to reflect verified code reality.
   - Verify 100% test pass rate across all modules.
