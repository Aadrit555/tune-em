# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-25

### Added
- **Core Decision API**: Strongly typed `DecisionEngine` and `Question` factory methods (`binary`, `choice`, `ordinal`, `score`, `multi_choice`, `custom`).
- **Readout System**: Direct vocabulary logit extraction and multi-token sequence scoring (`sum`, `mean`, `length_normalized`, `conditional`).
- **Decision Hierarchy**:
  - **L0 (Raw)**: Single forward pass, direct model probabilities.
  - **L1 (Zero-Label)**: Systematic option-order permutations, prompt template ensembles, and invariance debiasing.
  - **L2 (Calibrated)**: Post-hoc statistical calibration layers (Temperature Scaling, Vector Scaling, Isotonic Regression, Platt Scaling).
- **Statistical Rigor**:
  - Expected Calibration Error (ECE) and Adaptive ECE.
  - Multi-class Brier Score and Negative Log-Likelihood (NLL).
  - Reliability diagrams and selective risk vs. coverage evaluation curves.
  - Conformal prediction sets with $(1-\alpha)$ finite-sample coverage guarantees.
- **Selective Prediction & Abstention**:
  - `min_confidence`, `max_entropy`, and `target_error` risk thresholds.
  - Selective accuracy and coverage tracking.
- **Diagnostics & OOD Detection**:
  - Observable metrics for permutation agreement, position bias score, Shannon entropy, and distribution shift heuristics.
  - Complete step-by-step audit traces (`DecisionTrace`).
- **Inference & Backends**:
  - Local Hugging Face Transformers causal model backend.
  - High-throughput vLLM backend with prompt logprob reuse.
  - Deterministic high-speed Mock backend for fast CI/CD and offline development.
  - Synchronous and asynchronous batch decision execution (`batch_decide`, `async_decide`).
- **Online Adaptation**:
  - Streaming `observe()` interface with partition guards preventing evaluation data leakage.
- **Artifact System**:
  - Cryptographically hashed (SHA-256) JSON calibration artifacts with strict model/revision compatibility checks.
- **Tooling & Interfaces**:
  - Rich CLI with `ask`, `benchmark`, `inspect-artifact`, `serve`, `demo`.
  - Production FastAPI service (`/decide`, `/batch`, `/calibrate`, `/health`, `/metrics`, `/metrics/prometheus`).
  - Interactive Gradio research demo visualizing L0 $\to$ L1 $\to$ L2 $\to$ uncertainty $\to$ decision.
- **Comprehensive Benchmarks & Tests**:
  - 48 automated unit and integration tests passing at 100%.
  - Agent workflow safety benchmark comparing plain LLM vs calibrated abstention.

