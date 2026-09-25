# System Architecture & Technical Design

## Overview

`anydecision` (repository `tune-em`) is a production-grade, typed decision runtime for open-weight Large Language Models (LLMs). Rather than asking models to generate free-form text tokens and using regex or parser heuristics, `anydecision` evaluates candidate target tokens or sequences directly in the model's output probability distribution.

```
                      +-----------------------------+
                      |       Typed Question        |
                      |   Question.choice/binary    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Prompt Template Engine    |
                      |   Permutation Generator     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       Model Backend         |
                      | (Transformers / vLLM / Mock)|
                      |    Next-token Logits /      |
                      | Multi-token Sequence Scores |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Probability Normalization |
                      |    Stable Softmax / Log-Z   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   L1: Invariance Debiasing  |
                      |  Ensemble Aggregation & TV  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    L2: Statistical Calib    |
                      |  Temperature / Vector / Iso |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Uncertainty & Selective Risk|
                      |  Entropy / OOD Diagnostics  |
                      |   Abstention Controller     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Strongly Typed         |
                      |      Decision Object        |
                      +-----------------------------+
```

---

## Readout Pipeline Detail

### 1. Token vs Multi-Token Sequence Scoring
For single-token options (e.g. `yes` / `no`), the model performs a single forward pass over the prompt text. The vocabulary logits at the final position are extracted and normalized over the candidate set using log-softmax:
$$\log p(y_k \mid x) = z_k - \log \sum_{j} \exp(z_j)$$

For multi-token options (e.g. `urgent technical support`), `anydecision` computes the autoregressive joint probability of the candidate sequence given the prompt:
$$\log p(w_1, \dots, w_L \mid x) = \sum_{t=1}^L \log p(w_t \mid x, w_{<t})$$

To prevent bias toward shorter phrases, `SequenceScorer` provides length normalization using the Wu et al. (GNMT) penalty:
$$\text{Score}_{\text{norm}} = \frac{\sum_{t=1}^L \log p(w_t \mid x, w_{<t})}{\left(\frac{5 + L}{6}\right)^\alpha}$$

### 2. Decision Hierarchy

- **L0 (Raw)**: Single prompt forward pass, raw logit extraction.
- **L1 (Zero-Label Invariance)**: Generates $M$ deterministic permutations of candidate orderings across multiple prompt templates. Evaluates candidate sensitivity, calculates total variation distance, and aggregates distributions via harmonic mean, logit mean, or probability mean.
- **L2 (Calibrated)**: Post-hoc statistical transformation fitted on validation data without fine-tuning model weights.

### 3. Selective Abstention & Conformal Prediction Sets
Applications can specify `target_error` or `min_confidence`. If posterior risk exceeds tolerance, the engine returns `abstained=True` and records diagnostic reasons. In addition, split conformal prediction provides prediction sets with $(1-\alpha)$ coverage guarantees.

