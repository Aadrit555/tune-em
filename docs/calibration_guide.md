# Statistical Calibration & Selective Prediction Guide

## Why LLMs Are Miscalibrated

Modern instruction-tuned and RLHF-aligned open-weight models suffer from severe probability distortion:
1. **Supervised Fine-Tuning Overconfidence**: Cross-entropy training on formatted answers pushes candidate probability mass towards 1.0 even on uncertain predictions.
2. **Positional & Tokenizer Bias**: Options placed first (`A` or top of list) or with shorter token representations systematically receive higher logits.
3. **Format Sensitivity**: Minor wording changes drastically alter raw logits.

`anydecision` distinguishes five distinct quantities:
- **Model Probability**: Raw uncalibrated soft probability extracted from logits.
- **Confidence**: Top-1 probability score assigned to the selected answer.
- **Calibrated Probability**: Mathematically adjusted probability matching empirical frequency on validation samples.
- **Uncertainty**: Epistemic and aleatoric dispersion (entropy or mutual information).
- **Posterior Risk**: Expected loss or probability of error: $\text{Risk} = 1 - \max_k p_k$.

---

## Calibration Methods Supported

| Method | Parameters | Use Case |
|---|---|---|
| **Temperature Scaling** | 1 scalar $T > 0$ | Best for preserving accuracy while correcting overconfidence. |
| **Vector Scaling** | $K$ weights + $K$ biases | For multiclass tasks with systematic per-class skew. |
| **Isotonic Regression** | Monotonic piecewise steps | Non-parametric; large calibration datasets ($N > 500$). |
| **Platt Scaling** | Logistic regression head | Binary and multiclass logistic readout. |
| **Conformal Prediction** | Quantile $\hat{q}$ | Finite-sample $(1-\alpha)$ coverage guarantees on prediction sets. |

---

## Evaluation Metrics

- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
- **Adaptive ECE**: Equal-mass binning to prevent empty bins in sparse confidence regions.
- **Brier Score**: Multi-class mean squared error: $\frac{1}{N}\sum_{n} \|p_n - y_n\|_2^2$.
- **Negative Log-Likelihood (NLL)**: Proper scoring rule punishing confident wrong guesses.
- **Selective Accuracy & Coverage**: Accuracy evaluated only on samples where model did not abstain.

