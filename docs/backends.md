# Model Backends & Hardware Acceleration

`anydecision` features a modular backend abstraction separating probability extraction from decision policies, calibration, and serving.

## Supported Backends

### 1. Local Hugging Face Transformers (`transformers`)
- **Engine**: Native PyTorch `AutoModelForCausalLM` and `AutoTokenizer`.
- **Hardware**: CUDA, Apple Silicon (MPS), CPU (`float32`, `bfloat16`, `float16`).
- **Logit Extraction**: Directly inspects `model(**inputs).logits[0, -1, :]` for next-token decisions. For multi-token spans, computes exact teacher-forced autoregressive conditional log probabilities $\sum \log P(w_t \mid w_{<t})$.
- **Zero Generation**: Never invokes `model.generate()`.

### 2. High-Throughput vLLM (`vllm`)
- **Engine**: vLLM PagedAttention runtime.
- **Hardware**: NVIDIA Ampere/Hopper GPUs, multi-GPU tensor parallelism.
- **Logit Extraction**: Uses prompt logprobs and sampling logprobs without free-form generation.
- **Use Case**: Production clusters handling 100+ requests per second with prefix caching.

### 3. High-Speed Deterministic Mock (`mock`)
- **Engine**: CPU synthetic generator computing reproducible probability distributions from semantic text hashes.
- **Use Case**: Offline CI/CD testing, rapid prototyping, unit tests, and demo UI without downloading multi-gigabyte model weights.

---

## Adding a Custom Backend

Implement the `BaseBackend` abstract interface:

```python
from anydecision.backends.base import BaseBackend, ModelMetadata

class CustomInferenceBackend(BaseBackend):
    def get_metadata(self) -> ModelMetadata:
        ...

    def next_token_logprobs(self, prompt: str, candidate_strings: dict[str, str]) -> dict[str, float]:
        ...

    def sequence_logprobs(self, prompt: str, candidate_strings: dict[str, str], scoring_method: str) -> dict[str, float]:
        ...
```

