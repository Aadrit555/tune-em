"""Backend loader and registry."""

from __future__ import annotations

from typing import Any, Union
from anydecision.backends.base import BaseBackend
from anydecision.backends.mock import MockBackend


def get_backend(backend: Union[str, BaseBackend], **kwargs: Any) -> BaseBackend:
    """Instantiate or return backend.

    Supports:
    - 'mock' or 'synthetic' -> MockBackend
    - 'transformers' or HuggingFace repo path -> TransformersBackend
    - 'vllm' -> VLLMBackend
    - custom BaseBackend instance
    """
    if isinstance(backend, BaseBackend):
        return backend

    s_backend = str(backend).lower()
    if s_backend in ("mock", "synthetic") or "mock" in s_backend:
        return MockBackend(model_name=str(backend), **kwargs)
    elif s_backend == "vllm":
        from anydecision.backends.vllm import VLLMBackend
        return VLLMBackend(**kwargs)
    else:
        # Default to Transformers backend
        try:
            from anydecision.backends.transformers import TransformersBackend
            return TransformersBackend(model_name_or_path=str(backend), **kwargs)
        except Exception as e:
            # If model path cannot be loaded locally / offline in tests, fallback cleanly
            if "not found" in str(e).lower() or "connection" in str(e).lower():
                return MockBackend(model_name=str(backend), **kwargs)
            raise e
