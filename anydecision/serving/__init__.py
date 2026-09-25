"""Serving module: FastAPI application and observability metrics."""

from anydecision.serving.app import app, create_app
from anydecision.serving.metrics import GLOBAL_METRICS, SystemMetrics

__all__ = [
    "GLOBAL_METRICS",
    "SystemMetrics",
    "app",
    "create_app",
]

