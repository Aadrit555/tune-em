"""FastAPI HTTP service: public decision API + protected admin calibration API."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from anydecision.core.decision import Decision
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.core.types import DecisionLevel
from anydecision.serving.metrics import GLOBAL_METRICS

logger = logging.getLogger("anydecision.serving")


class ServiceLimits(BaseModel):
    """Bounds enforced on every request to keep the service safe and predictable."""

    max_question_chars: int = 4000
    max_options: int = 20
    max_batch_size: int = 16
    max_permutations: int = 8
    request_timeout_s: float = 30.0
    max_concurrency: int = 32
    rate_limit_per_minute: int = 300


class DecideRequest(BaseModel):
    question: Question
    level: Optional[str] = "L0"
    min_confidence: Optional[float] = None
    target_error: Optional[float] = None
    allow_abstain: Optional[bool] = True
    trace: bool = False


class BatchDecideRequest(BaseModel):
    questions: List[Question]
    level: Optional[str] = "L0"
    min_confidence: Optional[float] = None
    target_error: Optional[float] = None
    allow_abstain: Optional[bool] = True


class CalibrateRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    method: str = "temperature"


def _error_envelope(code: str, message: str, request_id: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
    )


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter (per process)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: Dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> bool:
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


def create_app(
    engine: Optional[DecisionEngine] = None,
    admin_api_key: Optional[str] = None,
    limits: Optional[ServiceLimits] = None,
) -> FastAPI:
    """FastAPI application factory.

    PUBLIC endpoints (no auth): /health, /model, /decide, /batch, /metrics.
    ADMIN endpoints (require X-API-Key == admin_api_key): /calibrate.
    If admin_api_key is None, /calibrate is disabled and returns 403.
    """
    cfg = limits or ServiceLimits()
    resolved_admin_key = admin_api_key if admin_api_key is not None else os.environ.get("ANYDECISION_ADMIN_KEY")
    # Default to mock engine for offline dev/tests, but ALWAYS flag it explicitly.
    active_engine = engine or DecisionEngine(model="mock")
    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    rate_limiter = _RateLimiter(cfg.rate_limit_per_minute)

    app = FastAPI(
        title="anydecision Runtime Service",
        description="Typed probabilistic decision runtime over open-weight causal LMs.",
        version="0.2.0",
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    def _check_rate(request: Request) -> Optional[JSONResponse]:
        rid = request.headers.get("X-Request-ID") or "unknown"
        if not rate_limiter.check("global"):
            return _error_envelope("rate_limited", "Rate limit exceeded. Retry later.", rid, 429)
        return None

    def _validate_question(q: Question) -> Optional[str]:
        if len(q.text or "") > cfg.max_question_chars:
            return f"question.text exceeds {cfg.max_question_chars} characters"
        if len(q.options or []) > cfg.max_options:
            return f"question has more than {cfg.max_options} options"
        if len(q.options or []) == 0:
            return "question must define at least one option"
        return None

    def _is_mock() -> bool:
        try:
            return str(getattr(active_engine.metadata, "backend_name", "")) == "mock"
        except Exception:
            return False

    @app.get("/health")
    def health_check() -> Dict[str, str]:
        return {"status": "ok", "service": "anydecision"}

    @app.get("/model")
    def get_model_info() -> Dict[str, Any]:
        info = active_engine.metadata.model_dump()
        info["backend"] = info.get("backend_name", "unknown")
        info["is_mock"] = _is_mock()
        if info["is_mock"]:
            info["warning"] = (
                "Mock backend active: outputs are deterministic test fixtures, "
                "NOT genuine model evaluation. Configure a real model for evaluation."
            )
        return info

    @app.post("/decide", response_model=Decision)
    async def decide_endpoint(req: DecideRequest, request: Request) -> Any:
        limited = _check_rate(request)
        if limited is not None:
            return limited
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        problem = _validate_question(req.question)
        if problem is not None:
            return _error_envelope("invalid_request", problem, request_id, 422)
        try:
            level = DecisionLevel(req.level) if req.level else None
        except Exception:
            return _error_envelope("invalid_request", f"Unknown level {req.level!r}.", request_id, 422)
        try:
            t0 = time.perf_counter()
            async with semaphore:
                dec = await asyncio.wait_for(
                    active_engine.async_decide(
                        question=req.question,
                        level=level,
                        min_confidence=req.min_confidence,
                        target_error=req.target_error,
                        allow_abstain=req.allow_abstain,
                        trace=req.trace,
                    ),
                    timeout=cfg.request_timeout_s,
                )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            calls = dec.diagnostics.number_of_backend_calls if dec.diagnostics else 1
            GLOBAL_METRICS.record_decision(lat_ms, abstained=dec.abstained, calls=calls)
            return Response(
                content=dec.model_dump_json(),
                media_type="application/json",
                headers={"X-Backend-Mock": "true" if _is_mock() else "false"},
            )
        except asyncio.TimeoutError:
            GLOBAL_METRICS.record_error()
            return _error_envelope("timeout", "Decision request timed out.", request_id, 504)
        except HTTPException:
            raise
        except Exception:
            logger.exception("decide failed request_id=%s", request_id)
            GLOBAL_METRICS.record_error()
            return _error_envelope("internal_error", "Decision failed due to an internal error.", request_id, 500)

    @app.post("/batch", response_model=List[Decision])
    async def batch_decide_endpoint(req: BatchDecideRequest, request: Request) -> Any:
        limited = _check_rate(request)
        if limited is not None:
            return limited
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        if len(req.questions) == 0:
            return _error_envelope("invalid_request", "questions must not be empty.", request_id, 422)
        if len(req.questions) > cfg.max_batch_size:
            return _error_envelope(
                "invalid_request",
                f"batch size {len(req.questions)} exceeds maximum {cfg.max_batch_size}.",
                request_id,
                422,
            )
        for q in req.questions:
            problem = _validate_question(q)
            if problem is not None:
                return _error_envelope("invalid_request", problem, request_id, 422)
        try:
            level = DecisionLevel(req.level) if req.level else None
        except Exception:
            return _error_envelope("invalid_request", f"Unknown level {req.level!r}.", request_id, 422)
        try:
            t0 = time.perf_counter()
            async with semaphore:
                results = await asyncio.wait_for(
                    active_engine.async_batch_decide(
                        questions=req.questions,
                        level=level,
                        min_confidence=req.min_confidence,
                        target_error=req.target_error,
                        allow_abstain=req.allow_abstain,
                    ),
                    timeout=cfg.request_timeout_s * max(1, len(req.questions)),
                )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            for dec in results:
                calls = dec.diagnostics.number_of_backend_calls if dec.diagnostics else 1
                GLOBAL_METRICS.record_decision(
                    lat_ms / max(1, len(results)), abstained=dec.abstained, calls=calls
                )
            return Response(
                content="[" + ",".join(d.model_dump_json() for d in results) + "]",
                media_type="application/json",
                headers={"X-Backend-Mock": "true" if _is_mock() else "false"},
            )
        except asyncio.TimeoutError:
            GLOBAL_METRICS.record_error()
            return _error_envelope("timeout", "Batch decision request timed out.", request_id, 504)
        except Exception:
            logger.exception("batch decide failed request_id=%s", request_id)
            GLOBAL_METRICS.record_error()
            return _error_envelope("internal_error", "Batch decision failed due to an internal error.", request_id, 500)

    @app.post("/calibrate")
    def calibrate_endpoint(
        req: CalibrateRequest,
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Any:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        if not resolved_admin_key:
            return _error_envelope(
                "admin_disabled",
                "Calibration endpoint is disabled: set ANYDECISION_ADMIN_KEY to enable admin access.",
                request_id,
                403,
            )
        if x_api_key != resolved_admin_key:
            return _error_envelope("forbidden", "Invalid or missing admin API key.", request_id, 403)
        # Bound calibration payloads: dataset size + method allowlist.
        if len(req.dataset) == 0 or len(req.dataset) > 5000:
            return _error_envelope(
                "invalid_request", "dataset must contain 1..5000 records.", request_id, 422
            )
        if req.method not in ("temperature", "platt", "vector", "isotonic"):
            return _error_envelope(
                "invalid_request", f"Unknown calibration method {req.method!r}.", request_id, 422
            )
        try:
            calibrator = active_engine.calibrate(
                calibration_dataset=req.dataset,
                method=req.method,
            )
            GLOBAL_METRICS.record_calibration()
            return {"status": "calibrated", "calibrator": calibrator.to_dict()}
        except ValueError as e:
            GLOBAL_METRICS.record_error()
            # Client data errors: safe message, no traceback.
            return _error_envelope("calibration_failed", str(e)[:500], request_id, 400)
        except Exception:
            logger.exception("calibrate failed request_id=%s", request_id)
            GLOBAL_METRICS.record_error()
            return _error_envelope("internal_error", "Calibration failed due to an internal error.", request_id, 500)

    @app.get("/metrics")
    def metrics_endpoint() -> Dict[str, Any]:
        return GLOBAL_METRICS.to_dict()

    @app.get("/metrics/prometheus")
    def prometheus_metrics() -> Response:
        return Response(content=GLOBAL_METRICS.prometheus_format(), media_type="text/plain")

    return app


app = create_app()
