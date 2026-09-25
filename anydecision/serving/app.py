"""FastAPI HTTP service for typed decisions, batch queries, and calibration."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from anydecision.core.decision import Decision
from anydecision.core.engine import DecisionEngine
from anydecision.core.question import Question
from anydecision.core.types import DecisionLevel
from anydecision.serving.metrics import GLOBAL_METRICS


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


def create_app(engine: Optional[DecisionEngine] = None) -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="anydecision Runtime Service",
        description="High-throughput typed decision runtime directly extracting decisions from model probabilities.",
        version="0.2.0",
    )
    # Default to mock engine if none provided
    active_engine = engine or DecisionEngine(model="mock")

    @app.get("/health")
    def health_check() -> Dict[str, str]:
        return {"status": "ok", "service": "anydecision"}

    @app.get("/model")
    def get_model_info() -> Dict[str, Any]:
        return active_engine.metadata.model_dump()

    @app.post("/decide", response_model=Decision)
    async def decide_endpoint(req: DecideRequest) -> Decision:
        try:
            t0 = time.perf_counter()
            dec = await active_engine.async_decide(
                question=req.question,
                level=DecisionLevel(req.level) if req.level else None,
                min_confidence=req.min_confidence,
                target_error=req.target_error,
                allow_abstain=req.allow_abstain,
                trace=req.trace,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            calls = dec.diagnostics.number_of_backend_calls if dec.diagnostics else 1
            GLOBAL_METRICS.record_decision(lat_ms, abstained=dec.abstained, calls=calls)
            return dec
        except Exception as e:
            GLOBAL_METRICS.record_error()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch", response_model=List[Decision])
    async def batch_decide_endpoint(req: BatchDecideRequest) -> List[Decision]:
        try:
            t0 = time.perf_counter()
            results = await active_engine.async_batch_decide(
                questions=req.questions,
                level=DecisionLevel(req.level) if req.level else None,
                min_confidence=req.min_confidence,
                target_error=req.target_error,
                allow_abstain=req.allow_abstain,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            for dec in results:
                calls = dec.diagnostics.number_of_backend_calls if dec.diagnostics else 1
                GLOBAL_METRICS.record_decision(
                    lat_ms / max(1, len(results)), abstained=dec.abstained, calls=calls
                )
            return results
        except Exception as e:
            GLOBAL_METRICS.record_error()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/calibrate")
    def calibrate_endpoint(req: CalibrateRequest) -> Dict[str, Any]:
        try:
            calibrator = active_engine.calibrate(
                calibration_dataset=req.dataset,
                method=req.method,
            )
            GLOBAL_METRICS.record_calibration()
            return {"status": "calibrated", "calibrator": calibrator.to_dict()}
        except Exception as e:
            GLOBAL_METRICS.record_error()
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/metrics")
    def metrics_endpoint() -> Dict[str, Any]:
        return GLOBAL_METRICS.to_dict()

    @app.get("/metrics/prometheus")
    def prometheus_metrics() -> Response:
        return Response(content=GLOBAL_METRICS.prometheus_format(), media_type="text/plain")

    return app


app = create_app()
