from __future__ import annotations

import json
import logging
import time
import uuid
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.benchmark import router as benchmark_router
from app.benchmark.evaluation import monitor_evaluations
from app.api.events import router as events_router
from app.api.experiments import router as experiments_router
from app.api.forecasts import router as forecasts_router
from app.api.graph import router as graph_router
from app.api.knowledge import router as knowledge_router
from app.settings import get_settings
from app.llm.http_pool import close_shared_http_clients
from app.storage.run_logger import RunLogger, reset_request_run_id, set_request_run_id

# Uvicorn configures its own loggers separately from application logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    monitor = asyncio.create_task(monitor_evaluations())
    try:
        yield
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        await close_shared_http_clients()


app = FastAPI(
    title="NextWeave API",
    version="2.0.0",
    description="Evidence-grounded event forecasting and market evaluation.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(benchmark_router)
app.include_router(events_router)
app.include_router(experiments_router)
app.include_router(graph_router)
app.include_router(knowledge_router)
app.include_router(forecasts_router)


@app.middleware("http")
async def audit_http_request(request: Request, call_next):
    """Persist request data, status and wall time without ever storing credentials."""
    request_run_id = str(uuid.uuid4())
    token = set_request_run_id(request_run_id)
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = {"byte_count": len(body_bytes), "preview": body_bytes[:2000].decode("utf-8", errors="replace")}
    event_id = str(body.get("event_id") or "") if isinstance(body, dict) else ""
    logger = RunLogger(
        "request_logs",
        run_id=request_run_id,
        operation=f"{request.method}_{request.url.path}",
        event_id=event_id or None,
    )
    logger.record(
        source="api",
        round_id="na",
        agent_id="caller",
        content_type="http-request",
        status="started",
        data={
            "method": request.method,
            "path": request.url.path,
            "query": dict(request.query_params),
            "client": request.client.host if request.client else None,
            "body": body,
        },
    )
    started = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        success = response.status_code < 400
        logger.record(
            source="api",
            round_id="na",
            agent_id="server",
            content_type="http-response-status",
            status="success" if success else "failed",
            duration_ms=duration_ms,
            data={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )
        logger.finish(
            success=success,
            data={"status_code": response.status_code, "duration_ms": round(duration_ms, 3)},
            error=None if success else f"HTTP {response.status_code}",
        )
        response.headers["X-Run-ID"] = request_run_id
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.record(
            source="api",
            round_id="na",
            agent_id="server",
            content_type="http-request-error",
            status="failed",
            duration_ms=duration_ms,
            data={"error_type": type(exc).__name__, "message": str(exc)},
        )
        logger.finish(success=False, data={"duration_ms": round(duration_ms, 3)}, error=exc)
        raise
    finally:
        reset_request_run_id(token)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
