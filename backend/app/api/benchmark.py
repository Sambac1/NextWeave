"""Read-only market import and local benchmark jobs."""
from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import re
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import httpx
from pydantic import BaseModel, Field

from app.benchmark.evaluation import (
    EvaluationRequest as CohortRequest, active_contract_runs, evaluation_csv, evaluation_store,
    pause_evaluation, queue_evaluation, read_evaluation, refresh_settlements,
)
from app.benchmark.markets import MarketSelectionRequired, _location, import_market
from app.benchmark.runner import BenchmarkRequest, QA_METHODS, SEQUENTIAL_METHODS, new_run, run_benchmark, stores, timestamp
from app.benchmark.scoring import EvaluationCase, EvaluationForecast, metrics_csv, score_forecasts
from app.settings import BACKEND_DIR, get_settings
from app.simulation.outlook import HARNESS_VERSION
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import read_json

router = APIRouter(prefix="/api/benchmark", tags=["market benchmark"])
logger = logging.getLogger(__name__)
_tasks: dict[str, asyncio.Task] = {}
_import_slots = asyncio.Semaphore(2)
COMPARISON_ARCHIVE = BACKEND_DIR / "evaluation_data" / "historical-live"


@router.get("/comparison/{kind}/{record_id}")
def get_comparison_record(kind: Literal["evaluations", "runs"], record_id: uuid.UUID):
    value = read_json(COMPARISON_ARCHIVE / kind / f"{record_id}.json", None)
    if value is None:
        raise HTTPException(404, "Saved comparison record not found.")
    return value


class MarketImport(BaseModel):
    market_url: str = Field(min_length=1, max_length=1500)
    as_of: datetime | None = None
    seed_wiki_urls: list[str] = Field(default_factory=list, max_length=5)
    archived_market: dict | None = None
    max_pages: int = Field(default=8, ge=1, le=12)


def _case_key(case_id: str) -> str:
    if not re.fullmatch(r"market_[a-f0-9]{24}", case_id):
        raise HTTPException(422, "Invalid market case ID")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nextweave:benchmark:{case_id}"))


def _load_case(case_id, cases):
    case = cases.load(_case_key(case_id), "case")
    if case is None:
        raise HTTPException(404, "Market snapshot not found.")
    return case


def _public_case(case):
    return case | {
        "evidence": [{key: value for key, value in page.items() if key != "text"}
                     for page in case["evidence"]]
    }


@router.post("/cases")
async def create_case(request: MarketImport):
    try:
        if request.as_of is not None:
            timestamp(request.as_of)
        async with _import_slots:
            case = await import_market(**request.model_dump())
    except MarketSelectionRequired as exc:
        raise HTTPException(422, {"message": str(exc), "choices": exc.choices}) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "The market or Wikipedia service is unavailable.") from exc
    cases, _ = stores()
    # The importer hashes the frozen rules, cutoff and evidence bundle into case_id.
    cases.save(_case_key(case["case_id"]), "case", case)
    return _public_case(case)


@router.get("/cases/{case_id}")
def get_case(case_id: str):
    cases, _ = stores()
    return _public_case(_load_case(case_id, cases))


@router.post("/runs", status_code=202)
async def start_benchmark(request: BenchmarkRequest):
    if len(_tasks) >= 2:
        raise HTTPException(409, "Two market experiments are running. Wait for one to finish.")
    cases, runs = stores()
    case = _load_case(request.case_id, cases)
    case = _confirmed_case(case, request)
    settings = get_settings()
    request = request.model_copy(update={
        "model": request.model or settings.llm_model,
        "base_url": request.base_url or settings.llm_base_url,
    })
    return _queue(request, case, runs, new_run(request, runs))


def _confirmed_case(case, request):
    try:
        if (request.market_probability is None) != (request.market_observed_at is None):
            raise ValueError("A market reference requires both a probability and a timestamp.")
        if request.market_observed_at and timestamp(request.market_observed_at) > timestamp(case["as_of"]):
            raise ValueError("The market reference must be observed at or before the forecast cutoff.")
        if request.confirmed_outcome_deadline is not None:
            deadline = timestamp(request.confirmed_outcome_deadline)
            if deadline <= timestamp(case["as_of"]):
                raise ValueError("The resolution deadline must be after the forecast cutoff.")
            return case | {"outcome_deadline": deadline.isoformat(), "deadline_basis": "confirmed_for_run"}
        if set(request.methods) - QA_METHODS:
            raise ValueError("Review the contract rules and confirm the deadline before starting a simulation.")
        return case
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _queue(request, case, runs, status):
    run_id = status["run_id"]
    status.update(as_of=case["as_of"], outcome_deadline=case["outcome_deadline"],
                  confirmed_outcome_deadline=case["outcome_deadline"] if request.confirmed_outcome_deadline else None,
                  group_id=case["group_id"], forecast_key=case.get("condition_id") or case["case_id"],
                  methods=request.methods, evidence_mode=request.evidence_mode, model=request.model)
    status["settings"] = {
        key: getattr(request, key) for key in ("steps", "sample_count", "qa_samples", "terminal_head_branches",
            "max_model_calls", "max_total_tokens", "max_preparation_calls", "max_preparation_tokens", "seed",
            "evidence_mode", "round_retrieval", "bdi_updates", "action_selection", "terminal_probability")
    }
    if request.market_probability is not None:
        status["market_reference"] = {"probability": request.market_probability, "label": "Market reference at cutoff",
                                      "observed_on": timestamp(request.market_observed_at).isoformat(),
                                      "url": case.get("market_url")}
    runs.save(run_id, "status", status)
    job = asyncio.create_task(run_benchmark(run_id, request, case, store=runs))
    _tasks[run_id] = job

    def finish(task):
        _tasks.pop(run_id, None)
        if not task.cancelled() and (error := task.exception()):
            logger.error("Benchmark %s failed to start: %s", run_id, type(error).__name__)

    job.add_done_callback(finish)
    return status


@router.get("/runs")
def list_benchmark_runs():
    _, runs = stores()
    entries = []
    for path in runs.root.glob("*/status.json"):
        try:
            entries.append((path.stat().st_mtime, path.parent.name))
        except FileNotFoundError:
            continue
    recent = []
    for _, run_id in sorted(entries, reverse=True)[:10]:
        try:
            status = get_benchmark(run_id)
        except HTTPException as exc:
            if exc.status_code in {404, 422}:
                continue
            raise
        recent.append({key: status.get(key) for key in ("run_id", "status", "stage", "model")})
    return recent


@router.get("/runs/{run_id}")
def get_benchmark(run_id: str):
    _, runs = stores()
    try:
        status = runs.load(run_id, "status")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if status is None:
        archive = Path(__file__).resolve().parents[1] / "benchmark" / "archives" / f"{run_id}.json"
        status = read_json(archive, None)
        if status is None:
            raise HTTPException(404, "Run not found.")
    if status["status"] in {"queued", "running"} and run_id not in _tasks and run_id not in active_contract_runs:
        try:
            pid = int((runs.directory(run_id) / ".running").read_text())
            if pid == os.getpid():
                # A same-process worker always registers the task before yielding.
                raise ProcessLookupError
            os.kill(pid, 0)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            return status | {"status": "interrupted", "error": "The worker stopped. Resume this run from saved progress."}
        except PermissionError:
            pass
    if status["status"] == "running" and (stage := status.get("stage")) and stage in (status.get("methods") or []):
        simulation_id = runs.load(run_id, f"simulation_ids/{stage}") if stage not in QA_METHODS else None
        if simulation_id:
            sim_store = EnsembleStore(runs.directory(run_id) / "simulations")
            progress = sim_store.load(simulation_id, "status", {})
            if progress:
                status = status | {"progress": {key: progress.get(key) for key in (
                    "stage", "sample_count", "completed_samples", "failed_samples", "completed_rounds", "model_calls",
                )}}
    contract = runs.load(run_id, "contract_spec", {})
    if (contract and set(status.get("methods") or []) & SEQUENTIAL_METHODS
            and contract.get("trajectory_protocol") != HARNESS_VERSION):
        return status | {"resume_available": False}
    incomplete = [row for row in status.get("results", []) if row["status"] != "completed"]
    if status.get("stage") == "finished" and incomplete and all(row.get("resume_available") is False for row in incomplete):
        return status | {"resume_available": False}
    return status


class ResumeBenchmark(BaseModel):
    api_key: str | None = None
    max_model_calls: int | None = Field(default=None, ge=1, le=5000)
    max_total_tokens: int | None = Field(default=None, ge=1000, le=10000000)
    max_preparation_calls: int | None = Field(default=None, ge=1, le=5000)
    max_preparation_tokens: int | None = Field(default=None, ge=1000, le=10000000)


@router.post("/runs/{run_id}/resume", status_code=202)
async def resume_benchmark(run_id: str, options: ResumeBenchmark):
    if run_id in _tasks:
        return get_benchmark(run_id)
    if len(_tasks) >= 2:
        raise HTTPException(409, "Two market experiments are running. Wait for one to finish.")
    cases, runs = stores()
    status = get_benchmark(run_id)
    if status["status"] == "completed":
        return status
    if status.get("resume_available") is False:
        raise HTTPException(409, "Start a new run to use updated evidence or the current simulation protocol.")
    config = runs.load(run_id, "config")
    if config is None:
        raise HTTPException(404, "The original experiment configuration is missing.")
    overrides = options.model_dump(exclude_none=True, exclude={"api_key"})
    if any(value < config[key] for key, value in overrides.items()):
        raise HTTPException(422, "Resume with the same or a higher budget.")
    request = BenchmarkRequest.model_validate({"terminal_branching": False, "terminal_probability": "replay"} | config | overrides | {"api_key": options.api_key})
    case = _confirmed_case(_load_case(request.case_id, cases), request)
    runs.save(run_id, "config", request.model_dump(exclude={"api_key"}))
    status.update(status="queued", error=None)
    runs.save(run_id, "status", status)
    return _queue(request, case, runs, status)


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1, max_length=5000)
    forecasts: list[EvaluationForecast] = Field(max_length=50000)
    methods: list[str] = Field(min_length=1, max_length=20)
    reference_method: str
    bootstrap_samples: int = Field(default=2000, ge=1, le=5000)
    seed: int = 0
    format: str = Field(default="json", pattern=r"^(json|csv)$")


@router.post("/evaluation")
def evaluate(request: EvaluationRequest):
    try:
        report = score_forecasts(request.cases, request.forecasts, methods=request.methods,
                                 reference_method=request.reference_method,
                                 bootstrap_samples=request.bootstrap_samples, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if request.format == "csv":
        return Response(metrics_csv(report), media_type="text/csv")
    return report


class ResumeCohort(BaseModel):
    api_key: str | None = None


def _evaluation(run_id):
    try:
        result = read_evaluation(run_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if result is None:
        raise HTTPException(404, "Evaluation not found.")
    return result


@router.post("/evaluations", status_code=202)
async def create_evaluation(request: CohortRequest):
    if {"single_step_simulation", "single_step_ensemble"}.intersection(request.methods):
        raise HTTPException(422, "Single-step methods are available only in saved evaluations.")
    if request.market_url is not None:
        market_url = request.market_url.strip()
        try:
            _location(market_url)
        except ValueError as exc:
            raise HTTPException(422, "Enter a valid HTTPS Polymarket contract or event URL.") from exc
        request = request.model_copy(update={"market_url": market_url})
    settings = get_settings()
    api_key = settings.llm_api_key if request.use_backend_credentials else request.api_key or settings.llm_api_key
    if not api_key.strip() or not (request.model or settings.llm_model).strip():
        raise HTTPException(422, "Configure a model and API key in Settings, or set the backend defaults.")
    try:
        return queue_evaluation(request)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/evaluations")
def list_evaluations():
    store = evaluation_store()
    result = []
    for path in store.root.glob("*/status.json"):
        status = read_json(path, None)
        if status and status.get("mode") in {"historical", "live"}:
            result.append({key: status.get(key) for key in ("run_id", "mode", "model", "status", "created_at", "quick_test")})
    return sorted(result, key=lambda row: row["created_at"] or "", reverse=True)[:20]


@router.get("/evaluations/{run_id}")
def get_evaluation(run_id: str):
    return _evaluation(run_id)


@router.post("/evaluations/{run_id}/resume", status_code=202)
async def resume_evaluation(run_id: str, request: ResumeCohort):
    current = _evaluation(run_id)
    if current["status"] not in {"Paused", "Failed", "Partial", "Collecting", "Running"}:
        return current
    store = evaluation_store()
    config = store.load(run_id, "config")
    if not config:
        raise HTTPException(404, "Evaluation configuration not found.")
    try:
        return queue_evaluation(CohortRequest.model_validate(config | {"api_key": request.api_key}), store=store, run_id=run_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/evaluations/{run_id}/pause")
async def stop_evaluation(run_id: str):
    _evaluation(run_id)
    await pause_evaluation(run_id)
    return _evaluation(run_id)


@router.post("/evaluations/{run_id}/settlement")
async def update_evaluation_settlement(run_id: str):
    _evaluation(run_id)
    await refresh_settlements(run_id)
    return _evaluation(run_id)


@router.get("/evaluations/{run_id}/export")
def export_evaluation(run_id: str, report: Literal["forecasts", "metrics"] = "forecasts"):
    result = _evaluation(run_id)
    if report == "metrics" and result["scores"] is None:
        raise HTTPException(409, "No resolved contracts are available for scoring.")
    csv_data = metrics_csv(result["scores"]) if report == "metrics" else evaluation_csv(result)
    return Response(csv_data, media_type="text/csv")
