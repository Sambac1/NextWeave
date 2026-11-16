"""Persist automatic cohort jobs and reuse the existing contract runner and scorer."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Literal
import uuid

import httpx
from pydantic import Field, model_validator

from app.domain.common import LLMOptions
from app.knowledge.scraper import USER_AGENT
from app.settings import get_settings
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import read_json, write_json

from .cohort import collect_cohort, daily_snapshot, official_outcome
from .dynamics import simulation_case
from .evidence import utc_timestamp
from .markets import MarketSelectionRequired, _gamma
from .runner import DEFAULT_TRAJECTORIES, BenchmarkRequest, new_run, run_benchmark, sampling_allowance, stores
from .scoring import EvaluationCase, EvaluationForecast, score_forecasts

logger = logging.getLogger(__name__)
METHODS = {
    "direct_qa": "direct_qa", "qa_ensemble": "qa_ensemble", "single_step_simulation": "one_step",
    "single_trajectory_simulation": "single_multistep", "single_step_ensemble": "one_step_ensemble",
    "multi_trajectory_simulation": "trajectory_sampling", "mcts_simulation": "mcts_simulation",
}
CANONICAL_METHOD = {value: key for key, value in METHODS.items()}
USAGE_FIELDS = ("calls", "tokens", "elapsed_seconds", "unknown_usage_requests")
_jobs = {}
active_contract_runs: set[str] = set()
_monitor_lock = asyncio.Lock()
ACTIVE_STATES = {"Collecting", "Running", "Pausing"}


class EvaluationRequest(LLMOptions):
    mode: Literal["historical", "live"] = "historical"
    market_url: str | None = Field(default=None, min_length=1, max_length=1500)
    question_count: int = Field(default=30, ge=1, le=60)
    quick_test: bool = False
    use_backend_credentials: bool = False
    methods: list[str] = Field(min_length=1, max_length=7)
    qa_samples: int = Field(default=5, ge=2, le=64)
    single_step_samples: int = Field(default=5, ge=2, le=64)
    steps: int | None = Field(default=None, ge=2, le=12)
    sample_count: int = Field(default=DEFAULT_TRAJECTORIES, ge=2, le=64)
    mcts_iterations: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def check_methods(self):
        if len(set(self.methods)) != len(self.methods) or set(self.methods) - METHODS.keys():
            raise ValueError("Choose distinct supported forecasting methods.")
        return self


def evaluation_store():
    return EnsembleStore(get_settings().data_dir / "benchmark" / "evaluations")


def knowledge_cutoff(model):
    try:
        item = json.loads(os.getenv("BENCHMARK_MODEL_KNOWLEDGE_CUTOFFS", "{}")).get(model)
        if item and item.get("source"):
            return {"status": "Verified", "at": utc_timestamp(item["at"], "model cutoff").isoformat(),
                    "source": item["source"]}
    except (ValueError, AttributeError, TypeError, KeyError):
        pass
    return {"status": "Knowledge Cutoff Unverified"}


FORECAST_PROTOCOL = "sequential-action-outlook-v6"


def queue_evaluation(request, *, store=None, run_id=None):
    store = store or evaluation_store()
    if request.use_backend_credentials:
        request = request.model_copy(update={"api_key": None, "base_url": None if run_id is None else request.base_url})
    if run_id and run_id in _jobs:
        return store.load(run_id, "status")
    if _jobs:
        raise ValueError("An evaluation is already running. Wait for it to finish.")
    previous = store.load(run_id, "status", {}) if run_id else {}
    if run_id and previous.get("forecast_protocol") != FORECAST_PROTOCOL and store.load(run_id, "cohort"):
        raise ValueError("This evaluation uses an earlier evidence and action protocol. Start a new evaluation to use the updated protocol.")
    if run_id is None:
        run_id = str(uuid.uuid4())
        settings = get_settings()
        if request.market_url:
            request = request.model_copy(update={"question_count": 1, "quick_test": False})
        calls, tokens = 180 if request.quick_test else 250, 1_600_000 if request.quick_test else 3_000_000
        if "multi_trajectory_simulation" in request.methods:
            sample_calls, sample_tokens = sampling_allowance(request.sample_count, request.steps)
            calls, tokens = max(calls, sample_calls), max(tokens, sample_tokens)
        request = request.model_copy(update={
            "model": request.model or settings.llm_model, "base_url": request.base_url or settings.llm_base_url,
            "language": "en-US", "max_model_calls": calls, "max_total_tokens": tokens,
        })
        store.save(run_id, "config", request.model_dump(exclude={"api_key"}))
    status = previous | {"run_id": run_id, "mode": request.mode, "model": request.model,
                         "forecast_protocol": FORECAST_PROTOCOL,
                         "status": "Running" if previous.get("total") else "Collecting",
                         "quick_test": request.quick_test, "methods": request.methods, "error": None,
                         "created_at": previous.get("created_at", datetime.now(timezone.utc).isoformat())}
    if request.market_url:
        status["market_url"] = request.market_url
    store.save(run_id, "control", {"pause_requested": False})
    store.save(run_id, "status", status)
    task = asyncio.create_task(run_evaluation(run_id, request, store))
    _jobs[run_id] = task
    task.add_done_callback(lambda _: _jobs.pop(run_id, None))
    return status


def _pause_requested(store, run_id):
    return bool(store.load(run_id, "control", {}).get("pause_requested"))


async def pause_evaluation(run_id, store=None):
    store = store or evaluation_store()
    status = store.load(run_id, "status")
    if status is None or status["status"] not in ACTIVE_STATES:
        return
    store.save(run_id, "control", {"pause_requested": True})
    task = _jobs.get(run_id)
    if task is not None:
        if not task.cancelling():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # Cancellation can occur before the job coroutine has entered its try block.
        current = store.load(run_id, "status")
        if current["status"] in ACTIVE_STATES:
            store.save(run_id, "status", current | {"status": "Paused", "error": None})


async def run_evaluation(run_id, request, store):
    store.acquire(run_id)
    status = store.load(run_id, "status")
    owner = asyncio.current_task()

    async def watch_pause():
        while not _pause_requested(store, run_id):
            await asyncio.sleep(.5)
        if owner is not None and not owner.cancelling():
            owner.cancel()

    watcher = asyncio.create_task(watch_pause())
    try:
        limit = 1 if request.market_url or request.quick_test else request.question_count if request.mode == "historical" else 30
        cohort = store.load(run_id, "cohort")
        if cohort is None:
            cutoff = knowledge_cutoff(request.model) if request.mode == "historical" else {"status": "Prospective"}
            try:
                cases, labels, selection = await collect_cohort(
                    request.mode, limit, store.root,
                    minimum_cutoff=utc_timestamp(cutoff["at"], "model cutoff") if "at" in cutoff else None,
                    market_url=request.market_url,
                )
            except MarketSelectionRequired as exc:
                status.update(status="Review Required", market_choices=exc.choices,
                              error="This event contains multiple contracts. Choose one to continue.")
                return
            except (ValueError, httpx.HTTPError):
                if not request.market_url:
                    raise
                status.update(status="Review Required",
                              error="The selected contract could not be loaded. Check the Polymarket URL and try again.")
                return
            store.save(run_id, "settlements", labels)
            cohort = {"cases": cases, "selection": selection, "knowledge_cutoff": cutoff}
            store.save(run_id, "cohort_index", {
                **cohort, "cases": [{key: case[key] for key in ("case_id", "question", "as_of", "group_id")} for case in cases],
            })
            store.save(run_id, "cohort", cohort)
        if not cohort["cases"]:
            exclusions = cohort["selection"].get("excluded", [])
            message = "No contracts passed the rules, evidence and time checks."
            if request.market_url and exclusions:
                message = f"The selected contract could not be evaluated. {exclusions[0]['reason']}."
            status.update(status="Review Required", error=message)
            return
        if request.mode == "historical" and not request.quick_test and len(cohort["cases"]) < limit:
            status.update(status="Review Required", error=(
                f"Only {len(cohort['cases'])} of {limit} contracts passed the checks. "
                "Review the exclusions or use Quick Test to run one eligible contract."
            ))
            return
        links = store.load(run_id, "contract_runs", {})
        labels = store.load(run_id, "settlements", {})
        case_store, run_store = stores()
        for index, case in enumerate(cohort["cases"], 1):
            case = simulation_case(case)
            case_id = case["case_id"]
            child_id = links.get(case_id)
            previous = run_store.load(child_id, "status", {}) if child_id else {}
            if previous.get("status") == "completed":
                continue
            if request.mode == "live":
                try:
                    async with httpx.AsyncClient(timeout=12, headers={"User-Agent": USER_AGENT}) as client:
                        latest = await _gamma(client, "market", labels[case_id]["slug"])
                    if latest.get("conditionId") != labels[case_id]["condition_id"] or latest.get("closed") is not False:
                        labels[case_id].update(status="Review Required", reason="Contract closed or changed before forecast execution")
                        store.save(run_id, "settlements", labels)
                        continue
                except ValueError:
                    labels[case_id].update(status="Review Required", reason="Could not verify market status before forecasting")
                    store.save(run_id, "settlements", labels)
                    continue
            request_data = request.model_dump() | {
                "case_id": case_id, "methods": [METHODS[method] for method in request.methods],
                "confirmed_outcome_deadline": case["outcome_deadline"], "max_preparation_calls": 45,
                "max_preparation_tokens": 300000,
            }
            child_request = BenchmarkRequest.model_validate(request_data)
            key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nextweave:benchmark:{case_id}"))
            case_store.save(key, "case", case)
            if not child_id:
                child = new_run(child_request, run_store)
                child_id = child["run_id"]
                child.update(model=request.model, methods=child_request.methods, as_of=case["as_of"],
                             forecast_key=case["condition_id"], group_id=case["group_id"])
                run_store.save(child_id, "status", child)
                links[case_id] = child_id
                store.save(run_id, "contract_runs", links)
            status.update(status="Running", current=index, total=len(cohort["cases"]), active_run_id=child_id)
            store.save(run_id, "status", status)
            active_contract_runs.add(child_id)
            try:
                await run_benchmark(child_id, child_request, case, store=run_store)
            finally:
                active_contract_runs.discard(child_id)
        reports = [run_store.load(child, "status", {}) for child in links.values()]
        if request.market_url and not reports:
            reason = next(iter(labels.values())).get("reason", "The contract was unavailable at launch")
            status.update(status="Review Required", error=f"{reason}.", active_run_id=None, total=len(cohort["cases"]))
            return
        results = [report.get("status") for report in reports]
        methods = [result for report in reports for result in report.get("results", [])]
        if request.mode == "historical":
            final = ("Completed" if results and all(state == "completed" for state in results)
                     else "Partial" if any(state in {"completed", "partial"} for state in results)
                     else "Review Required" if methods and all(result["status"] == "unsupported" for result in methods)
                     else "Failed")
        else:
            final = "Awaiting Resolution"
        status.update(status=final, active_run_id=None, total=len(cohort["cases"]))
    except asyncio.CancelledError:
        if _pause_requested(store, run_id):
            status.update(status="Paused", error=None)
        else:
            status.update(status="Failed", error="The evaluation was interrupted. Resume from its saved checkpoints.")
        raise
    except Exception:
        logger.exception("Evaluation %s failed", run_id)
        status.update(status="Failed", error="The market service or evaluation worker failed. Resume to retry.")
    finally:
        watcher.cancel()
        store.save(run_id, "status", status)
        store.release(run_id)
        await asyncio.gather(watcher, return_exceptions=True)


def read_evaluation(run_id, store=None):
    store = store or evaluation_store()
    status = store.load(run_id, "status")
    if status is None:
        return None
    paused_requested = _pause_requested(store, run_id)
    if status["status"] in ACTIVE_STATES and run_id not in _jobs:
        lock = store.directory(run_id) / ".running"
        try:
            pid = int(lock.read_text())
            if pid == os.getpid():
                raise ProcessLookupError
            os.kill(pid, 0)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            status = status | ({"status": "Paused", "error": None} if paused_requested else
                               {"status": "Failed", "error": "The worker stopped. Resume from saved progress."})
        except PermissionError:
            pass
    if paused_requested and status["status"] in ACTIVE_STATES:
        status = status | {"status": "Pausing"}
    cohort = store.load(run_id, "cohort_index", {})
    labels = store.load(run_id, "settlements", {})
    links = store.load(run_id, "contract_runs", {})
    _, run_store = stores()
    cases, forecasts, questions = [], [], []
    methods = status["methods"]
    method_usage = {method: {field: {"total": 0, "count": 0} for field in USAGE_FIELDS} for method in methods}
    for case in cohort.get("cases", []):
        case_id = case["case_id"]
        label = labels.get(case_id, {})
        child_id = links.get(case_id)
        child = run_store.load(child_id, "status", {}) if child_id else {}
        results = child.get("results", [])
        resolved = label.get("status") == "Resolved"
        if resolved:
            cases.append(EvaluationCase(case_id=case_id, group_id=case["group_id"], as_of=case["as_of"],
                                        observed_yes=label["observed_yes"], fallback_probability=.5,
                                        market_probability=label.get("market_probability"),
                                        market_observed_at=label.get("market_observed_at")))
        probabilities = {}
        partial_forecasts = {}
        for row in results:
            method = CANONICAL_METHOD.get(row["method"])
            if method not in methods:
                continue
            probabilities[method] = row.get("p_yes")
            # Usage includes saved attempts without a forecast or an official outcome.
            for field, metric in method_usage[method].items():
                value = row.get(field)
                if value is not None:
                    metric["total"] += value
                    metric["count"] += 1
            diagnostics = row.get("simulation_diagnostics") or {}
            if row.get("p_yes") is None and row.get("unknown_bounds") and diagnostics.get("samples"):
                partial_forecasts[method] = {"valid_samples": diagnostics["valid_estimate_samples"],
                    "planned_samples": len(diagnostics["samples"]), "covered_p_yes": row.get("covered_p_yes"),
                    "unknown_bounds": row["unknown_bounds"]}
            if resolved:
                after_resolution = status["mode"] == "live" and (not row.get("forecast_created_at") or
                    utc_timestamp(row["forecast_created_at"], "forecast creation") >= utc_timestamp(label["resolved_at"], "settlement"))
                forecasts.append(EvaluationForecast(case_id=case_id, as_of=case["as_of"], method=method,
                    p_yes=None if after_resolution else row.get("p_yes"), calls=row.get("calls"), tokens=row.get("tokens"),
                    elapsed_seconds=row.get("elapsed_seconds")))
        questions.append({"case_id": case_id, "question": case["question"], "as_of": case["as_of"],
                          "run_id": child_id, "status": child.get("status"), "resolution": label.get("status"),
                          "round_plans": {CANONICAL_METHOD[method]: plan for method, plan in child.get("round_plans", {}).items()
                                          if method in CANONICAL_METHOD},
                          "observed_yes": label.get("observed_yes"), "quote": label.get("market_probability"),
                          "quote_at": label.get("market_observed_at"), "forecasts": probabilities,
                          "partial_forecasts": partial_forecasts,
                          "reason": label.get("reason")})
    if status["mode"] == "live" and status["status"] == "Awaiting Resolution" and questions:
        if all(row["resolution"] == "Resolved" for row in questions):
            status = status | {"status": "Resolved"}
        elif not any(row["resolution"] == "Awaiting Resolution" for row in questions):
            status = status | {"status": "Review Required"}
    finished = status["status"] not in ACTIVE_STATES | {"Paused"}
    report = score_forecasts(cases, forecasts, methods=methods, reference_method="direct_qa" if "direct_qa" in methods else methods[0]) if cases and finished else None
    active_id = status.get("active_run_id")
    active_stage = run_store.load(active_id, "status", {}).get("stage") if active_id else None
    config = store.load(run_id, "config", {})
    settings = {key: config[key] for key in ("question_count", "qa_samples", "single_step_samples", "steps",
                "sample_count", "mcts_iterations", "max_model_calls", "max_total_tokens") if key in config}
    return status | {"settings": settings, "selection": cohort.get("selection"), "knowledge_cutoff": cohort.get("knowledge_cutoff"),
                     "questions": questions, "scores": report, "method_usage": method_usage, "active_stage": active_stage}


async def refresh_settlements(run_id, store=None, client=None):
    store = store or evaluation_store()
    if client is None:
        async with httpx.AsyncClient(timeout=12, headers={"User-Agent": USER_AGENT}) as session:
            return await refresh_settlements(run_id, store, session)
    async with _monitor_lock:
        if run_id in _jobs:
            return
        status = store.load(run_id, "status", {})
        if status.get("mode") != "live":
            return
        labels = store.load(run_id, "settlements", {})
        for label in labels.values():
            if label.get("status") not in {"Awaiting Resolution", "Review Required"}:
                continue
            try:
                market = await _gamma(client, "market", label["slug"])
                if market.get("conditionId") == label["condition_id"]:
                    updated = official_outcome(market)
                    if updated["status"] != "Awaiting Resolution":
                        label.update(updated)
            except (ValueError, KeyError):
                continue
        if run_id not in _jobs:
            store.save(run_id, "settlements", labels)


async def monitor_evaluations():
    """Capture Top 30 at midnight UTC, and monitor settlements once per UTC hour."""
    await asyncio.sleep(60)
    store = evaluation_store()
    last_hour = None
    last_day = None
    while True:
        now = datetime.now(timezone.utc)
        hour = (now.date(), now.hour)
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
            if now.hour == 0 and now.date() != last_day:
                try:
                    await daily_snapshot(client, store.root)
                    daily_config = os.getenv("BENCHMARK_DAILY_EVALUATION")
                    marker = store.root / "scheduled" / f"{now.date().isoformat()}.json"
                    if daily_config and read_json(marker, None) is None:
                        payload = json.loads(daily_config)
                        scheduled = EvaluationRequest.model_validate(payload | {"mode": "live", "quick_test": False})
                        started = queue_evaluation(scheduled, store=store)
                        write_json(marker, {"run_id": started["run_id"]})
                    last_day = now.date()
                except (ValueError, httpx.HTTPError):
                    logger.warning("Daily market collection or configured forecast launch failed")
            if hour != last_hour:
                for path in store.root.glob("*/settlements.json"):
                    try:
                        await refresh_settlements(path.parent.name, store, client)
                    except (ValueError, OSError):
                        logger.warning("Could not refresh settlement for %s", path.parent.name)
                last_hour = hour
        await asyncio.sleep(60)


def evaluation_csv(report):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["Contract ID", "Question", "Forecast Cutoff", "Resolution Status", "Observed Yes",
                     "Market Yes Probability", "Market Quote Timestamp", *report["methods"]])
    for row in report["questions"]:
        # Prevent spreadsheet interpretation of a remote market title as a formula.
        question = row["question"]
        if question.startswith(("=", "+", "-", "@", "\t", "\r")):
            question = "'" + question
        writer.writerow([row["case_id"], question, row["as_of"], row["resolution"], row["observed_yes"],
                         row["quote"], row["quote_at"], *(row["forecasts"].get(method) for method in report["methods"])])
    return output.getvalue()
