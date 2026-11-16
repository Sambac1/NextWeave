"""Asynchronous experiment jobs and local evidence navigation."""
from __future__ import annotations

import asyncio
import math

from fastapi import APIRouter, HTTPException, Query, Path as PathParam
from pydantic import BaseModel, Field

from app.domain.forecast_task import QuestionExperimentRequest
from app.domain.ensemble import EnsembleRequest, EvaluationObservation, ReassessmentRequest
from app.knowledge.okf import KnowledgeStore
from app.knowledge.brief import NarrativeBrief, brief_catalog, forecast_brief_preview, narrative_document
from app.simulation.ensemble import (prepare_experiment, run_experiment, result_view, reassess_experiment,
                                     saved_reassessment, RESUMABLE_ENGINES)
from app.llm.budget import BudgetExceeded
from app.llm.client import LLMError
from app.storage.ensemble_store import EnsembleStore
from app.storage.extraction_status import ExtractionStatusRepository
from app.storage.json_store import BENCHMARK_EVENT_ORIGIN, EventRepository, read_json
from app.storage.analysis_trash import AnalysisTrash


router = APIRouter(prefix="/api", tags=["knowledge and experiments"])
_tasks: dict[str, asyncio.Task] = {}
_reassessing: set[str] = set()
_starting = 0
_preparing_cases: set[str] = set()


@router.get("/knowledge/cases")
def saved_cases():
    events = EventRepository()
    states = ExtractionStatusRepository(events)
    values = []
    for path in sorted(events.root.glob("*/event.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = read_json(path, {})
        if data.get("origin") == BENCHMARK_EVENT_ORIGIN:
            continue
        state = states.get(path.parent.name)
        values.append({"id": data.get("event_id"), "title": data.get("title"),
                       "created_at": data.get("created_at"),
                       "has_actors": bool(events.load_artifact(path.parent.name, 'agents', [])) and state.get('status') != 'partial',
                       "resume_available": state.get('resume_available', False)})
        if len(values) == 50:
            break
    return {"cases": values}


@router.get("/knowledge/trash")
def analysis_trash():
    return {"entries": AnalysisTrash().list()}


@router.delete("/knowledge/cases/{case_id}")
def delete_case(case_id: str):
    from app.api.knowledge import _inflight_extractions
    if case_id in _inflight_extractions or case_id in _preparing_cases:
        raise HTTPException(409, "Source collection or extraction is running. Wait for it to finish before moving the analysis to Trash.")
    try:
        return AnalysisTrash().delete(case_id, active_runs=set(_tasks) | _reassessing)
    except FileNotFoundError as exc:
        raise HTTPException(404, "The analysis could not be found. It may have been moved to Trash.") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/knowledge/trash/{trash_id}/restore")
def restore_case(trash_id: str):
    try:
        return AnalysisTrash().restore(trash_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/knowledge/trash/{trash_id}")
def purge_case(trash_id: str):
    try:
        return AnalysisTrash().purge(trash_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, "Permanent deletion is incomplete. Try again. An analysis cannot be restored after deletion starts.") from exc


async def _run(run_id, request):
    try:
        await run_experiment(run_id, request)
    except Exception:
        # The runner persists the actual error; keep background exceptions from
        # becoming unobserved task errors and retain the resumable checkpoints.
        pass
    finally:
        _tasks.pop(run_id, None)


@router.get('/experiments')
def case_experiments(event_id: str = Query(min_length=1, max_length=100)):
    store = EnsembleStore()
    runs = []
    for path in store.root.glob('*/config.json'):
        config = read_json(path, {})
        if config.get('request', {}).get('event_id') != event_id:
            continue
        state = store.load(path.parent.name, 'status', {})
        runs.append({'run_id': path.parent.name, 'created_at': config.get('created_at', ''),
                     'status': state.get('status'), 'sample_count': state.get('sample_count'),
                     'question_forecast': bool(config.get('request', {}).get('question_simulation')),
                     'steps': state.get('steps')})
    return {'runs': sorted(runs, key=lambda v: v['created_at'], reverse=True)[:50]}


@router.post('/experiments/context-preview')
async def preview_context(request: EnsembleRequest):
    from app.api.knowledge import _inflight_extractions
    if request.event_id and (request.event_id in _inflight_extractions or request.event_id in _preparing_cases):
        raise HTTPException(409, 'This analysis is busy. Wait for the current task to finish before estimating.')
    try:
        return await asyncio.to_thread(prepare_experiment, request, preview_only=True)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/experiments", status_code=202)
async def start_experiment(request: EnsembleRequest):
    if request.question_simulation is not None:
        raise HTTPException(422, "Start a new question forecast from a confirmed analysis.")
    return await _queue_experiment(request)


async def _queue_experiment(request: EnsembleRequest):
    global _starting
    if request.resume_run_id in _reassessing:
        raise HTTPException(409, "This experiment is being reassessed. Wait for reassessment to finish before resuming.")
    if request.resume_run_id in _tasks:
        return {"run_id": request.resume_run_id, "status": "running", "reused": True}
    if len(_tasks) + len(_reassessing) + _starting >= 2:
        raise HTTPException(409, "Two experiments are already running; resume or wait for them before starting another")
    from app.api.knowledge import _inflight_extractions
    if request.event_id and (request.event_id in _inflight_extractions or request.event_id in _preparing_cases):
        raise HTTPException(409,"Extraction or experiment preparation is running. Wait for the current step to finish.")
    _starting += 1
    if request.event_id:
        _preparing_cases.add(request.event_id)
    try:
        run_id, resolved = await asyncio.to_thread(prepare_experiment, request)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        _starting -= 1
        if request.event_id:
            _preparing_cases.discard(request.event_id)
    _tasks[run_id] = asyncio.create_task(_run(run_id, resolved))
    return {"run_id": run_id, "status": "queued", "sample_count": resolved.sample_count,
            "steps": resolved.steps, "resume_supported": True}


@router.post("/experiments/from-question", status_code=202)
async def start_question_experiment(request: QuestionExperimentRequest):
    from app.application.forecast_task import prepare_question_experiment

    try:
        resolved = await asyncio.to_thread(prepare_question_experiment, request, EventRepository())
    except FileNotFoundError as exc:
        raise HTTPException(404, "Forecast analysis not found.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return await _queue_experiment(resolved)


def _existing(run_id):
    store = EnsembleStore()
    try:
        config = store.load(run_id, "config")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not config:
        raise HTTPException(404, "Experiment not found")
    return store, config


@router.get("/experiments/{run_id}")
def experiment_status(run_id: str):
    store, config = _existing(run_id)
    status = store.load(run_id, "status", {})
    if status.get("status") == "running" and run_id not in _tasks:
        # This may be another API worker. Only report restart interruption when
        # the lock's owning process is gone, not merely absent from this dict.
        import os
        lock = store.directory(run_id) / ".running"
        try:
            os.kill(int(lock.read_text()), 0)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            status["status"] = "interrupted"
    from app.llm.budget import ModelBudget
    from app.simulation.ensemble import ENGINE_VERSION
    budget=store.load(run_id,"budget") or ModelBudget(EnsembleRequest.model_validate(config["request"])).report()
    question_forecast = bool(config["request"].get("question_simulation"))
    original = None if question_forecast else store.load(run_id, "result")
    report = store.load(run_id, 'report')
    if question_forecast:
        from app.simulation.question_report import report_view
        report = report_view(run_id, store, config, report)
    return status | ({"event_id": config["request"].get("event_id"), "question_forecast": True}
                     if question_forecast else {}) | {
                     "plan": store.load(run_id, "plan"), "knowledge_version": config["knowledge_version"],
                     "decision_policy": config["request"].get("decision_policy", "single"),
                     "report": report,
                     "result": result_view(run_id, store, config, original), "reassessment": saved_reassessment(run_id, store, original),
                     "runtime_plan":store.load(run_id,"runtime_plan"),
                     "budget":{k:v for k,v in budget.items() if k!="requests"},
                     "resume_supported":config.get("engine_version") in RESUMABLE_ENGINES}


@router.post("/experiments/{run_id}/reassess")
async def reassess_saved_experiment(run_id: str, request: ReassessmentRequest):
    store, config = _existing(run_id)
    if run_id in _tasks or run_id in _reassessing:
        raise HTTPException(409, "Wait for the active task to finish before reassessing.")
    if len(_tasks) + len(_reassessing) + _starting >= 2:
        raise HTTPException(409, "Two experiments are already running. Try again when one finishes.")
    _reassessing.add(run_id)
    try:
        return await reassess_experiment(run_id, request, store=store, config=config)
    except BudgetExceeded as exc:
        raise HTTPException(422, "Reassessment reached its budget before saving a result.") from exc
    except LLMError as exc:
        raise HTTPException(502, "The model call for reassessment failed. Check the model settings or try again.") from exc
    except TimeoutError as exc:
        raise HTTPException(504, "Reassessment timed out before saving a result.") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        _reassessing.discard(run_id)


@router.post("/experiments/{run_id}/stop")
def stop_experiment(run_id: str):
    store, _ = _existing(run_id)
    store.save(run_id, "control", {"cancel": True})
    if run_id in _tasks:
        _tasks[run_id].cancel()
    return {"run_id": run_id, "status": "stop_requested"}


@router.get("/experiments/{run_id}/samples/{sample_id}")
def experiment_sample(run_id: str, sample_id: int = PathParam(ge=1, le=400)):
    store, _ = _existing(run_id)
    record = store.sample(run_id, sample_id)
    if not record:
        raise HTTPException(404, "Sample not started yet")
    # UI replay uses public events/actions only. Complete BDI/private snapshots
    # are stored locally for audit, not returned from the public replay view.
    rounds = [{key: r.get(key) for key in ("step", "simulation_interval", "state_summary", "timeline", "new_relationships",
                                           "power_impacts", "next_watchpoints", "issue_updates", "continuing_state_notes",
                                           "measurement_states")} for r in record.get("rounds", [])]
    from app.simulation.activities import observations
    for frame, stored in zip(rounds, record.get('rounds', [])):
        public_actions = [a for a in stored.get('agent_actions', []) if a.get('visibility', 'public') == 'public']
        frame['public_action_count'] = sum(a.get('decision') == 'act' for a in public_actions)
        frame['public_wait_count'] = sum(a.get('decision') in {'wait', 'monitor', 'no_action'} for a in public_actions)
        frame['agent_actions'] = [{k:a.get(k) for k in ('agent_id','agent_name','decision','action_type','action','target')} for a in public_actions]
        frame['timeline']=[dict(event, activity_annotations=observations(event)) for event in frame.get('timeline') or []]
    return {"sample_id": sample_id, "status": record.get("status"), "rounds": rounds,
            "validation_issues": record.get("snapshot",{}).get("validation_issues",[]),
            "classification": record.get("classification"), "error": record.get("error")}


@router.get("/experiments/{run_id}/samples/{sample_id}/actor-states")
def experiment_sample_actor_states(run_id: str, sample_id: int = PathParam(ge=1, le=400)):
    """Expose only the saved final BDI snapshot; never reconstruct past rounds."""
    store, _ = _existing(run_id)
    record = store.sample(run_id, sample_id)
    if not record:
        raise HTTPException(404, "Sample not started yet")
    snapshot = record.get("snapshot") or {}
    actors = []
    for actor in snapshot.get("agents") or []:
        if not isinstance(actor, dict):
            continue
        actors.append({key: actor.get(key) for key in
                       ("agent_id", "name", "beliefs", "desires", "current_plan")})
    return {"sample_id": sample_id, "round_id": snapshot.get("round_id"),
            "state_scope": "latest_saved_snapshot", "actors": actors}


class EvaluationBatch(BaseModel):
    observations: list[EvaluationObservation] = Field(min_length=1, max_length=10000)


class BackgroundText(BaseModel):
    text: str = Field(default='', max_length=30000)
    brief: NarrativeBrief | None = None


@router.post("/knowledge/{case_id}/background")
def save_background(case_id: str, request: BackgroundText):
    try:
        events = EventRepository()
        if request.brief:
            articles = events.load_sources(case_id)
            document = narrative_document(request.brief, articles, brief_catalog(articles))
            events.save_artifact(case_id, 'background_brief', document)
            text = document['text']
        else:
            text = request.text
        events.save_artifact(case_id, "background", text)
        return {"saved": True, "text":text}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/knowledge/{case_id}/background")
def get_background(case_id: str):
    try:
        events = EventRepository()
        text = events.load_artifact(case_id, 'background', '')
        document = events.load_artifact(case_id, 'background_brief', None)
        if document and document.get('text') != text:
            document = None
        if not document and (spec := events.load_artifact(case_id, 'forecast_spec', None)):
            document = forecast_brief_preview(spec, events.load_sources(case_id))
            text = document['text']
        return {'text':text, 'document':document}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/forecast-evaluation")
def evaluate_forecasts(request: EvaluationBatch):
    """Score held-out observations. No LLM assigns its own reliability score."""
    seen = set()
    brier, baseline, logloss, bins = [], [], [], {}
    for item in request.observations:
        if item.case_key in seen:
            raise HTTPException(422, "Each held-out case_key must appear once")
        seen.add(item.case_key)
        values = item.probabilities
        brier.append(sum((p-int(k == item.observed_category))**2 for k, p in values.items()))
        n = len(values)
        baseline.append((1-1/n)**2 + (n-1)/n**2)
        logloss.append(-math.log(max(1e-15, values[item.observed_category])))
        predicted = max(values, key=values.get)
        confidence = values[predicted]
        bucket = min(9, int(confidence*10))
        bins.setdefault(bucket, []).append((confidence, int(predicted == item.observed_category)))
    size = len(brier)
    return {"case_count": size, "multiclass_brier": sum(brier)/size,
            "uniform_baseline_brier": sum(baseline)/size, "log_loss": sum(logloss)/size,
            "calibration_bins": [{"lower": k/10, "upper": (k+1)/10, "count": len(v),
                "mean_confidence": sum(p for p, _ in v)/len(v), "accuracy": sum(y for _, y in v)/len(v)}
                for k, v in sorted(bins.items())],
            "note": "Scores require genuinely held-out, time-correct observed outcomes; they are not evidence of validity for untested domains."}


@router.post("/knowledge/{case_id}/publish")
def publish_knowledge(case_id: str):
    _require_stable_case(case_id)
    try:
        return KnowledgeStore().publish(case_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/knowledge/{case_id}/search")
def search_knowledge(case_id: str, q: str = "", kind: str = Query("all", pattern="^(all|timeline|actor|section|background)$"),
                     actor_id: str = "", from_month: str = "", to_month: str = "", limit: int = Query(30, ge=1, le=100),
                     version: str | None = None):
    _require_stable_case(case_id)
    try:
        store = KnowledgeStore()
        hits = store.search(case_id, q, kind=kind, actor_id=actor_id, from_month=from_month, to_month=to_month, limit=limit, version=version)
        manifest = store.manifest(case_id, version=version)
        return {"hits": hits, "manifest": {k: v for k, v in manifest.items() if k != "documents"}}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/knowledge/{case_id}/graph")
def knowledge_graph(case_id: str, q: str = "", actor_id: str = "", from_month: str = "", to_month: str = "", limit: int = Query(60, ge=1, le=100), view: str = Query("evidence", pattern="^(actors|timeline|evidence)$"), focus_id: str = "", version: str | None = None):
    _require_stable_case(case_id)
    try:
        return KnowledgeStore().graph(case_id, q, limit, view=view, focus_id=focus_id, version=version, actor_id=actor_id, from_month=from_month, to_month=to_month)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc


def _require_stable_case(case_id):
    from app.api.knowledge import _inflight_extractions
    if case_id in _inflight_extractions or case_id in _preparing_cases:
        raise HTTPException(409,"This analysis is being updated. Refresh when extraction or experiment preparation finishes.")


@router.get("/knowledge/{case_id}/quality")
def knowledge_quality(case_id: str):
    _require_stable_case(case_id)
    from app.knowledge.extraction import _sections
    from app.knowledge.evidence import validate_history, VERSION
    try:
        events=EventRepository()
        history=events.load_artifact(case_id,"timeline_full",None)
        if history is None: history=events.load_artifact(case_id,"timeline",[])
        sources=events.load_sources(case_id)
        accepted,rejected=validate_history(history,_sections(sources)) if history and sources else ([],[])
        return {"validator":VERSION,"historical_events":len(history),"source_supported_events":len(accepted),
                "repaired_events":sum(e.get("date_evidence",{}).get("quote_resolution")=="ordered_ellipsis_restored" or e.get("date_evidence",{}).get("date_resolution")=="explicit_start_end_clause" for e in accepted),
                "needs_review":rejected,"extraction_plan":events.load_artifact(case_id,"extraction_plan",{}),
                "chapter_assessment":events.load_artifact(case_id,"chapter_assessment",{}),
                "extraction_budget":events.load_artifact(case_id,"extraction_budget",{}),
                "rejected_candidates":events.load_artifact(case_id,"timeline_evidence_review",{})}
    except (FileNotFoundError,ValueError) as exc:
        raise HTTPException(422,str(exc)) from exc


@router.get("/knowledge/{case_id}/records/{record_id}")
def knowledge_record(case_id: str, record_id: str, version: str | None = None):
    _require_stable_case(case_id)
    try: return KnowledgeStore().record(case_id,record_id,version=version)
    except FileNotFoundError as exc: raise HTTPException(404,str(exc)) from exc
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
