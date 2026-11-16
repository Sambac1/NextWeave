from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.domain.forecast_task import ForecastSpec
from app.domain import (
    EntityRequest,
    FiguresRequest,
    RelationshipsRequest,
    SummaryRequest,
    TranslationRequest,
)
from app.functional_agents.relationships import RelationshipAgent
from app.knowledge.extraction import extract_figures, figures_result_from_artifacts, extraction_resume_budget
from app.knowledge.operations import combine_summary, extract_entities, translate
from app.llm.client import LLMError
from app.storage.json_store import EventRepository
from app.storage.extraction_status import ExtractionStatusRepository


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/llm", tags=["knowledge"])
events = EventRepository()
_inflight_extractions: set[str] = set()


def _failure(exc: Exception) -> HTTPException:
    return HTTPException(status_code=502 if isinstance(exc, LLMError) else 422, detail=str(exc))


def _status_payload(event_id: str) -> dict:
    return ExtractionStatusRepository(events).get(event_id)


def _pipeline_payload(
    *,
    accepted: bool,
    reused: bool = False,
    already_running: bool = False,
    event_id: str,
    result: dict | None = None,
) -> dict:
    payload = {
        "accepted": accepted,
        "reused": reused,
        "already_running": already_running,
        "status": _status_payload(event_id),
    }
    if result:
        payload.update(result)
    return payload


async def _run_figures_pipeline(request: FiguresRequest) -> None:
    event_id = request.event_id
    try:
        await extract_figures(request, events)
    except Exception as exc:
        logger.exception("Figures pipeline failed for %s", event_id)
        try:
            if ExtractionStatusRepository(events).get(event_id).get('status') != 'partial':
                ExtractionStatusRepository(events).fail(event_id, exc)
        except (FileNotFoundError, ValueError):
            pass
    finally:
        _inflight_extractions.discard(event_id)


@router.post("/combined-summary")
async def combined_summary(request: SummaryRequest) -> dict[str, str]:
    try:
        return {"text": await combine_summary(request)}
    except Exception as orig:
        raise _failure(orig) from orig


@router.post("/entities")
async def entities(request: EntityRequest) -> dict:
    try:
        return {"entities": await extract_entities(request)}
    except Exception as orig:
        raise _failure(orig) from orig


@router.post("/figures-pipeline")
async def figures_pipeline(request: FiguresRequest, background_tasks: BackgroundTasks) -> dict:
    if request.timeline_mode != "parallel":
        raise HTTPException(422,"New extractions use the parallel pipeline for section coverage, date validation, and shared usage accounting.")
    try:
        events.directory(request.event_id)
    except (ValueError, FileNotFoundError) as orig:
        raise HTTPException(status_code=404, detail="Event not found") from orig

    saved_spec = events.load_artifact(request.event_id, "forecast_spec", None)
    if saved_spec:
        from app.application.forecast_task import question_contract

        spec = ForecastSpec.model_validate(saved_spec)
        predicate = question_contract(spec, [])["spec"]
        request = request.model_copy(update={
            "event_title": spec.question, "history_before": spec.as_of.date(),
            "required_decision_subject": predicate.get("subject", "") if predicate["route"] == "actor" else "",
            "exclude_poll_only_profiles": predicate.get("predicate") in {"rank", "threshold"},
            "planning_mode": "rules", "allow_empty_timeline": True,
        })
    elif request.allow_empty_timeline:
        request = request.model_copy(update={"allow_empty_timeline": False})

    if request.event_id in _inflight_extractions:
        return _pipeline_payload(
            accepted=False,
            already_running=True,
            event_id=request.event_id,
        )

    from app.api.experiments import _preparing_cases
    if request.event_id in _preparing_cases:
        raise HTTPException(409,"An experiment is being prepared for this analysis. Wait for preparation to finish before extracting again.")

    if request.resume_profiles:
        try:
            extraction_resume_budget(request, events)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    elif not request.force and _status_payload(request.event_id).get('status') == 'partial':
        raise HTTPException(422, 'This analysis is partially complete. Resume profile generation to reuse completed work.')
    saved = None if request.force or request.resume_profiles else figures_result_from_artifacts(request.event_id, events)
    if saved:
        current = _status_payload(request.event_id)
        if current.get("status") != "completed":
            ExtractionStatusRepository(events).update(
                request.event_id,
                run_id=str(current.get("run_id") or saved.get("run_id") or ""),
                stage="completed",
                progress=100,
                message=f"Loaded {len(saved['figures'])} saved Agents.",
                status="completed",
            )
        return _pipeline_payload(
            accepted=False,
            reused=True,
            event_id=request.event_id,
            result=saved,
        )

    ExtractionStatusRepository(events).update(
        request.event_id,
        run_id="queued",
        stage="queued",
        progress=1,
        message="Extraction queued...",
        status="running",
    )
    _inflight_extractions.add(request.event_id)
    background_tasks.add_task(_run_figures_pipeline, request)
    return _pipeline_payload(accepted=True, event_id=request.event_id)


@router.get("/figures-pipeline/{event_id}/status")
async def figures_pipeline_status(event_id: str) -> dict:
    try:
        return _status_payload(event_id)
    except FileNotFoundError as orig:
        raise HTTPException(status_code=404, detail="Event not found") from orig
    except ValueError as orig:
        raise HTTPException(status_code=422, detail=str(orig)) from orig


@router.get("/figures-pipeline/{event_id}/result")
async def figures_pipeline_result(event_id: str) -> dict:
    try:
        events.directory(event_id)
    except (ValueError, FileNotFoundError) as orig:
        raise HTTPException(status_code=404, detail="Event not found") from orig
    result = figures_result_from_artifacts(event_id, events)
    if result is None:
        raise HTTPException(status_code=404, detail="No extracted figures are saved for this event")
    return result


@router.get("/figures-pipeline/{event_id}/timeline")
async def figures_pipeline_timeline(event_id: str) -> dict:
    try:
        events.directory(event_id)
    except (ValueError, FileNotFoundError) as orig:
        raise HTTPException(status_code=404, detail="Event not found") from orig
    cached = events.load_artifact(event_id, "timeline_cache", {})
    timeline = cached.get("timeline") if isinstance(cached, dict) else None
    if not isinstance(timeline, list) or not timeline:
        raise HTTPException(status_code=404, detail="Timeline is not ready yet")
    return {"timeline": timeline}


@router.post("/relationships")
async def relationships(request: RelationshipsRequest) -> dict:
    try:
        return {"relationships": await RelationshipAgent().initialize(request, events)}
    except Exception as orig:
        raise _failure(orig) from orig


@router.post("/translate")
async def translation(request: TranslationRequest) -> dict:
    try:
        translations = await translate(request)
        if request.texts is not None:
            return {"translations": translations}
        return {"text": translations[0]}
    except Exception as orig:
        raise _failure(orig) from orig
