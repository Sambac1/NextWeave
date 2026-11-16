from __future__ import annotations

import asyncio
from typing import Any
import uuid

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.application.event_ingestion import EventIngestionService, SourceCollectionError
from app.application.forecast_task import create_forecast_case, preview_question, preview_store
from app.benchmark.markets import MarketSelectionRequired
from app.domain.forecast_task import ForecastConfirmation, ForecastPreviewRequest
from app.knowledge.scraper import USER_AGENT
from app.domain import EventCreateRequest, WorldSnapshot
from app.storage.json_store import EventRepository, GraphRepository, SnapshotRepository
from app.storage.run_logger import RunLogger


router = APIRouter(prefix="/api/events", tags=["events"])
events = EventRepository()
graphs = GraphRepository(events)
snapshots = SnapshotRepository(events)


@router.post("/")
async def create_event(request: EventCreateRequest) -> dict[str, Any]:
    try:
        return await EventIngestionService(events, graphs).create(request)
    except SourceCollectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


_source_slots = asyncio.Semaphore(2)


def forecast_http_client():
    return httpx.AsyncClient(timeout=12, headers={"User-Agent": USER_AGENT})


@router.post("/forecast-preview")
async def forecast_preview(request: ForecastPreviewRequest):
    try:
        async with forecast_http_client() as client:
            spec = await preview_question(request.query, client)
    except MarketSelectionRequired as exc:
        return {"status": "select_contract", "choices": exc.choices}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "The market provider is unavailable. Try again.") from exc
    preview_id = str(uuid.uuid4())
    data = spec.model_dump(mode="json", exclude_none=True)
    preview_store().save(preview_id, "preview", data)
    return {"status": "confirm", "preview_id": preview_id, "spec": data}


@router.post("/forecast")
async def confirm_forecast(request: ForecastConfirmation):
    try:
        async with _source_slots, forecast_http_client() as client:
            return await create_forecast_case(request, client, previews=preview_store(), events=events)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Wikipedia is temporarily unavailable. Try again.") from exc


@router.get("/context")
async def event_context(
    event_title: str | None = Query(default=None, min_length=1),
    event_id: str | None = None,
) -> dict[str, Any]:
    if not event_id and event_title:
        event = events.find_by_title(event_title)
        event_id = str(event.get("event_id")) if event else None
    if not event_id:
        raise HTTPException(status_code=404, detail="Event not found")
    try:
        event_dir = events.directory(event_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Event not found") from None

    graph = graphs.get(event_id)
    figures = events.load_artifact(event_id, "agents", [])
    artifact_timeline = events.load_artifact(event_id, "timeline", [])
    artifact_relationships = events.load_artifact(event_id, "relationships", [])
    name_mapping = events.load_artifact(event_id, "name_mapping", {})
    committed = snapshots.get(event_id)
    snapshot = WorldSnapshot.model_validate(committed) if committed else None
    if not snapshot and figures:
        from app.domain.agency import normalize_entity_record
        from app.domain.decision_core import DecisionCore
        from app.knowledge.decision_seats import restore_seat_type
        from app.simulation.decision_profile import dossier_evidence_gaps, initial_beliefs

        plan = events.load_artifact(event_id, "decision_seat_plan", {})
        for index, figure in enumerate(figures):
            if not (figure.get("decision_core") or {}).get("seat"):
                continue
            figure = normalize_entity_record(restore_seat_type(figure, plan))
            try:
                core = DecisionCore.model_validate(figure["decision_core"])
            except ValueError:
                continue
            core.evidence_gaps = dossier_evidence_gaps(core)
            figures[index] = figure | {"decision_core": core.model_dump(),
                                      "beliefs": initial_beliefs(core).model_dump()}
    timeline = snapshot.timeline if snapshot else artifact_timeline
    relationships = (
        [item.model_dump() for item in snapshot.relationships]
        if snapshot
        else artifact_relationships
    )
    profile_source = [item.model_dump() for item in snapshot.agents] if snapshot else figures
    figures_by_id = {str(item.get("agent_id")): item for item in profile_source}
    figures_by_name = {str(item.get("name", "")).casefold(): item for item in profile_source}
    profiles_by_node_id: dict[str, dict[str, Any]] = {}
    enriched_nodes = []
    for wrapped in graph["nodes"]:
        node = wrapped.get("data", {})
        profile = figures_by_id.get(str(node.get("id"))) or figures_by_name.get(str(node.get("label", "")).casefold()) or {}
        if profile:
            profiles_by_node_id[str(node.get("id"))] = profile
        node_relationships = [
            item
            for item in relationships
            if str(item.get("source", "")) == str(node.get("id", ""))
            or str(item.get("target", "")) == str(node.get("id", ""))
        ]
        enriched_nodes.append(
            {
                "id": node.get("id"),
                "label": node.get("label"),
                "type": node.get("type", "agent"),
                "properties": node.get("properties", {}),
                "profile": profile,
                "timeline": timeline,
                "relationships": node_relationships,
            }
        )
    response = {
        "event_id": event_id,
        "event_title": events.load_artifact(event_id, "event", {}).get("title") or event_title,
        "forecast_spec": events.load_artifact(event_id, "forecast_spec", None),
        "background": snapshot.background if snapshot else events.load_artifact(event_id, "background", ""),
        "event_folder": str(event_dir),
        "graph": graph,
        "nodes": enriched_nodes,
        "profiles_by_node_id": profiles_by_node_id,
        "context_source": "simulation" if snapshot else "artifacts",
        "background_brief": events.load_artifact(event_id, "background_brief", None),
        "latest_context_meta": {
            "round_id": snapshot.round_id if snapshot else 0,
            "snapshot_version": snapshot.snapshot_version if snapshot else 0,
        },
        "timeline": timeline,
        "timeline_quality": events.load_artifact(event_id, "extraction_manifest", {}),
        "relationships": relationships,
        "relationship_action_evidence": events.load_artifact(event_id, "relationship_action_evidence", {}),
        "name_mapping": name_mapping,
        "figures_quality": events.load_artifact(event_id, "extraction_manifest", {}),
        "artifacts": {
            "figures": bool(profile_source),
            "timeline": bool(artifact_timeline),
            "relationships": bool(artifact_relationships),
            "latest_snapshot": bool(snapshot),
        },
    }
    logger = RunLogger("event_logs", operation="event-context-read", event_id=event_id)
    logger.record(
        source="event-context",
        round_id=snapshot.round_id if snapshot else 0,
        agent_id="all",
        content_type="timeline-agent-graph-context",
        status="success",
        data=response,
    )
    logger.finish(
        success=True,
        data={
            "context_source": response["context_source"],
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
        },
    )
    return response
