from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from app.knowledge.retrieval import event_upper_bound
from typing import Any, Awaitable, Callable, TypeVar

from app.agents import AgentSDKClient
from app.domain import (
    FiguresRequest,
    TimelineEvent,
)
from app.domain.common import checkpoint_languages
from app.domain.identity import stable_agent_id
from app.functional_agents.initialization import (
    TimelineExtractionAgent,
)
from app.knowledge.operations import make_client
from app.knowledge.scraper import split_sections
from app.knowledge.timeline_agent import to_events
from app.knowledge.coverage import extract_coverage
from app.knowledge.okf import KnowledgeStore
from app.layers.pipeline.schemas import LayerStageRequest
from app.layers.pipeline.stage import run_layer_stage
from app.storage.json_store import EventRepository, GraphRepository, SnapshotRepository
from app.storage.extraction_status import ExtractionState, ExtractionStatusRepository
from app.storage.run_logger import audit_finish, audit_record
from app.settings import get_settings
from app.knowledge.evidence import VERSION as EVIDENCE_VERSION, validate_history
from app.knowledge.okf import digest
from app.llm.budget import BudgetExceeded, ModelBudget, budget_scope
from app.llm.client import LLMError
from app.prompts.knowledge import TIMELINE_EXTRACTION_PROMPT


ResultT = TypeVar("ResultT")


async def _gather_with_progress(
    awaitables: list[Awaitable[ResultT]],
    on_completed: Callable[[int, int], None],
) -> list[ResultT]:
    """Preserve input order while exposing completion progress."""
    if not awaitables:
        return []

    async def indexed(index: int, awaitable: Awaitable[ResultT]) -> tuple[int, ResultT]:
        return index, await awaitable

    tasks = [asyncio.ensure_future(indexed(index, awaitable)) for index, awaitable in enumerate(awaitables)]
    results: list[ResultT | None] = [None] * len(tasks)
    completed = 0
    try:
        for task in asyncio.as_completed(tasks):
            index, value = await task
            results[index] = value
            completed += 1
            on_completed(completed, len(tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [value for value in results if value is not None]


def _is_fallback_source(source: dict[str, Any]) -> bool:
    """Whether this source came from the SDK fallback channel.

    The fallback reconstructs a page from alternative web sources, so nothing
    about its section structure is guaranteed. Flagged by ``sdk_scrape``.
    """
    return source.get("sdk_fallback") is True


def _sections(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    skipped = 0
    for source_number, source in enumerate(sources, 1):
        if source.get("error"):
            continue
        content = str(source.get("content") or "")
        persisted = [s for s in (source.get("sections") or []) if isinstance(s, dict)]
        if _is_fallback_source(source):
            if not persisted:
                skipped += 1
                continue
            stored_sections = persisted
        else:
            # Reconstruct section paths from the stored source text.
            reconstructed = split_sections(content, source_number)
            stored_sections = persisted if len(persisted) > len(reconstructed) else reconstructed
            if not stored_sections:
                stored_sections = persisted
        for section in stored_sections:
            if section.get("section_id") and section.get("content"):
                fallback_path = f"source_{source_number:02d}/{section.get('title') or section['section_id']}"
                result.append({
                    "section_id": str(section["section_id"]),
                    "title": str(section.get("title") or "Untitled"),
                    "path": str(section.get("path") or fallback_path),
                    "chapter": str(section.get("chapter") or section.get("path") or fallback_path),
                    "source_title": str(source.get("title") or ""),
                    "source_url": str(section.get("url") or source.get("url") or ""),
                    "content": str(section["content"]),
                })
    if not result:
        if skipped:
            raise ValueError(
                f"No structured source sections are available: {skipped} fallback source(s) "
                "contained no section structure. Recreate the event with the current scraper."
            )
        raise ValueError("No structured source sections are available; recreate the event with the current scraper")
    return result


def _local_section_index(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create inexpensive selector metadata without one model call per section."""
    date_pattern = re.compile(
        r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)?\d{2}\b|"
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
        re.IGNORECASE,
    )
    indexed = []
    for section in sections:
        content = str(section.get("content") or "")
        # Keep an excerpt for relevance selection; the full text remains the
        # canonical evidence used by the subsequent Timeline extraction.
        excerpt = re.sub(r"\s+", " ", content).strip()[:900]
        dates = date_pattern.findall(content)
        indexed.append({
            "section_id": section["section_id"],
            "path": section["path"],
            "chapter": section["chapter"],
            "title": section["title"],
            "source_title": section["source_title"],
            "source_url": section["source_url"],
            "summary": excerpt,
            "has_accurate_dates": bool(dates),
            "timeline_extractable": len(content) >= 120,
        })
    return indexed


def _timeline_cache_key(request: FiguresRequest, sources: list[dict[str, Any]]) -> str:
    source_fingerprints = []
    for item in sources:
        body = str(item.get("content") or "").encode("utf-8")
        source_fingerprints.append({
            "url": str(item.get("url") or ""),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "sections": [
                {
                    "path": str(section.get("path") or ""),
                    "url": str(section.get("url") or ""),
                    "content_sha256": hashlib.sha256(
                        str(section.get("content") or "").encode("utf-8")
                    ).hexdigest(),
                }
                for section in item.get("sections") or []
                if isinstance(section, dict)
            ],
        })
    options = {
        "pipeline_version": 5,
        "evidence_policy": EVIDENCE_VERSION, "prompt_hash":digest(TIMELINE_EXTRACTION_PROMPT),
        **({"history_before": request.history_before.isoformat()} if request.history_before else {}),
        **({"allow_empty_timeline": True} if request.allow_empty_timeline else {}),
        "planning_mode":request.planning_mode,
        "timeline_mode": request.timeline_mode,
        "title": request.event_title,
        "sources": source_fingerprints,
        "model": request.model or get_settings().llm_model,
        "base_url": request.base_url or get_settings().llm_base_url,
        "language": request.language,
        "use_timeline_agent": request.use_timeline_agent,
    }
    encoded = json.dumps(options, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _saved_timeline_matches(request, sources, cache, current_key):
    if not isinstance(cache, dict) or not isinstance(cache.get("timeline"), list) or (not cache["timeline"] and not request.allow_empty_timeline):
        return False
    key = cache.get("cache_key")
    return key == current_key or request.resume_profiles and any(
        key == _timeline_cache_key(request.model_copy(update={"language": language}), sources)
        for language in checkpoint_languages(request.language)[1:]
    )


def _filter_time_window(items: list[TimelineEvent], request: FiguresRequest) -> tuple[list[TimelineEvent], list[TimelineEvent]]:
    if request.time_window == "all" or not request.recent_years:
        return items, []
    dated_years = [int(match.group()) for item in items if (match := re.search(r"(?:19|20)\d{2}", item.timestamp))]
    if not dated_years:
        return items, []
    # Match the original project: "recent" is relative to the newest event in
    # this timeline, not to the wall-clock year when extraction happens.
    cutoff = max(dated_years) - request.recent_years + 1
    used, older = [], []
    for item in items:
        match = re.search(r"(?:19|20)\d{2}", item.timestamp)
        (used if not match or int(match.group()) >= cutoff else older).append(item)
    return used, older


def figures_result_from_artifacts(event_id: str, events: EventRepository) -> dict[str, Any] | None:
    """Return the figures-pipeline payload from saved artifacts, or None if none exist."""
    if ExtractionStatusRepository(events).get(event_id).get('status') == 'partial':
        return None
    agents = events.load_artifact(event_id, "agents", [])
    if not isinstance(agents, list) or not agents:
        return None
    from app.domain.agency import normalize_entity_record
    from app.domain.decision_core import project_record
    agents = [project_record(normalize_entity_record(a)) for a in agents]
    if spec := events.load_artifact(event_id, "forecast_spec", None):
        from app.knowledge.decision_seats import restore_seat_type
        from app.simulation.registry import simulation_readiness

        seat_plan = events.load_artifact(event_id, "decision_seat_plan", {})
        for actor in agents:
            readiness = simulation_readiness(
                restore_seat_type(actor, seat_plan), require_grounded=True,
                on_date=spec["as_of"][:10], require_bdi=True,
            )
            actor["forecast_role"] = "decision_actor" if readiness in {"ready", "restricted"} else "evidence_participant"
    timeline = events.load_artifact(event_id, "timeline", [])
    return {
        "figures": agents,
        "timeline": timeline if isinstance(timeline, list) else [],
        "name_mapping": events.load_artifact(event_id, "name_mapping", {}),
        "profiles": events.load_artifact(event_id, "profiles", []),
        "profiles_view": events.load_artifact(event_id, "profiles_view", {}),
        "quality": events.load_artifact(event_id, "extraction_manifest", {}),
        "run_id": str((events.load_artifact(event_id, "extraction_status", {}) or {}).get("run_id") or ""),
    }


async def _extract_figures(request: FiguresRequest, events: EventRepository) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    sources = events.load_sources(request.event_id)
    sections = _sections(sources)
    client, recorder = make_client(request, "knowledge_runs")
    agent_runtime = AgentSDKClient(request, recorder)
    timeline_agent = TimelineExtractionAgent(agent_runtime, utility_client=client)
    status_repository = ExtractionStatusRepository(events)

    def report_progress(
        stage: str,
        progress: int,
        message: str,
        completed_items: int = 0,
        total_items: int = 0,
        status: ExtractionState = "running",
    ) -> None:
        value = status_repository.update(
            request.event_id,
            run_id=recorder.run_id,
            stage=stage,
            progress=progress,
            message=message,
            status=status,
            completed_items=completed_items,
            total_items=total_items,
        )
        audit_record(
            recorder,
            source="knowledge-extraction",
            round_id="extraction",
            agent_id="all",
            content_type="extraction-progress",
            status=status,
            data=value,
        )

    timeline_started = time.perf_counter()
    timeline_key = _timeline_cache_key(request, sources)
    timeline_cache = events.load_artifact(request.event_id, "timeline_cache", {})
    cached_timeline = _saved_timeline_matches(request, sources, timeline_cache, timeline_key)
    missing_month_events = int(timeline_cache.get("timeline_events_missing_month") or 0) if cached_timeline else 0
    if cached_timeline:
        summaries = timeline_cache.get("summaries") or []
        selected_ids = set(timeline_cache.get("selected_section_ids") or [])
        selected_sections = [item for item in sections if item.get("section_id") in selected_ids]
        raw_timeline_dicts = timeline_cache.get("timeline_candidates") or []
        timeline = [TimelineEvent.model_validate(item) for item in timeline_cache["timeline"]]
        timeline_source = str(timeline_cache.get("timeline_source") or "timeline_agent")
        timeline_agent_artifact = timeline_cache.get("timeline_agent") or {}
        report_progress("timeline_cached", 68, "Reusing the saved Timeline; continuing with decision-makers and profiles...")
    else:
        report_progress(
            "summarizing_sources",
            5,
            f"Indexing {len(sections)} source sections locally...",
            total_items=len(sections),
        )
        audit_record(
            recorder,
            source="knowledge-extraction",
            round_id="extraction",
            agent_id="all",
            content_type="source-sections",
            status="info",
            # The canonical source corpus already stores each full page body.
            # Keep the audit trail useful without duplicating every section's text.
            data=[
                {
                    "section_id": item["section_id"],
                    "title": item["title"],
                    "path": item["path"],
                    "source_title": item["source_title"],
                    "source_url": item["source_url"],
                    "char_count": len(item["content"]),
                }
                for item in sections
            ],
        )
        audit_record(
            recorder,
            source="knowledge-extraction",
            round_id="extraction",
            agent_id="all",
            content_type="persisted-graph-before-extraction",
            status="info",
            data=GraphRepository(events).get(request.event_id),
        )

        summaries = _local_section_index(sections)
        report_progress(
            "summarizing_sources",
            20,
            f"Indexed {len(summaries)} sections; selecting Timeline evidence...",
            len(summaries),
            len(summaries),
        )
        # Keyed by section_id, not path: id is unique by construction, while two
        # sections can share a path (a fallback source whose excerpts reuse a title).
        summary_by_id = {item["section_id"]: item for item in summaries}
        selection_input = [section | summary_by_id[section["section_id"]] for section in sections]
        report_progress("selecting_timeline_sections", 28, "Selecting timeline evidence sections...")
        if request.timeline_mode == "parallel":
            selected_section_ids = [s["section_id"] for s in sections]
        else:
            selected_section_ids = await timeline_agent.select_sections(request.event_title, selection_input)
        selected_id_set = set(selected_section_ids)
        selected_sections = [item for item in sections if item["section_id"] in selected_id_set]
        if not selected_sections:
            raise ValueError("No source sections survived evidence selection")

        # The agent path is the default. It is asked for the whole recorded history
        # in one pass; the per-section path below stays intact as the fallback,
        # because a failed agent should cost a warning rather than the run.
        agent_outcome = None
        if request.timeline_mode == "parallel":
            agent_outcome = await extract_coverage(
                sections, client, events.directory(request.event_id),
                lambda done, total, hits: report_progress(
                    "extracting_timeline", 30 + int(30 * done / total),
                    f"Timeline chunks {done}/{total}; reused {hits} cached chunks", done, total),
                deadline=request.extraction_deadline_seconds, planning_mode=request.planning_mode,
                **({"allow_empty": True} if request.allow_empty_timeline else {}),
            )
        elif request.use_timeline_agent:
            report_progress(
                "building_timeline",
                30,
                f"Building the event timeline from {len(selected_sections)} selected sections...",
                total_items=len(selected_sections),
            )
            agent_outcome = await timeline_agent.build_timeline(
                request.event_title,
                [item for item in selection_input if item["section_id"] in selected_id_set],
            )
            audit_record(
                recorder,
                source="timeline-extraction",
                round_id="extraction",
                agent_id="all",
                content_type="timeline-agent-outcome",
                status="success" if agent_outcome.applied else "info",
                data={
                    "applied": agent_outcome.applied,
                    "reason": agent_outcome.reason,
                    "stats": agent_outcome.stats,
                },
            )

        if agent_outcome is not None and agent_outcome.applied:
            raw_timeline_dicts = list(agent_outcome.historical)
            merged_dicts = list(agent_outcome.historical)
            agent_missing_month = int(
                (agent_outcome.stats or {}).get("historical_missing_month", 0)
            )
            report_progress(
                "extracting_timeline",
                60,
                f"TimelineAgent produced {len(merged_dicts)} historical events.",
            )
            audit_record(
                recorder,
                source="timeline-extraction",
                round_id="extraction",
                agent_id="all",
                content_type="merged-timeline-before-ids",
                status="success",
                data=merged_dicts,
            )
        else:
            agent_missing_month = 0
            report_progress(
                "extracting_timeline",
                30,
                f"Extracting events from {len(selected_sections)} selected sections...",
                total_items=len(selected_sections),
            )
            raw_groups = await _gather_with_progress(
                [timeline_agent.extract_section(section) for section in selected_sections],
                lambda completed, total: report_progress(
                    "extracting_timeline",
                    30 + int(25 * completed / total),
                    f"Extracted timeline events from {completed} of {total} sections.",
                    completed,
                    total,
                ),
            )
            raw_timeline_dicts = [item for group in raw_groups for item in group]
            audit_record(
                recorder,
                source="timeline-extraction",
                round_id="extraction",
                agent_id="all",
                content_type="raw-timeline-candidates",
                status="success",
                data=raw_timeline_dicts,
            )
            if not raw_timeline_dicts and not request.allow_empty_timeline:
                raise ValueError("Selected source sections contained no extractable timeline events")
            report_progress(
                "deduplicating_timeline",
                60,
                f"Deduplicating {len(raw_timeline_dicts)} timeline candidates in bounded batches...",
            )
            merged_dicts = await timeline_agent.merge(raw_groups) if raw_timeline_dicts else []
            audit_record(
                recorder,
                source="timeline-extraction",
                round_id="extraction",
                agent_id="all",
                content_type="merged-timeline-before-ids",
                status="success",
                data=merged_dicts,
            )
        valid_paths = {str(item.get("path") or "") for item in sections}
        merged_dicts, date_validation = to_events(
            {"historical": merged_dicts},
            valid_paths, limit=None if request.timeline_mode == "parallel" else 60,
        )
        if request.timeline_mode == "parallel":
            merged_dicts, rejected = validate_history(merged_dicts, sections)
            review = events.load_artifact(request.event_id,"timeline_evidence_review",{})
            review["canonical_rejected"] = rejected
            events.save_artifact(request.event_id,"timeline_evidence_review",review)
        missing_month_events = agent_missing_month + date_validation["historical_missing_month"]
        if request.history_before:
            boundary = request.history_before.isoformat()
            merged_dicts = [
                item for item in merged_dicts
                if (upper := event_upper_bound(item)) and upper < boundary
            ]
        if not merged_dicts and not request.allow_empty_timeline:
            raise ValueError(
                "No timeline events have a resolvable year and month; undated or year-only events were discarded"
            )
        merged_timeline = []
        previous = events.load_artifact(request.event_id, "timeline_full", [])
        identities = {(e.get("timestamp"), e.get("description"), e.get("evidence_text")): e["timeline_id"] for e in previous}
        evidence_matches = {}
        for e in previous:
            evidence_matches.setdefault((e.get("timestamp"), e.get("evidence_text")), []).append(e["timeline_id"])
        used_ids = set()
        for index, item in enumerate(merged_dicts, 1):
            identity_seed = f"{item['timestamp']}:{item['description']}:{item.get('evidence_text', '')}"
            token = stable_agent_id(request.event_id, request.event_title, identity_seed).removeprefix("agent_")
            identifier = identities.get((item["timestamp"], item["description"], item.get("evidence_text")))
            matches = evidence_matches.get((item["timestamp"], item.get("evidence_text")), [])
            if not identifier and len(matches) == 1:
                identifier = matches[0]
            if not identifier or identifier in used_ids:
                identifier = f"hist_{token}"
            if identifier in used_ids:
                raise ValueError("Historical event identity collision")
            used_ids.add(identifier)
            merged_timeline.append(TimelineEvent(
                timeline_id=identifier,
                timestamp=item["timestamp"],
                description=item["description"],
                actors=item.get("actors") or [],
                event_type=item.get("event_type") or "other",
                confidence=float(item.get("confidence") or 0.7),
                importance=float(item.get("importance_score") or 0.5),
                date_precision=item.get("date_precision") or "unknown",
                date_original=item.get("date_original") or "",
                time_role=item.get("time_role") or "occurrence",
                modality=item.get("modality") or "occurred",
                date_evidence=item.get("date_evidence") or {},
                date_range=item.get("date_range") or {},
                evidence_text=item.get("evidence_text") or "",
                source_sections=item.get("source_sections") or [],
                source_event_ids=[str(value) for value in item.get("source_event_ids") or []],
                source_event_count=int(item.get("source_event_count") or 0),
                origin="historical",
            ))
        # The window still narrows the timeline the stage and the forecast see; the
        # events it excludes are simply dropped now, rather than kept in a
        # timeline_history.json nothing ever read.
        events.save_artifact(request.event_id, "timeline_full", [item.model_dump() for item in merged_timeline])
        timeline = merged_timeline
        audit_record(
            recorder,
            source="timeline-extraction",
            round_id="extraction",
            agent_id="all",
            content_type="final-timeline",
            status="success",
            data=[item.model_dump() for item in timeline],
        )
        if not timeline and not request.allow_empty_timeline:
            raise ValueError("No timeline events remain in the requested time window")

        timeline_source = "parallel" if request.timeline_mode == "parallel" else ("timeline_agent" if agent_outcome is not None and agent_outcome.applied else "per_section")
        timeline_agent_artifact = (
            agent_outcome.artifact
            if agent_outcome is not None
            else {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "event_title": request.event_title,
                "applied": False,
                "reason": "not attempted: the request set use_timeline_agent=false",
            }
        )
        events.save_artifact(request.event_id, "timeline_cache", {
            "cache_key": timeline_key,
            "summaries": summaries,
            "selected_section_ids": [item["section_id"] for item in selected_sections],
            "timeline_candidates": raw_timeline_dicts,
            "timeline": [item.model_dump() for item in timeline],
            "timeline_source": timeline_source,
            "allow_empty_timeline": request.allow_empty_timeline,
            "timeline_agent": timeline_agent_artifact,
            "timeline_events_missing_month": missing_month_events,
        })
    if cached_timeline:
        events.save_artifact(request.event_id, "timeline_full", [item.model_dump() for item in timeline])
    timeline, _ = _filter_time_window(timeline, request)
    timeline_elapsed = time.perf_counter() - timeline_started
    report_progress(
        "extracting_actor_evidence",
        70,
        f"Extracting actor and action evidence from {len(timeline)} timeline events...",
    )
    # Actor extraction onward is the layered-figure stage: it extracts actors and
    # directed actions, merges aliases, assigns principals and members, and builds the
    # layered profiles.
    actor_profile_started = time.perf_counter()
    stage_result = await run_layer_stage(
        request=LayerStageRequest(
            event_id=request.event_id,
            event_title=request.event_title,
            evidence_cutoff=request.history_before.isoformat() if request.history_before else "",
            api_key=getattr(client, "api_key", request.api_key or ""),
            model=getattr(client, "model", request.model or ""),
            base_url=getattr(client, "base_url", request.base_url),
            language=request.language,
            max_key_figures=request.max_key_figures,
            resume_profiles=request.resume_profiles,
            required_decision_subject=request.required_decision_subject,
            exclude_poll_only_profiles=request.exclude_poll_only_profiles,
        ),
        timeline=timeline,
        events=events,
    )
    agents = stage_result["agents"]
    if stage_result.get('profile_warning') or (
        not agents and (stage_result.get('quality') or {}).get('key_figures_selected', 0)
    ):
        raise ValueError(stage_result.get('profile_warning') or 'Actors were identified and progress was saved. Profiles are still incomplete.')
    audit_record(
        recorder,
        source="layered-figure-stage",
        round_id="extraction",
        agent_id="all",
        content_type="stage-result",
        status="success",
        data={
            "profiles": len(stage_result.get("profiles") or []),
            "agents": len(agents),
            "name_mapping": stage_result.get("name_mapping") or {},
            "quality": stage_result.get("quality") or {},
        },
    )
    figure_result = stage_result
    actor_profile_elapsed = time.perf_counter() - actor_profile_started
    for agent in agents:
        audit_record(
            recorder,
            source="agent-profile-extraction",
            round_id="extraction",
            agent_id=agent.agent_id,
            content_type="final-agent-profile",
            status="success",
            data=agent.model_dump(),
        )
    artifacts = {
        "section_summaries": summaries,
        "selected_sections": [
            {key: item[key] for key in ("section_id", "title", "source_title", "source_url")}
            for item in selected_sections
        ],
        "timeline_candidates": raw_timeline_dicts,
        "agents": [item.model_dump() for item in agents],
        "timeline": [item.model_dump() for item in timeline],
        "timeline_agent": timeline_agent_artifact,
        "name_mapping": figure_result["name_mapping"],
        "profiles": figure_result.get("profiles") or [],
        "profiles_view": figure_result.get("profiles_view") or {},
        "layer_quality": figure_result.get("quality") or {},
        "extraction_manifest": {
            "run_id": recorder.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_sections": len(sections),
            "selected_sections": len(selected_sections),
            "raw_timeline_events": len(raw_timeline_dicts),
            **({key: timeline_agent_artifact[key] for key in ("extracted_candidates", "evidence_rejected")
                if key in timeline_agent_artifact}),
            "timeline_events": len(timeline),
            "timeline_events_missing_month": missing_month_events,
            "timeline_source": timeline_source,
            "ordinary_agents": len(agents),
            "profile_status": (stage_result.get("quality") or {}).get("profile_status", "completed"),
            "profile_error": (stage_result.get("quality") or {}).get("profile_error", ""),
            "actor_coverage": (stage_result.get("quality") or {}).get("actor_coverage", {}),
            "timeline_cache_hit": cached_timeline,
            "stage_timings_seconds": {
                "timeline": round(timeline_elapsed, 3),
                "actors_and_profiles": round(actor_profile_elapsed, 3),
                "actor_profile_steps": (stage_result.get("quality") or {}).get("stage_timings_seconds", {}),
                "total": round(time.perf_counter() - pipeline_started, 3),
            },
        },
    }
    report_progress("persisting_artifacts", 96, "Persisting validated extraction artifacts...")
    for name, value in artifacts.items():
        events.save_artifact(request.event_id, name, value)
    graph_repository = GraphRepository(events)
    for agent in agents:
        dumped = agent.model_dump()
        graph_repository.add_node(
            request.event_id,
            {
                "id": agent.agent_id,
                "label": agent.name,
                "type": "agent",
                "properties": {
                    "role": dumped.get("role") or "",
                    "title": dumped.get("title") or "",
                    "affiliation": dumped.get("affiliation") or "",
                    "description": dumped.get("description") or "",
                    "agent_type": dumped.get("agent_type") or "",
                    "decision_subject": dumped.get("decision_subject") or "",
                },
            },
        )
    removed_agent_ids = graph_repository.remove_unregistered_agents(
        request.event_id,
        {item.agent_id for item in agents},
    )
    audit_record(
        recorder,
        source="knowledge-extraction",
        round_id="extraction",
        agent_id="all",
        content_type="persisted-graph-after-registry-cleanup",
        status="success",
        data={
            "removed_agent_ids": removed_agent_ids,
            "graph": graph_repository.get(request.event_id),
        },
    )
    SnapshotRepository(events).clear(request.event_id)
    # Final timeline contains the normalized actor/action links from the layered
    # stage. Merge that enrichment into the full historical corpus.
    enriched = {e["timeline_id"]: e for e in artifacts["timeline"]}
    full = events.load_artifact(request.event_id, "timeline_full", artifacts["timeline"])
    events.save_artifact(request.event_id, "timeline_full", [enriched.get(e["timeline_id"], e) for e in full])
    try:
        await asyncio.to_thread(KnowledgeStore(events).publish, request.event_id)
    except Exception as exc:
        audit_record(recorder, source="okf-export", round_id="extraction", agent_id="all",
                     content_type="publication-error", status="warning", data={"error": str(exc)})
    response = {
        "figures": artifacts["agents"],
        "timeline": artifacts["timeline"],
        "name_mapping": figure_result["name_mapping"],
        "profiles": artifacts["profiles"],
        "profiles_view": artifacts["profiles_view"],
        "quality": artifacts["extraction_manifest"],
        "run_id": recorder.run_id,
    }
    audit_finish(
        recorder,
        success=True,
        data={
            "timeline_events": len(timeline),
            "timeline_source": timeline_source,
            "agents": len(agents),
            "artifacts": list(artifacts),
        },
    )
    return response


def extraction_resume_budget(request: FiguresRequest, events: EventRepository):
    """Validate before queuing or dispatching a paid request; never reset charges."""
    if not request.resume_profiles:
        return None
    if request.force:
        raise ValueError('Choose either resume or reanalyze.')
    cache = events.load_artifact(request.event_id, 'timeline_cache', {})
    sources = events.load_sources(request.event_id)
    if not _saved_timeline_matches(request, sources, cache, _timeline_cache_key(request, sources)):
        raise ValueError('The saved timeline differs from the current model, sources, or timeline settings. Restore the original settings to resume.')
    previous = events.load_artifact(request.event_id, 'extraction_budget', {})
    # Omitted API fields inherit original hard limits/pricing. The UI sends
    # its visible request/token caps explicitly; never silently remove a cost cap.
    inherited = {
        'max_model_calls': previous.get('limits', {}).get('max_calls'),
        'max_total_tokens': previous.get('limits', {}).get('max_total_tokens'),
        'max_cost_usd': previous.get('limits', {}).get('max_cost_usd'),
        'input_price_per_million': previous.get('pricing', {}).get('input'),
        'output_price_per_million': previous.get('pricing', {}).get('output'),
        'cached_input_price_per_million': previous.get('pricing', {}).get('cached'),
    }
    for field, value in inherited.items():
        if field not in request.model_fields_set and (value is not None or field.endswith('_usd') or 'price_' in field or (field == 'max_model_calls' and 'max_calls' in previous.get('limits', {})) or (field == 'max_total_tokens' and 'max_total_tokens' in previous.get('limits', {}))):
            setattr(request, field, value)
    if request.max_model_calls is not None and previous.get('actual_http_requests', 0) >= request.max_model_calls:
        raise ValueError(f"The task has already used {previous.get('actual_http_requests', 0)} requests. Raise the total request limit to resume.")
    if request.max_total_tokens is not None and previous.get('accounted_tokens_upper', 0) >= request.max_total_tokens:
        raise ValueError('The task reached its token limit. Raise the total token limit to resume.')
    ModelBudget(request, previous)  # Includes pricing consistency validation.
    return previous


async def extract_figures(request: FiguresRequest, events: EventRepository) -> dict[str, Any]:
    """One task-local budget spans planner, timeline and layered profile calls."""
    previous = extraction_resume_budget(request, events)
    budget = ModelBudget(request, previous, on_change=lambda value: events.save_artifact(request.event_id, "extraction_budget", value))
    try:
        started = time.monotonic()
        with budget_scope(budget):
            async with asyncio.timeout(request.extraction_deadline_seconds):
                result = await _extract_figures(request, events)
            if events.load_artifact(request.event_id, "forecast_spec", None):
                from app.domain import LLMOptions, SummaryRequest
                from app.knowledge.brief import FORECAST_BRIEF_VERSION
                from app.knowledge.operations import combine_summary

                existing = events.load_artifact(request.event_id, "background_brief", {}) or {}
                remaining = (request.extraction_deadline_seconds or float("inf")) - (time.monotonic() - started)
                if existing.get("version") != FORECAST_BRIEF_VERSION and remaining > 0:
                    try:
                        async with asyncio.timeout(min(15, remaining)):
                            await combine_summary(SummaryRequest(
                                **request.model_dump(include=set(LLMOptions.model_fields)),
                                event_id=request.event_id, event_title=request.event_title, article_summaries=[],
                            ), events=events)
                    except (LLMError, BudgetExceeded, TimeoutError, ValueError):
                        pass  # Keep the cited preview if optional synthesis is unavailable.
        full = events.load_artifact(request.event_id, "timeline_full", [])
        manifest = result["quality"]
        sources = events.load_sources(request.event_id)
        manifest.update(evidence_policy=EVIDENCE_VERSION if request.timeline_mode=="parallel" else "legacy_unverified",
            history_fingerprint=digest(full), profile_fingerprint=digest(events.load_artifact(request.event_id,"agents",[])),
            profile_source_fingerprint=digest([{ "url":s.get("url"),"content":s.get("content"),"sections":s.get("sections",[])} for s in sources]),
            extraction_plan=events.load_artifact(request.event_id,"extraction_plan",{}),
            evidence_review=events.load_artifact(request.event_id,"timeline_evidence_review",{}),
            budget=budget.report())
        events.save_artifact(request.event_id,"extraction_manifest",manifest)
        # Publish once more after the policy marker is present; the knowledge
        # version now records exactly which profile inputs passed the gate.
        await asyncio.to_thread(KnowledgeStore(events).publish,request.event_id)
        limited = manifest.get('profile_status') == 'limited'
        ExtractionStatusRepository(events).update(request.event_id, run_id=result['run_id'],
            stage='profiles_pending' if limited else 'completed', progress=95 if limited else 100,
            status='partial' if limited else 'completed',
            error=manifest.get('profile_error', '') if limited else '',
            message=('The timeline and available profiles were saved. Required decision actors or their decision states need review; resume to complete preparation.'
                     if limited else f"Extraction completed with {len(result['timeline'])} timeline events and {len(result['figures'])} Agents."
                     if result['figures'] else 'The timeline was saved. The available evidence did not identify actors eligible for simulation.'))
        return result
    except Exception as exc:
        statuses = ExtractionStatusRepository(events)
        state = statuses.get(request.event_id)
        cache = events.load_artifact(request.event_id, 'timeline_cache', {})
        resumable = bool(cache.get('timeline') or (cache.get('allow_empty_timeline') and cache.get('cache_key')))
        statuses.update(request.event_id, run_id=state.get('run_id') or '',
            stage='profiles_pending' if resumable else state.get('stage', 'failed'),
            progress=min(95, state.get('progress', 0)), status='partial' if resumable else 'failed',
            message='The timeline and actor extraction were saved. Resume to finish the profiles.' if resumable else 'Extraction is incomplete.',
            error=str(exc))
        raise
    finally:
        events.save_artifact(request.event_id,"extraction_budget",budget.report())
