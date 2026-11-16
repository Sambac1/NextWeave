from __future__ import annotations

import asyncio
import re

import pytest

from app.domain import FiguresRequest
from app.knowledge import extraction
from app.storage.json_store import EventRepository


class Recorder:
    run_id = "extract-test"


class ExtractionClient:
    def __init__(self, fail_section: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_section = fail_section

    async def json_call(self, **kwargs):
        name = kwargs["call_name"]
        schema = kwargs["schema"]
        self.calls.append(name)
        if schema.__name__ == "SectionSummary":
            return schema.model_validate({
                "summary": "Alice made a dated decision.",
                "has_accurate_dates": True,
                "timeline_extractable": True,
            })
        if schema.__name__ == "ChapterEvaluations":
            paths = list(dict.fromkeys(re.findall(r"source_\d+/[^:\n]+", kwargs["prompt"])))
            chapter = paths[0].rsplit("/", 1)[0] if paths else "source_01/Document"
            return schema.model_validate({chapter: {"relevant": True, "timeline_suitable": True}})
        if schema.__name__ == "SectionPathSelection":
            paths = list(dict.fromkeys(re.findall(r"source_\d+/[^:\n]+", kwargs["prompt"])))
            return schema.model_validate({"selected": paths})
        if schema.__name__ == "TimelineBatch":
            section_id = re.search(r"source_01_section_\d{3}", name).group(0)
            if self.fail_section and self.fail_section in section_id:
                raise RuntimeError(f"{section_id} extraction failed")
            return schema.model_validate({"events": [
                {"timestamp": "2026-01-01", "description": f"{section_id} first event", "date_precision": "day", "evidence_text": "evidence one", "actors": ["Alice"], "event_type": "policy", "confidence": 0.9, "importance_score": 0.7},
                {"timestamp": "2026-01-02", "description": f"{section_id} second event", "date_precision": "day", "evidence_text": "evidence two", "actors": ["Alice"], "event_type": "communication", "confidence": 0.8, "importance_score": 0.6},
            ]})
        if schema.__name__ == "TimelineDeduplication":
            events = re.findall(r'\{[^{}]*"timestamp"[^{}]*\}', kwargs["prompt"])
            parsed = []
            for value in events:
                try:
                    parsed.append(__import__("json").loads(value))
                except ValueError:
                    pass
            return schema.model_validate({"deduplicated": parsed})
        raise AssertionError(name)


AGENT_HISTORICAL = [{
    "timestamp": "2026-01-01",
    "description": "Alice made a dated decision.",
    "date_precision": "day",
    "evidence_text": "evidence one",
    "actors": ["Alice"],
    "event_type": "policy",
    "confidence": 0.9,
    "importance_score": 0.7,
}]


class TimelineAgentClient(ExtractionClient):
    """Answers the one-shot Timeline Agent call, then behaves like the base client.

    Keyed on the submitted schema rather than the call name so the double works
    whichever client the timeline agent is handed.
    """

    def __init__(self, historical: list[dict] | None = None) -> None:
        super().__init__()
        self.historical = AGENT_HISTORICAL if historical is None else historical

    async def json_call(self, **kwargs):
        if kwargs["schema"].__name__ == "TimelineAgentResult":
            self.calls.append(kwargs["call_name"])
            return kwargs["schema"].model_validate({
                "historical": self.historical,
                "analysis": "Reading of the record.",
            })
        return await super().json_call(**kwargs)


def event_with_sections(tmp_path):
    events = EventRepository(tmp_path)
    event_id = events.create("Event", ["https://example.test"], None)["event_id"]
    events.save_sources(event_id, [{
        "title": "Source", "url": "https://example.test",
        "content": "\n".join(f"== Section {i} ==\nAlice acted on 2026-01-01." for i in range(1, 13)),
        "sections": [
            {"section_id": f"source_01_section_{index:03d}", "title": f"Section {index}", "content": "Alice acted on 2026-01-01."}
            for index in range(1, 13)
        ],
    }])
    return events, event_id


def _stub_layer_stage(monkeypatch, agents=None, profiles=None):
    """Replace the layered-figure stage with a canned result.

    Timeline extraction and the stage use different LLM paths -- the framework's
    ``client.json_call`` versus the stage's own HTTP client -- so the stage is
    stubbed at its boundary rather than through the knowledge client.
    """
    if agents is None:
        agents = [_stub_agent()]
    result = {
        "agents": agents,
        "profiles": profiles or [],
        "profiles_view": {},
        "name_mapping": {"Alice": "Alice"},
        "quality": {},
        "timeline_quality": {},
        "profile_path": "",
        "fallback_warning": "",
        "profile_warning": "",
    }

    async def _run_stage(*args, **kwargs):
        return result

    monkeypatch.setattr(extraction, "run_layer_stage", _run_stage)
    return result


def _stub_agent():
    from app.domain import AgentProfile
    from app.domain.agents import Beliefs, DecisionPolicy

    return AgentProfile(
        agent_id="agent_0000000000000001",
        name="Alice",
        agent_type="person",
        decision_subject="Alice",
        role="initiator",
        description="Alice",
        beliefs=Beliefs(),
        decision_policy=DecisionPolicy(
            risk_tolerance=0.4, compromise_tendency=0.5, response_speed=0.6
        ),
    )


def test_extraction_batches_timeline_deduplication_and_commits_artifacts(tmp_path, monkeypatch) -> None:
    events, event_id = event_with_sections(tmp_path)
    client = ExtractionClient()
    monkeypatch.setattr(extraction, "make_client", lambda request, category: (client, Recorder()))
    monkeypatch.setattr(extraction, "AgentSDKClient", lambda request, recorder: client)
    _stub_layer_stage(monkeypatch)
    result = asyncio.run(extraction.extract_figures(
        FiguresRequest(event_id=event_id, event_title="Event", api_key="x", model="m", timeline_mode="legacy_agent", use_timeline_agent=False),
        events,
    ))
    dedup_calls = [name for name in client.calls if name.startswith("timeline_deduplication_")]
    assert dedup_calls
    assert all("_batch_" in name for name in dedup_calls)
    assert all(not name.startswith("timeline_deduplication_all") for name in dedup_calls)
    assert result["figures"][0]["agent_id"].startswith("agent_")
    assert not (events.directory(event_id) / "figures.json").exists()
    assert (events.directory(event_id) / "agents.json").exists()
    assert (events.directory(event_id) / "extraction_manifest.json").exists()
    assert events.load_artifact(event_id, "extraction_status", {})["status"] == "completed"
    manifest = events.load_artifact(event_id, "extraction_manifest", {})
    assert manifest["timeline_source"] == "per_section"
    # Twelve selected sections, two events each from the per-section double.
    assert manifest["timeline_events"] == 24
    # The switch was off, so nothing was attempted -- but the artifact still
    # records that fact instead of leaving the run's path undocumented.
    assert events.load_artifact(event_id, "timeline_agent", {})["applied"] is False


def test_timeline_agent_path_skips_deduplication_and_keeps_history_only(tmp_path, monkeypatch) -> None:
    events, event_id = event_with_sections(tmp_path)
    client = TimelineAgentClient()
    monkeypatch.setattr(extraction, "make_client", lambda request, category: (client, Recorder()))
    monkeypatch.setattr(extraction, "AgentSDKClient", lambda request, recorder: client)
    _stub_layer_stage(monkeypatch)
    result = asyncio.run(extraction.extract_figures(
        FiguresRequest(event_id=event_id, event_title="Event", api_key="x", model="m", timeline_mode="legacy_agent"),
        events,
    ))
    assert "timeline_agent_build" in client.calls
    # One pass over every selected section replaces both the per-section
    # extraction and the deduplication batches.
    assert not [name for name in client.calls if name.startswith("timeline_deduplication_")]
    timeline = result["timeline"]
    assert [item["origin"] for item in timeline] == ["historical"]
    assert timeline[0]["description"] == AGENT_HISTORICAL[0]["description"]
    agent_artifact = events.load_artifact(event_id, "timeline_agent", {})
    assert agent_artifact["applied"] is True
    assert agent_artifact["stats"]["historical_events"] == 1
    assert agent_artifact["sections_sent"]
    manifest = events.load_artifact(event_id, "extraction_manifest", {})
    assert manifest["timeline_source"] == "timeline_agent"
    assert manifest["timeline_events"] == 1
    assert not (events.directory(event_id) / "timeline_projected.json").exists()
    assert not (events.directory(event_id) / "timeline_history.json").exists()


def test_timeline_agent_without_usable_history_falls_back_to_per_section(tmp_path, monkeypatch) -> None:
    events, event_id = event_with_sections(tmp_path)
    client = TimelineAgentClient(historical=[])
    monkeypatch.setattr(extraction, "make_client", lambda request, category: (client, Recorder()))
    monkeypatch.setattr(extraction, "AgentSDKClient", lambda request, recorder: client)
    _stub_layer_stage(monkeypatch)
    result = asyncio.run(extraction.extract_figures(
        FiguresRequest(event_id=event_id, event_title="Event", api_key="x", model="m", timeline_mode="legacy_agent"),
        events,
    ))
    assert "timeline_agent_build" in client.calls
    assert [name for name in client.calls if name.startswith("timeline_deduplication_")]
    assert all(item["origin"] == "historical" for item in result["timeline"])
    assert events.load_artifact(event_id, "timeline_agent", {})["applied"] is False
    assert events.load_artifact(event_id, "extraction_manifest", {})["timeline_source"] == "per_section"


def test_a_capture_from_the_projected_era_still_loads(tmp_path) -> None:
    """A snapshot written before the projected half was removed keeps working."""
    from app.domain import TimelineEvent

    event = TimelineEvent.model_validate({
        "timeline_id": "proj_legacy",
        "timestamp": "2030-01-01",
        "description": "A projection captured before the removal",
        "origin": "projected",
        "confidence": 0.5,
        "importance": 0.5,
    })
    assert event.origin == "projected"


def test_failed_section_does_not_commit_partial_domain_artifacts(tmp_path, monkeypatch) -> None:
    """A failing timeline section must abort the run, not persist half a result."""
    events, event_id = event_with_sections(tmp_path)
    client = ExtractionClient(fail_section="section_005")
    monkeypatch.setattr(extraction, "make_client", lambda request, category: (client, Recorder()))
    monkeypatch.setattr(extraction, "AgentSDKClient", lambda request, recorder: client)
    _stub_layer_stage(monkeypatch)
    with pytest.raises(RuntimeError, match="section_005 extraction failed"):
        asyncio.run(extraction.extract_figures(
            FiguresRequest(event_id=event_id, event_title="Event", api_key="x", model="m", timeline_mode="legacy_agent", use_timeline_agent=False),
            events,
        ))
    assert not (events.directory(event_id) / "agents.json").exists()
    assert not (events.directory(event_id) / "timeline.json").exists()


def test_cached_timeline_completes_without_model_calls_and_preserves_date_counts(tmp_path, monkeypatch):
    events, event_id = event_with_sections(tmp_path)
    client = TimelineAgentClient(historical=[
        *AGENT_HISTORICAL,
        {**AGENT_HISTORICAL[0], "timestamp": "2025", "description": "No resolvable month"},
    ])
    monkeypatch.setattr(extraction, "make_client", lambda request, category: (client, Recorder()))
    monkeypatch.setattr(extraction, "AgentSDKClient", lambda request, recorder: client)
    _stub_layer_stage(monkeypatch)
    request = FiguresRequest(event_id=event_id, event_title="Event", api_key="x", model="m", timeline_mode="legacy_agent")
    first = asyncio.run(extraction.extract_figures(request, events))
    client.calls.clear()
    cached = asyncio.run(extraction.extract_figures(request, events))
    assert not client.calls
    assert cached["quality"]["timeline_cache_hit"] is True
    assert cached["timeline"] == first["timeline"]
    assert cached["quality"]["timeline_events_missing_month"] == first["quality"]["timeline_events_missing_month"] == 1
    assert events.load_artifact(event_id, "extraction_status", {})["status"] == "completed"
