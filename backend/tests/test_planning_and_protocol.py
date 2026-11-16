import asyncio
import json
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.domain import TimelineEvent, WorldSnapshot, PrivateObservation, ActionProposal
from app.domain.protocol import ScenarioProtocol
from app.knowledge.planning import (
    assess_sections,
    ExtractionPlan,
    ChapterJudgments,
    plan_extraction,
)
from app.knowledge.coverage import extract_coverage, _potential_duplicates
from app.knowledge.timeline import TimelineBatch
from app.simulation.context import actor_context, public_context
from app.simulation.protocol import (
    prepare_step,
    program_projection,
    validate_program_projection,
    public_digest,
    can_advance_quiet,
)
from app.simulation.bdi_updates import process_agent_information, InformationReview
from test_ensemble import actor


def test_full_parallel_coverage_retains_more_than_sixty_source_supported_events(
    tmp_path,
):
    sections = [
        {
            "section_id": f"s{i}",
            "path": f"source_01/2024/{i}",
            "title": str(i),
            "content": f"{date(2024,1,1)+timedelta(days=i)}: event_{i} occurred.",
        }
        for i in range(75)
    ]

    class Client:
        calls = 0
        model = "fixture"

        async def json_call(self, **kw):
            self.calls += 1
            i = int(kw["call_name"].split("section_s")[1].split("_")[0])
            return TimelineBatch(
                events=[
                    {
                        "timestamp": str(date(2024, 1, 1) + timedelta(days=i)),
                        "description": f"event_{i}",
                        "evidence_text": f"event_{i} occurred.",
                    }
                ]
            )

    client = Client()
    result = asyncio.run(
        extract_coverage(
            sections, client, tmp_path, lambda *_: None, planning_mode="rules"
        )
    )
    assert len(result.historical) == 75
    again = asyncio.run(
        extract_coverage(
            sections, client, tmp_path, lambda *_: None, planning_mode="rules"
        )
    )
    assert len(again.historical) == 75 and client.calls == 75
    assert _potential_duplicates(
        [
            {"description": "工人拒绝了公司的合同提案"},
            {"description": "公司合同提案遭工人拒绝"},
        ]
    )


def test_chapter_classifier_cannot_skip_unread_passages_or_misspelled_coverage(
    tmp_path,
):
    class Client:
        model = "fixture"

        async def json_call(self, **kw):
            payload = __import__("ast").literal_eval(kw["prompt"])
            return ChapterJudgments(
                sections=[
                    {
                        "section_id": p["section_id"],
                        "decision": "skip",
                        "reason": "static",
                    }
                    for p in payload
                ]
            )

    data = [
        {
            "section_id": "long",
            "path": "source_01/History",
            "content": "Static description. " * 300 + " Workers walked out.",
        },
        {
            "section_id": "short",
            "path": "source_01/2024-03",
            "content": "The port closed.",
        },
    ]
    selected = asyncio.run(assess_sections(data, Client(), tmp_path, ExtractionPlan()))
    assert any(s["section_id"] == "long" for s in selected)
    # Source-content hash must change the decision cache even if snippets don't.
    paths = set((tmp_path / "chapter_cache").glob("*.json"))
    data[0]["content"] = (
        data[0]["content"][:1800] + " EXTRA EVENT " + data[0]["content"][1800:]
    )
    asyncio.run(assess_sections(data, Client(), tmp_path, ExtractionPlan()))
    assert set((tmp_path / "chapter_cache").glob("*.json")) != paths


def test_private_memory_and_new_public_facts_survive_context_compression():
    a = actor()
    a.public_state = {
        "semantic": {"duplicate": "x" * 20000},
        "source_profile": {"secret": "UNRELATED_PRIVATE"},
    }
    a.private_observations = [
        PrivateObservation(
            observation_id="p",
            description="OWN_PRIVATE",
            received_round=1,
            processed=True,
        )
    ]
    old = [
        TimelineEvent(
            timeline_id=f"h{i}",
            timestamp="2020-01",
            description="History",
            confidence=1,
            importance=1,
        )
        for i in range(15)
    ]
    a.relevant_timeline_refs = [e.timeline_id for e in old]
    new = TimelineEvent(
        timeline_id="new",
        timestamp="next",
        description="New offer",
        origin="simulated",
        round_id=1,
        confidence=1,
        importance=1,
    )
    snapshot = WorldSnapshot(
        event_title="Case",
        agents=[a],
        timeline=[*old, new],
        round_id=1,
        world_state={"stock": 7},
    )
    public = public_context(snapshot, "decision", a)
    assert "new" in {e["timeline_id"] for e in public["timeline"]}
    assert "OWN_PRIVATE" in json.dumps(actor_context(a, observations=True))
    assert "PRIVATE" not in json.dumps(public)
    assert (
        len(json.dumps(public)) + len(json.dumps(actor_context(a)))
        < len(snapshot.model_dump_json()) * 0.4
    )


def test_protocol_rates_schedules_and_program_owned_fields_are_consistent():
    protocol = ScenarioProtocol(
        rates=[{"path": "stock", "delta_per_step": -1}],
        scheduled_observations=[{"step": 1, "description": "A deadline was announced"}],
        action_mechanisms=[
            {
                "action_type": "ship",
                "preconditions": [{"path": "stock", "operator": "ge", "value": 1}],
                "effects": [{"path": "stock", "delta_per_step": -1}],
            }
        ],
        idle_policy="advance_if_unchanged",
    )
    baseline = WorldSnapshot(
        event_title="Supply",
        agents=[actor()],
        world_state={"stock": 10, "scenario_protocol": protocol.model_dump()},
    )
    working = prepare_step(baseline, protocol, 1)
    assert baseline.world_state["stock"] == 10 and working.world_state["stock"] == 9
    expected = program_projection(working, [])
    with pytest.raises(ValueError):
        validate_program_projection(expected, {"stock": 0})
    later = prepare_step(working, protocol, 2)
    assert any(
        o.get("description") == "A deadline was announced"
        for o in later.world_state["program_observations"]
    )
    record = {
        "quiet_candidate": True,
        "post_round_public_digest": public_digest(working),
    }
    assert not can_advance_quiet(later, record, protocol)
    with pytest.raises(ValidationError):
        ScenarioProtocol(
            action_mechanisms=[{"action_type": "ship", "precondition": []}]
        )


def test_first_step_scheduled_information_is_processed_and_unchanged_info_is_not_repeated():
    class Client:
        calls = 0

        async def json_call(self, **kw):
            self.calls += 1
            return InformationReview(
                agent_id="actor_a", round_id=1, snapshot_version=0, assessment="Noted"
            )

    protocol = ScenarioProtocol(
        scheduled_observations=[{"step": 1, "description": "A new deadline"}]
    )
    snapshot = prepare_step(
        WorldSnapshot(event_title="Case", agents=[actor()]), protocol, 1
    )
    client = Client()
    updated = asyncio.run(process_agent_information(snapshot, client, "test"))
    assert client.calls == 1
    asyncio.run(process_agent_information(updated, client, "test"))
    assert client.calls == 1
