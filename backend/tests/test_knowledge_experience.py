import asyncio
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import TimelineEvent, WorldSnapshot, ActionProposal
from app.domain.behavior import grounded_behavior
from app.knowledge.evidence import locate_quote, validate_event
from app.knowledge.okf import KnowledgeStore
from app.layers.pipeline.profile_service import _format_history, _format_actions
from app.layers.pipeline.seams import _policy_from_style, to_stage_timeline
from app.simulation.context import (
    actor_context,
    audit_context,
    public_context,
    behavior_context,
    compact_json,
)
from app.simulation.ensemble import attach_evidence
from test_ensemble import actor
from test_okf import corpus


def test_ellipsis_repairs_keep_unique_real_contiguous_source_spans():
    text = "On October 16, KWQC-TV reported that many small businesses in the Quad Cities area (near the plants) were offering discounts."
    quote = "On October 16, KWQC-TV reported that many small businesses in the Quad Cities area ... were offering discounts"
    span = locate_quote(text, quote)
    assert span and text[span[0] : span[1]].endswith("were offering discounts")
    event, reason = validate_event(
        {"timestamp": "2021-10-16", "evidence_text": quote},
        {"content": text, "source_title": "2021 strike"},
    )
    assert reason == "accepted" and event["evidence_text"] == text[span[0] : span[1]]
    assert event["date_evidence"]["quote_resolution"] == "ordered_ellipsis_restored"
    assert locate_quote(text + "\n" + text, quote) is None
    assert (
        locate_quote(text.replace("(near the plants)", "\n(near the plants)"), quote)
        is None
    )


def test_repeated_ellipsis_adversarial_input_has_bounded_runtime():
    started = time.monotonic()
    for _ in range(50):
        assert (
            locate_quote(
                ("abcdefghij ") * 200, " ... ".join(["abcdefghij"] * 4 + ["zzzzzzzzzz"])
            )
            is None
        )
    assert time.monotonic() - started < 1


def test_only_explicit_start_end_pairs_can_split_two_event_dates():
    source = {
        "content": "The strike began on October 14 and ended on November 17.",
        "source_title": "2021 strike",
    }
    result, _ = validate_event(
        {
            "timestamp": "2021-10-14",
            "description": "The strike began.",
            "date_original": "October 14",
            "evidence_text": source["content"],
        },
        source,
    )
    assert result and result["timestamp"] == "2021-10-14"
    assert result["date_evidence"]["date_resolution"] == "explicit_start_end_clause"
    other = {
        "content": "The strike began on October 14 and another strike began on November 17.",
        "source_title": "2021 strike",
    }
    assert (
        validate_event(
            {
                "timestamp": "2021-10-14",
                "description": "The strike began.",
                "date_original": "October 14",
                "evidence_text": other["content"],
            },
            other,
        )[0]
        is None
    )


def test_exported_quotes_share_validator_offsets_and_profiles_have_actual_activities(
    tmp_path,
):
    events, case_id = corpus(tmp_path)
    history = events.load_artifact(case_id, "timeline", [])[:1]
    history[0]["retrieval_actors"] = [
        {
            "name": "Local newspaper",
            "entity_type": "company",
            "role": "Reported the trial",
            "title": "Regional publisher",
        }
    ]
    history.append(
        history[0]
        | {
            "timeline_id": "hist_2",
            "timestamp": "2020-02",
            "description": "A bank funded the trial.",
            "date_original": "Q4 2020",
            "retrieval_actors": [
                {"name": "Bank", "entity_type": "company", "role": "Financier"}
            ],
        }
    )
    events.save_artifact(case_id, "timeline_full", history)
    store = KnowledgeStore(events)
    manifest = store.publish(case_id)
    profiles = store.search(case_id, kind="actor", limit=100)
    report = next(p for p in profiles if p["title"] == "Local newspaper")
    bank = next(p for p in profiles if p["title"] == "Bank")
    assert report["metadata"]["description"] != bank["metadata"]["description"]
    card = report["metadata"]["profile"]
    assert card["title"] == "Regional publisher" and card["event_count"] == 1
    assert card["observed_activities"][0]["timeline_id"] == "hist_1"
    assert bank["metadata"]["profile"]["event_count"] == 0
    assert bank["metadata"]["profile"]["date_range"] == ["", ""]
    assert bank["metadata"]["profile"]["background_records"]
    assert store.record(case_id, "hist_2")["metadata"]["timestamp"] == ""
    assert "beliefs" not in card and "private_observations" not in card
    record = store.record(case_id, "hist_1")
    assert record["metadata"]["actor_names"][report["id"]] == "Local newspaper"
    from app.knowledge.okf import PROJECTION_VERSION
    assert store.manifest(case_id)["projection_version"] == PROJECTION_VERSION
    with pytest.raises(FileNotFoundError):
        store.record(case_id, "../../private")
    from app.api import experiments

    app = FastAPI()
    app.include_router(experiments.router)
    from unittest.mock import patch

    with patch.object(experiments, "KnowledgeStore", lambda: store):
        response = TestClient(app).get(f"/api/knowledge/{case_id}/records/hist_1")
    assert response.status_code == 200 and response.json()["id"] == "hist_1"


def test_behavior_hypotheses_are_separate_dimensions_with_validated_refs():
    model = grounded_behavior(
        {
            "decision_patterns": [
                {
                    "dimension": "initiative",
                    "description": "Acts before the deadline",
                    "timeline_refs": ["h1", "fabricated"],
                    "status": "observed",
                }
            ],
            "capabilities": "invalid",
        },
        ["h1"],
    )
    statement = model.decision_patterns[0]
    assert statement.status == "inferred" and statement.timeline_refs == ["h1"]
    assert not model.capabilities
    policy = _policy_from_style({"preferences": {"action_pace": "proactive"}})
    assert policy.risk_tolerance == 0.5 and policy.response_speed == 0.5
    assert "h1" in _format_history(
        [{"timeline_id": "h1", "timestamp": "2020-01", "description": "A decision"}]
    )
    assert "h1" in _format_actions(
        [{"timeline_id": "h1", "source": "Lab", "action": "decided"}]
    )


def test_behavior_references_retrieve_older_evidence_and_default_context_stays_bounded():
    agents = []
    for i in range(8):
        a = actor().model_copy(
            update={
                "agent_id": f"actor_{i}",
                "name": f"Institution {i}",
                "description": "Scenario-specific organization. " * 12,
            }
        )
        claims = {
            key: [
                {
                    "description": "A contextual hypothesis with conditions. " * 4,
                    "scope": "contract negotiations",
                    "conditions": "Only while this contract is in force",
                    "timeline_refs": ["old"],
                }
                for _ in range(8)
            ]
            for key in (
                "contextual_roles",
                "capabilities",
                "decision_patterns",
                "objectives",
            )
        }
        a.behavioral_model = grounded_behavior(claims, ["old"])
        assert len(compact_json(behavior_context(a))) < 1900
        agents.append(a)
    timeline = [
        TimelineEvent(
            timeline_id="old" if i == 0 else f"e{i}",
            timestamp="2020-01",
            description="Evidence for a hypothesis",
            evidence_text="An observed decision.",
            confidence=1,
            importance=1,
        )
        for i in range(25)
    ]
    snapshot = WorldSnapshot(event_title="A scenario", agents=agents, timeline=timeline)
    attach_evidence(snapshot, snapshot.model_copy(deep=True), [], "", 3)
    payload = public_context(snapshot, "decision", agents[0])
    assert "old" in {e["timeline_id"] for e in payload["timeline"]}
    assert (
        len(compact_json(payload)) + len(compact_json(actor_context(agents[0]))) < 24000
    )
    proposals = [
        ActionProposal(
            agent_id=a.agent_id,
            agent_name=a.name,
            round_id=1,
            snapshot_version=0,
            action_id=f"action_1_{a.agent_id}",
            decision="act",
            action_type="proposal",
            action="Propose a discussion",
            reasoning="Within the current scenario",
            confidence=0.5,
            timeline_refs=["old"],
        )
        for a in agents
    ]
    audit = audit_context(snapshot, proposals)
    assert "old" in {e["timeline_id"] for e in audit["public"]["timeline"]}
    assert (
        len(compact_json(audit))
        + len(compact_json([p.model_dump() for p in proposals]))
        < 23000
    )


def test_action_parent_keeps_announcement_role_and_date():
    event = TimelineEvent(
        timeline_id="h",
        timestamp="2024-03",
        description="Announced a policy effective in April",
        time_role="announcement",
        modality="announced",
        confidence=1,
        importance=1,
    )
    stage = to_stage_timeline([event])[0]
    assert stage["time_role"] == "announcement" and stage["timestamp"] == "2024-03"


def test_actor_identity_survives_retrieval_export_without_private_data():
    from app.layers.pipeline.seams import _retrieval_actor
    from app.knowledge.hierarchy import public_profiles

    row = _retrieval_actor(
        "Outlet",
        {
            "entity_type": "company",
            "role": "Reports the event",
            "private_observations": ["PRIVATE"],
            "beliefs": {"secret": "PRIVATE"},
        },
    )
    assert (
        row["entity_type"] == "company"
        and "private_observations" not in row
        and "beliefs" not in row
    )
    raw = actor().model_dump() | {
        "agent_type": "person",
        "description": "Company producing equipment",
    }
    profiles, _ = public_profiles([raw], "case", "Case")
    assert (
        profiles[0]["agent_type"] == "company"
        and profiles[0]["entity_type_status"] == "identity_descriptor_correction"
    )
    assert raw["agent_type"] == "person"  # original model record stays untouched
