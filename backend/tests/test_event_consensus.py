from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.forecasts import router
from app.domain import (
    ConsensusStateProjection, EventGrouping, EventMatchGroup,
    EventReference, ForecastRound, RelationshipProjection, SimulationRequest,
    TimelineEvent, WorldSnapshot,
)
from app.functional_agents.event_consensus import EventConsensusAgent
from app.simulation import consensus as consensus_module
from app.simulation import engine


def event(identifier: str, description: str = "An action") -> TimelineEvent:
    return TimelineEvent(
        timeline_id=identifier, timestamp="next", description=description,
        confidence=0.5, importance=0.5,
    )


def forecast(*events: TimelineEvent) -> ForecastRound:
    return ForecastRound(
        step=1, input_snapshot_version=0, output_snapshot_version=1,
        state_summary="sample-only state", timeline=list(events),
    )


def match_group(description: str, *members: tuple[int, str]) -> EventMatchGroup:
    return EventMatchGroup(
        event=event("model-event", description),
        members=[EventReference(sample_id=i, timeline_id=e) for i, e in members],
        reasoning="The model's semantic judgment",
    )


class Runtime:
    recorder = None

    def __init__(self, groups: list[EventMatchGroup]):
        self.groups = groups
        self.calls: list[dict] = []

    async def json_call(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["schema"]
        if schema is EventGrouping:
            return EventGrouping(groups=self.groups)
        if schema is ConsensusStateProjection:
            payload = json.loads(kwargs["prompt"])
            return ConsensusStateProjection(
                state_summary="Accepted state",
                next_world_state={**payload["previous_world_state"], "accepted": True},
                analysis="Only accepted facts",
            )
        if schema is RelationshipProjection:
            return RelationshipProjection()
        raise AssertionError(f"Unexpected model call: {schema}")


def test_model_groups_paraphrases_and_same_sample_duplicates_count_once():
    samples = [forecast(event("a", "Launch talks"), event("b", "Start negotiations")),
               forecast(event("c", "Open dialogue")), forecast(), forecast(event("d", "Escalate"))]
    runtime = Runtime([
        match_group("Talks begin", (1, "a"), (1, "b"), (2, "c")),
        match_group("Escalation", (4, "d")),
    ])
    result = asyncio.run(EventConsensusAgent(runtime).match(
        WorldSnapshot(event_title="Test"), samples, 50, "test",
    ))
    assert len(runtime.calls) == 1
    assert result.required_occurrences == 2
    assert [(e.occurrence_count, e.retained) for e in result.events] == [(2, True), (1, False)]
    assert result.events[0].sample_ids == [1, 2]
    assert result.events[0].event.source_event_count == 2
    assert result.events[0].event.timeline_id == "sim_1_consensus_001"


@pytest.mark.parametrize("count,percent,required,retained", [
    (4, 50, 2, True), (4, 51, 3, False), (4, 75, 3, False),
    (3, 50, 2, True), (4, 100, 4, False), (4, 1, 1, True),
])
def test_inclusive_threshold_and_ceiling(count, percent, required, retained):
    samples = [forecast(event("a")), forecast(event("b"))] + [forecast() for _ in range(count - 2)]
    result = asyncio.run(EventConsensusAgent(Runtime([
        match_group("Action", (1, "a"), (2, "b")),
    ])).match(WorldSnapshot(event_title="Test"), samples, percent, "test"))
    assert result.required_occurrences == required
    assert result.events[0].retained is retained


def test_identical_text_is_not_locally_merged_against_model_judgment():
    # Identical wording need not mean the same event. The model's groups are authoritative.
    result = asyncio.run(EventConsensusAgent(Runtime([
        match_group("Same words", (1, "a")), match_group("Same words", (2, "b")),
    ])).match(WorldSnapshot(event_title="Test"), [forecast(event("a")), forecast(event("b"))], 100, "test"))
    assert len(result.events) == 2
    assert not any(v.retained for v in result.events)


@pytest.mark.parametrize("groups,error", [
    ([match_group("Unknown", (3, "a"))], "unknown or repeated"),
    ([match_group("Repeated", (1, "a"), (1, "a"))], "unknown or repeated"),
    ([match_group("First", (1, "a")), match_group("Second", (1, "a"))], "unknown or repeated"),
    ([], "omitted"),
])
def test_invalid_model_references_fail_without_local_fallback(groups, error):
    with pytest.raises(ValueError, match=error):
        asyncio.run(EventConsensusAgent(Runtime(groups)).match(
            WorldSnapshot(event_title="Test"), [forecast(event("a"))], 50, "test",
        ))


def test_unknown_actor_is_dropped_and_the_event_survives():
    """A stray label must not cost the event, or the round it belongs to."""
    group = match_group("Action", (1, "a"))
    group.event.actors = ["agent_known", "invented-agent"]
    from app.domain import AgentProfile
    from app.domain.agents import Beliefs, DecisionPolicy

    snapshot = WorldSnapshot(event_title="Test", agents=[AgentProfile(
        agent_id="agent_known", name="Known", agent_type="person",
        decision_subject="Known", role="neutral", description="d",
        beliefs=Beliefs(),
        decision_policy=DecisionPolicy(
            risk_tolerance=0.5, compromise_tendency=0.5, response_speed=0.5),
    )])
    result = asyncio.run(EventConsensusAgent(Runtime([group])).match(
        snapshot, [forecast(event("a"))], 100, "test",
    ))
    assert result.events[0].event.actors == ["agent_known"]


def test_empty_predictions_do_not_call_matcher():
    runtime = Runtime([])
    result = asyncio.run(EventConsensusAgent(runtime).match(
        WorldSnapshot(event_title="Test"), [forecast() for _ in range(4)], 50, "test",
    ))
    assert result.events == []
    assert result.prediction_repeats == 4
    assert runtime.calls == []


def patch_samples(monkeypatch, events_by_sample):
    seen = []

    async def sample(self, snapshot, interventions, call_prefix):
        sample_id = int(call_prefix.rsplit("_", 1)[1])
        seen.append((snapshot.model_dump(), [a.model_dump() for a in interventions]))
        snapshot.world_state["candidate-leak"] = sample_id
        snapshot.state_summary = "Rejected sample summary"
        return snapshot, forecast(*events_by_sample[sample_id - 1])

    monkeypatch.setattr(consensus_module.ForecastRoundOrchestrator, "run", sample)
    return seen


def test_next_step_uses_only_consensus_state_with_isolated_samples(monkeypatch):
    seen = patch_samples(monkeypatch, [
        [event("a"), event("discard", "rejected-event-marker")], [event("b")], [], [],
    ])
    runtime = Runtime([
        match_group("Accepted action", (1, "a"), (2, "b")),
        match_group("rejected-event-marker", (1, "discard")),
    ])
    base = WorldSnapshot(event_title="Test", state_summary="Base", world_state={"baseline": True})
    trajectory = asyncio.run(engine.run_trajectory(base, 2, runtime, "test", [], 4, 50))
    assert len(seen) == 8
    assert all(s[0]["world_state"] == {"baseline": True} for s in seen[:4])
    assert all(s[0]["world_state"] == {"baseline": True, "accepted": True} for s in seen[4:])
    assert [s[0]["round_id"] for s in seen] == [0] * 4 + [1] * 4
    assert base.world_state == {"baseline": True}
    assert base.timeline == []
    final = trajectory.final_snapshot
    assert final.round_id == 2 and final.snapshot_version == 2
    assert [e.description for e in final.timeline] == ["Accepted action", "Accepted action"]
    assert len({e.timeline_id for e in final.timeline}) == 2
    assert all("candidate-leak" not in c["prompt"] and "rejected-event-marker" not in c["prompt"]
               for c in runtime.calls if c["schema"] is not EventGrouping)
    assert all(r.agent_actions == [] for r in trajectory.rounds)


def test_no_retained_events_preserve_state_and_skip_projection(monkeypatch):
    patch_samples(monkeypatch, [[event("a")], [], [], []])
    runtime = Runtime([match_group("Uncertain action", (1, "a"))])
    base = WorldSnapshot(event_title="Test", state_summary="Base", world_state={"baseline": True})
    updated, result = asyncio.run(consensus_module.forecast_consensus_step(base, [], runtime, "test", 4, 50))
    assert updated.world_state == base.world_state
    assert updated.state_summary == "Base"
    assert updated.timeline == [] and result.outcomes == []
    assert updated.round_id == 1
    assert [c["schema"] for c in runtime.calls] == [EventGrouping]


def test_success_commits_one_canonical_snapshot_and_reports_settings(monkeypatch):
    patch_samples(monkeypatch, [[event("a")], [event("b")], [], []])
    runtime = Runtime([match_group("Accepted action", (1, "a"), (2, "b"))])
    commits = []

    class Repository:
        def get(self, event_id):
            return None

        def commit(self, event_id, snapshot, *, expected_version):
            commits.append((event_id, snapshot, expected_version))

    monkeypatch.setattr(engine, "AgentSDKClient", lambda *_: runtime)
    monkeypatch.setattr(engine, "SnapshotRepository", Repository)
    response = asyncio.run(engine.run_linear_forecast(
        SimulationRequest(event_title="Test", event_id="test-consensus", steps=2), 2,
    ))
    assert len(commits) == 1 and commits[0][2] == 0
    assert commits[0][1]["snapshot_version"] == 2
    assert response.branches == [] and response.mode == "multi_step"
    assert response.prediction_repeats == 4 and response.event_threshold_percent == 50
    assert len(response.timeline) == 2


def test_failed_sample_does_not_commit_a_partial_path(monkeypatch):
    commits = []
    attempts = []

    class Repository:
        def get(self, event_id):
            return None

        def commit(self, *args, **kwargs):
            commits.append(args)

    async def fail(self, snapshot, interventions, call_prefix):
        attempts.append(call_prefix)
        raise RuntimeError("sample failed")

    monkeypatch.setattr(engine, "SnapshotRepository", Repository)
    monkeypatch.setattr(engine, "AgentSDKClient", lambda *_: Runtime([]))
    monkeypatch.setattr(consensus_module.ForecastRoundOrchestrator, "run", fail)
    with pytest.raises(RuntimeError, match="sample failed"):
        asyncio.run(engine.run_linear_forecast(SimulationRequest(event_title="Test", event_id="failure"), 1))
    assert commits == []
    # Two layers of retry sit in front of this failure: each of the four
    # samples is re-run SAMPLE_ATTEMPTS times, and the round itself is re-run
    # ROUND_ATTEMPTS times. Only when both are exhausted is the failure real.
    assert len(attempts) == 4 * consensus_module.SAMPLE_ATTEMPTS * engine.ROUND_ATTEMPTS


@pytest.mark.parametrize("settings", [
    {"prediction_repeats": 0}, {"prediction_repeats": 17}, {"prediction_repeats": 1.5},
    {"event_threshold_percent": 0}, {"event_threshold_percent": 101},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValidationError):
        SimulationRequest(event_title="Test", **settings)


def test_branch_endpoint_is_retired():
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post("/api/llm/multi-branch-simulate", json={"event_title": "Test"})
    assert response.status_code == 410


def test_parallel_sdk_error_keeps_original_message_and_status():
    from app.api.forecasts import _forecast_failure
    from app.llm.client import LLMError

    error = _forecast_failure(ExceptionGroup("samples failed", [LLMError("SDK timeout")]))
    assert error.status_code == 502
    assert error.detail == "SDK timeout"


def test_api_forwards_consensus_settings_for_both_single_path_endpoints(monkeypatch):
    from app.api import forecasts
    from app.domain import SimulationResponse

    seen = []

    async def run(request, steps):
        seen.append((request.prediction_repeats, request.event_threshold_percent, steps))
        return SimulationResponse(mode="multi_step", analysis="done")

    monkeypatch.setattr(forecasts, "run_linear_forecast", run)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    for endpoint in ("simulate", "multi-step-simulate"):
        response = client.post(f"/api/llm/{endpoint}", json={
            "event_title": "Test", "prediction_repeats": 7, "event_threshold_percent": 60, "steps": 3,
        })
        assert response.status_code == 200
    assert seen == [(7, 60, 1), (7, 60, 3)]


def _round_with(*impacts):
    from app.domain import PowerImpact
    return ForecastRound(
        step=1, input_snapshot_version=0, output_snapshot_version=1,
        state_summary="s",
        power_impacts=[PowerImpact(**item) for item in impacts],
    )


def test_power_impacts_need_a_majority_of_samples_not_just_one():
    """One sample's strength verdict must not become the canonical state."""
    from app.simulation.consensus import _consensus_power_impacts

    agreed = _consensus_power_impacts([
        _round_with({"agent": "Iran", "dimension": "economy", "status": "weakened"}),
        _round_with({"agent": "Iran", "dimension": "economy", "status": "weakened"}),
        _round_with({"agent": "Iran", "dimension": "economy", "status": "strengthened"}),
        _round_with(),
    ], 2)
    assert [(i.agent, i.dimension, i.status) for i in agreed] == [("Iran", "economy", "weakened")]


def test_power_impacts_are_dropped_when_the_samples_disagree():
    from app.simulation.consensus import _consensus_power_impacts

    agreed = _consensus_power_impacts([
        _round_with({"agent": "Iran", "dimension": "economy", "status": "weakened"}),
        _round_with({"agent": "Iran", "dimension": "economy", "status": "weakened"}),
        _round_with({"agent": "Iran", "dimension": "economy", "status": "strengthened"}),
        _round_with({"agent": "Iran", "dimension": "economy", "status": "strengthened"}),
    ], 3)
    assert agreed == []


def test_maintained_is_not_a_change_and_one_sample_votes_once():
    from app.simulation.consensus import _consensus_power_impacts

    maintained = [
        _round_with({"agent": "Iran", "dimension": "economy", "status": "maintained"})
        for _ in range(4)
    ]
    assert _consensus_power_impacts(maintained, 2) == []
    # The same sample stating one dimension twice still casts one vote.
    assert _consensus_power_impacts([_round_with(
        {"agent": "Iran", "dimension": "economy", "status": "weakened"},
        {"agent": "Iran", "dimension": "economy", "status": "weakened"},
    )], 2) == []
