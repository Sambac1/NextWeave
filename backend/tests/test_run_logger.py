from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from app.domain import (
    AgentProfile,
    Beliefs,
    DecisionPolicy,
    GameMasterSettlement,
    RelationshipProjection,
    WorldSnapshot,
)
from app.application import forecasting
from app.simulation import engine
from app.storage import run_logger


def test_readable_filename_metadata_duration_and_credential_redaction(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_logger, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    logger = run_logger.RunLogger(
        "forecast_runs",
        run_id="request-123",
        operation="single-step-forecast",
        event_id="event-456",
    )
    path = logger.record(
        source="ordinary-agent",
        round_id=2,
        agent_id="agent_abc",
        content_type="agent-self-prediction-prompt",
        status="started",
        duration_ms=12.3456,
        data={"prompt": "choose", "api_key": "secret", "nested": {"access_token": "token"}},
    )

    assert "source-ordinary-agent" in path.name
    assert "round-02" in path.name
    assert "agent-agent_abc" in path.name
    assert "content-agent-self-prediction-prompt" in path.name
    assert "status-started" in path.name
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["duration_ms"] == 12.346
    assert payload["metadata"]["event_id"] == "event-456"
    assert payload["data"]["api_key"] == "***REDACTED***"
    assert payload["data"]["nested"]["access_token"] == "***REDACTED***"


def test_trace_recorder_derives_round_agent_source_and_content_from_call_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_logger, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    recorder = run_logger.TraceRecorder("forecast_runs", run_id="run-abc")
    prompt_path = recorder.write(
        "linear_round_03_agent_agent_abc_decision",
        "agent_sdk_request",
        {"prompt": "predict"},
    )
    result_path = recorder.write(
        "linear_round_03_attempt_1_master_action_audit",
        "agent_sdk_validated_response",
        {"accepted": True},
        duration_ms=21,
    )

    assert "source-ordinary-agent" in prompt_path.name
    assert "round-03" in prompt_path.name
    assert "agent-agent_abc" in prompt_path.name
    assert "content-agent-self-prediction-prompt" in prompt_path.name
    assert "source-master-agent" in result_path.name
    assert "content-master-action-audit-result" in result_path.name
    assert "status-success" in result_path.name


def test_request_run_id_is_inherited_by_domain_recorders(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_logger, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    token = run_logger.set_request_run_id("shared-request-id")
    try:
        recorder = run_logger.TraceRecorder("knowledge_runs")
    finally:
        run_logger.reset_request_run_id(token)
    assert recorder.run_id == "shared-request-id"


def test_forecast_round_records_timeline_agents_graph_and_stage_results(monkeypatch) -> None:
    class Recorder:
        def __init__(self) -> None:
            self.records: list[dict] = []

        def record(self, **kwargs):
            self.records.append(kwargs)

    class Client:
        def __init__(self) -> None:
            self.recorder = Recorder()

    agent = AgentProfile(
        agent_id="agent_a",
        name="Alice",
        agent_type="person",
        decision_subject="Alice",
        role="initiator",
        description="Alice profile",
        beliefs=Beliefs(),
        decision_policy=DecisionPolicy(
            risk_tolerance=0.5,
            compromise_tendency=0.5,
            response_speed=0.5,
        ),
    )
    snapshot = WorldSnapshot(event_title="Event", agents=[agent])

    async def unchanged(self, value, *args, **kwargs):
        return value.model_copy(deep=True)

    async def no_active(self, value, *args, **kwargs):
        return value.model_copy(deep=True), [], []

    async def no_actions(self, *args, **kwargs):
        return []

    async def settlement(self, value, *args, **kwargs):
        return GameMasterSettlement(
            state_summary="After",
            next_world_state=value.world_state,
            agent_knowledge_updates=[{"agent_id": "agent_a", "information": []}],
            analysis="settled",
        )

    async def relationship_projection(self, *args, **kwargs):
        return RelationshipProjection()

    monkeypatch.setattr(forecasting.OrdinaryBDIAgent, "process_information", unchanged)
    monkeypatch.setattr(forecasting.ActivationMasterAgent, "select", no_active)
    monkeypatch.setattr(forecasting.ActionAuditMasterAgent, "produce", no_actions)
    monkeypatch.setattr(forecasting.GameMasterAgent, "settle", settlement)
    monkeypatch.setattr(forecasting.RelationshipAgent, "update", relationship_projection)
    client = Client()
    asyncio.run(engine.forecast_one_step(snapshot, [], client, "linear_round_01"))

    content_types = {item["content_type"] for item in client.recorder.records}
    assert {
        "round-input-snapshot",
        "timeline-before-round",
        "relationship-graph-before",
        "agent-profile-before-round",
        "game-master-settlement-result",
        "relationship-projection",
        "timeline-generated-this-round",
        "timeline-after-round",
        "relationship-graph-after",
        "agent-profile-after-round",
        "round-output-snapshot",
    } <= content_types
