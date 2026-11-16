import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.domain import ActionProposal, ForecastRound, WorldSnapshot
from app.domain.ensemble import EnsembleRequest, OutcomeCondition, OutcomeDefinition
from app.domain.protocol import ScenarioProtocol
from app.knowledge.evidence import VERSION, validate_event
from app.knowledge.okf import KnowledgeStore, digest
from app.llm.budget import ModelBudget
from app.simulation.ensemble import prepare_experiment, run_experiment
from app.simulation.planning import RuntimePlan
from app.simulation.protocol import program_projection
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository
from test_ensemble import actor, request
from test_okf import corpus


def test_bare_resume_inherits_persisted_limits_and_prices(tmp_path):
    store, events = EnsembleStore(tmp_path / "runs"), EventRepository(
        tmp_path / "events"
    )
    req = request(
        max_model_calls=5,
        max_total_tokens=10000,
        max_cost_usd=1,
        input_price_per_million=2,
        output_price_per_million=3,
        cached_input_price_per_million=0.2,
    )
    run_id, req = prepare_experiment(req, events=events, store=store)
    ledger = ModelBudget(req)
    ticket = ledger.reserve(30, 20)
    ledger.settle(ticket, {"prompt_tokens": 10, "completion_tokens": 4}, 200)
    saved = ledger.report()
    saved["limits"]["max_calls"] = 7
    store.save(run_id, "budget", saved)
    _, resumed = prepare_experiment(
        EnsembleRequest(event_title="Resume", resume_run_id=run_id),
        events=events,
        store=store,
    )
    assert resumed.max_model_calls == 7 and resumed.max_total_tokens == 10000
    assert resumed.max_cost_usd == 1 and resumed.cached_input_price_per_million == 0.2
    assert resumed.model == req.model and resumed.base_url == req.base_url
    assert ModelBudget(resumed, saved).calls == 1
    with pytest.raises(ValueError, match="original unit prices"):
        prepare_experiment(
            EnsembleRequest(
                event_title="Resume", resume_run_id=run_id, input_price_per_million=9
            ),
            events=events,
            store=store,
        )


@pytest.mark.parametrize(("unit", "size", "first_end", "last_end"), [
    ("auto", 1, "2026-11-07", "2026-11-21"),
    ("week", 2, "2026-11-14", "2026-12-12"),
])
def test_runtime_plan_is_frozen_and_explicit_quiet_policy_reduces_calls(
    tmp_path, monkeypatch, unit, size, first_end, last_end,
):
    store, events = EnsembleStore(tmp_path / "runs"), EventRepository(
        tmp_path / "events"
    )
    req = request()
    req.planning_mode = "auto"
    req.step_unit = unit
    req.step_size = size
    req.start_date = "2026-10-31"
    req.steps = 3
    req.event_questions = []
    req.scenario_protocol = ScenarioProtocol(
        idle_policy="advance_if_unchanged"
    ).model_dump()
    req.initial_world_state = {"stock": 10}
    req.outcome_definitions = [
        OutcomeDefinition(
            id="same",
            label="Same",
            criterion="same",
            conditions=[OutcomeCondition(path="stock", operator="eq", value=10)],
        ),
        OutcomeDefinition(
            id="other_stock",
            label="Changed",
            criterion="changed",
            conditions=[OutcomeCondition(path="stock", operator="ne", value=10)],
        ),
    ]

    class Planner:
        calls = 0

        async def json_call(self, **kw):
            self.calls += 1
            assert kw["schema"] is RuntimePlan
            return RuntimePlan(step_unit="week" if unit == "auto" else "month", reason="Fixture plan")

    calls = []

    async def no_change(snapshot, interventions, prefix):
        calls.append(prefix)
        before = snapshot.snapshot_version
        snapshot.round_id += 1
        snapshot.snapshot_version += 1
        # This stub stands in for the initial formal wait deliberation.
        for a in snapshot.agents:
            a.activation.last_deliberation_round = snapshot.round_id
        return snapshot, ForecastRound(
            step=snapshot.round_id,
            input_snapshot_version=before,
            output_snapshot_version=snapshot.snapshot_version,
            state_summary=snapshot.state_summary,
        )

    run_id, req = prepare_experiment(req, events=events, store=store)
    from app.simulation import ensemble

    class Orchestrator:
        def __init__(self, *args, **kwargs):
            pass

        run = staticmethod(no_change)

    monkeypatch.setattr(ensemble, "ForecastRoundOrchestrator", Orchestrator)
    model = Planner()
    result = asyncio.run(run_experiment(run_id, req, store=store, runtime=model))
    assert result["completed_samples"] == 4 and result["step_unit"] == "week"
    assert len(calls) == 4 and model.calls == 1
    rounds = store.sample(run_id, 1)["rounds"]
    assert len(rounds) == 3
    assert result["step_size"] == size
    assert rounds[0]["simulation_interval"]["interval_end"] == first_end
    assert rounds[1]["simulation_interval"]["interval_start"] == first_end
    assert rounds[-1]["simulation_interval"]["interval_end"] == last_end
    _, resumed = prepare_experiment(
        EnsembleRequest(event_title="Resume", resume_run_id=run_id),
        events=events,
        store=store,
    )
    assert resumed.step_size == size
    asyncio.run(run_experiment(run_id, resumed, store=store, runtime=model))
    assert model.calls == 1 and len(calls) == 4


def test_shared_resource_conflicts_fail_before_settlement():
    protocol = ScenarioProtocol(
        action_mechanisms=[
            {
                "action_type": "ship",
                "preconditions": [{"path": "stock", "operator": "ge", "value": 1}],
                "effects": [{"path": "stock", "delta_per_step": -1}],
            }
        ]
    )
    state = {
        "stock": 1,
        "scenario_protocol": protocol.model_dump(),
        "state_invariants": [{"paths": ["stock"], "rule": "range", "minimum": 0}],
    }
    snapshot = WorldSnapshot(
        event_title="Inventory", agents=[actor()], world_state=state
    )
    proposal = ActionProposal(
        agent_id="actor_a",
        agent_name="Lab",
        round_id=1,
        snapshot_version=0,
        action_id="ship1",
        decision="act",
        action_type="ship",
        action="Ship",
        reasoning="Order",
        confidence=0.8,
    )
    with pytest.raises(ValueError, match="range"):
        program_projection(
            snapshot, [proposal, proposal.model_copy(update={"action_id": "ship2"})]
        )
    with pytest.raises(ValidationError):
        EnsembleRequest(
            event_title="Rates",
            step_unit="auto",
            scenario_protocol={"rates": [{"path": "stock", "delta_per_step": 1}]},
        )
    with pytest.raises(ValidationError):
        ScenarioProtocol(
            action_mechanisms=[
                {
                    "action_type": "ship",
                    "preconditions": [
                        {"path": "stock", "operator": "ge", "value": 1, "typo": True}
                    ],
                }
            ]
        )


def test_hierarchy_search_and_deep_evidence_focus(tmp_path):
    events, case_id = corpus(tmp_path)
    agents = events.load_artifact(case_id, "agents", [])
    agents[0]["public_state"]["subordinates"] = [
        {"name": "Lab"},
        {"name": "Researcher", "title": "Director"},
    ]
    events.save_artifact(case_id, "agents", agents)
    historical = events.load_artifact(case_id, "timeline", [])[0]
    historical["retrieval_actors"]=[{"name":"Other Union"}]
    events.save_artifact(
        case_id,
        "timeline_full",
        [historical | {"timeline_id": f"hist_{i:03}"} for i in range(120)],
    )
    store = KnowledgeStore(events)
    store.publish(case_id)
    graph = store.graph(case_id, "Researcher", view="actors", limit=10)
    assert "actor_a" in {n["id"] for n in graph["nodes"]}
    assert "actor_a" in graph["expanded_parent_ids"]
    other=store.graph(case_id,"Other Union",view="actors",limit=10)
    assert "group:"+case_id+":mentioned" in other["expanded_parent_ids"]
    assert len([n for n in graph["nodes"] if n.get("decision_role") == "member"]) == 1
    assert len(store.graph(case_id, view="actors", limit=1)["nodes"]) <= 2
    focus = store.graph(case_id, view="evidence", focus_id="hist_119", limit=10)
    assert "hist_119" in {n["id"] for n in focus["nodes"]}
    dated = store.graph(case_id, view="actors", from_month="2025-01")
    assert not any(n["type"] == "actor" for n in dated["nodes"])


def test_profile_changes_require_reanalysis_and_api_guards_are_reciprocal(
    tmp_path, monkeypatch
):
    from app.api import experiments, knowledge

    events, case_id = corpus(tmp_path)
    history = events.load_artifact(case_id, "timeline", [])[:1]
    events.save_artifact(case_id, "timeline_full", history)
    sources = events.load_sources(case_id)
    agents = events.load_artifact(case_id, "agents", [])
    events.save_artifact(
        case_id,
        "extraction_manifest",
        {
            "evidence_policy": VERSION,
            "history_fingerprint": digest(history),
            "profile_fingerprint": digest(agents),
            "profile_source_fingerprint": digest(
                [
                    {
                        "url": s.get("url"),
                        "content": s.get("content"),
                        "sections": s.get("sections", []),
                    }
                    for s in sources
                ]
            ),
        },
    )
    store = KnowledgeStore(events)
    assert not store.publish(case_id)["requires_reanalysis"]
    agents[0]["description"] = "Edited"
    events.save_artifact(case_id, "agents", agents)
    assert store.publish(case_id)["requires_reanalysis"]
    with pytest.raises(ValueError, match="Reanalyze the sources"):
        prepare_experiment(
            request(event_id=case_id),
            events=events,
            store=EnsembleStore(tmp_path / "runs"),
        )
    app = FastAPI()
    app.include_router(experiments.router)
    app.include_router(knowledge.router)
    client = TestClient(app)
    monkeypatch.setattr(experiments, "EventRepository", lambda: events)
    monkeypatch.setattr(knowledge, "events", events)
    experiments._preparing_cases.add(case_id)
    try:
        assert client.get(f"/api/knowledge/{case_id}/graph").status_code == 409
        assert (
            client.post(
                "/api/llm/figures-pipeline",
                json={"event_id": case_id, "event_title": "Case"},
            ).status_code
            == 409
        )
    finally:
        experiments._preparing_cases.discard(case_id)
    knowledge._inflight_extractions.add(case_id)
    try:
        assert (
            client.post(
                "/api/experiments", json={"event_id": case_id, "event_title": "Case"}
            ).status_code
            == 409
        )
    finally:
        knowledge._inflight_extractions.discard(case_id)
    quality = client.get(f"/api/knowledge/{case_id}/quality")
    assert quality.status_code == 200 and quality.json()["historical_events"] == 1


def test_source_future_language_overrides_missing_model_modality():
    quote = "In March 2024 the company will begin production."
    result, reason = validate_event(
        {"timestamp": "2024-03", "evidence_text": quote}, {"content": quote}
    )
    assert result is None and reason == "planned_event_is_not_observed_occurrence"
