from __future__ import annotations

import asyncio
import json

import pytest

from app.domain import AgentProfile, Beliefs, DecisionPolicy, ForecastRound, TimelineEvent
from app.domain.ensemble import EnsembleRequest, OutcomeDefinition, EventQuestion, ClassificationBatch
from app.simulation.ensemble import prepare_experiment, run_experiment, wilson
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository


def actor():
    return AgentProfile(agent_id="actor_a", name="Laboratory", agent_type="organization",
                        decision_subject="Laboratory", role="neutral", description="A nonpolitical actor",
                        beliefs=Beliefs(), decision_policy=DecisionPolicy(risk_tolerance=.5, compromise_tendency=.5, response_speed=.5))


def request(**changes):
    return EnsembleRequest(event_title="Supply experiment", model="fixture-model", planning_mode="rules", agents=[actor().model_dump()],
        sample_count=4, steps=2,
        outcome_definitions=[OutcomeDefinition(id="left", label="Left", criterion="Left final state"),
                             OutcomeDefinition(id="right", label="Right", criterion="Right final state")],
        event_questions=[EventQuestion(id="move", label="Move", criterion="Any move event")], **changes)


class Classifier:
    recorder = None
    async def json_call(self, **kwargs):
        assert kwargs["schema"] is ClassificationBatch
        data = json.loads(kwargs["prompt"])
        return ClassificationBatch(samples=[{
            "sample_id": s["sample_id"], "category_id": s["final_state"], "matched_category_ids": [s["final_state"]], "reason": "Frozen criterion",
            "event_refs": [], "questions": [{"question_id": "move", "answer": "yes",
                "event_refs": [e["timeline_id"] for e in s["events"]], "reason": "Recorded"}]
        } for s in data["samples"]])


def runner(calls, fail_once=False):
    failed = set()
    async def run(snapshot, interventions, prefix):
        number = int(prefix.split("_sample_")[1].split("_")[0])
        calls.append((number, snapshot.round_id))
        assert snapshot.world_state.get("owner", number) == number
        if fail_once and number == 1 and snapshot.round_id == 1 and number not in failed:
            failed.add(number)
            raise RuntimeError("temporary provider failure")
        before = snapshot.snapshot_version
        snapshot.round_id += 1
        snapshot.snapshot_version += 1
        snapshot.world_state["owner"] = number
        snapshot.state_summary = "left" if number % 2 else "right"
        snapshot.agents[0].constraints.append(f"private_{number}_{snapshot.round_id}")
        assert all(c.startswith(f"private_{number}_") for c in snapshot.agents[0].constraints)
        event = TimelineEvent(timeline_id="provider-id", timestamp="next", description="A move",
                              actors=["actor_a"], confidence=.5, importance=.5,
                              origin="simulated", round_id=snapshot.round_id)
        snapshot.timeline.append(event)
        return snapshot, ForecastRound(step=snapshot.round_id, input_snapshot_version=before,
                                      output_snapshot_version=snapshot.snapshot_version,
                                      state_summary=snapshot.state_summary, timeline=[event.model_copy(deep=True)])
    return run


def test_full_paths_are_isolated_counted_once_and_never_commit_latest_snapshot(tmp_path):
    store = EnsembleStore(tmp_path / "runs")
    events = EventRepository(tmp_path / "events")
    req = request()
    run_id, req = prepare_experiment(req, events=events, store=store)
    before = store.load(run_id, "initial_snapshot")
    calls = []
    result = asyncio.run(run_experiment(run_id, req, store=store, runtime=Classifier(), round_runner=runner(calls)))
    assert len(calls) == 8
    assert result["completed_samples"] == 4
    assert [c["count"] for c in result["categories"]] == [2, 2, 0]
    assert sum(c["frequency"] for c in result["categories"]) == 1
    assert result["event_frequencies"][0]["count"] == 4  # eight occurrences, four trajectories
    assert result["event_frequencies"][0]["frequency"] == 1
    assert before == store.load(run_id, "initial_snapshot")
    assert not list(tmp_path.rglob("latest_snapshot.json"))
    assert "api_key" not in store.load(run_id, "config")["request"]
    ids = [e["timeline_id"] for r in store.sample(run_id, 1)["rounds"] for e in r["timeline"]]
    assert len(ids) == len(set(ids)) == 2


def test_resume_only_runs_missing_rounds_and_preserves_denominator(tmp_path):
    store, events = EnsembleStore(tmp_path/"runs"), EventRepository(tmp_path/"events")
    run_id, req = prepare_experiment(request(), events=events, store=store)
    calls = []
    run = runner(calls, fail_once=True)
    partial = asyncio.run(run_experiment(run_id, req, store=store, runtime=Classifier(), round_runner=run))
    assert partial["completed_samples"] == 3 and partial["failed_samples"] == 1
    assert partial["uncompleted_samples"] == 1
    assert store.sample(run_id, 1)["rounds"][0]["step"] == 1
    count = len(calls)
    _, resumed = prepare_experiment(request(resume_run_id=run_id), events=events, store=store)
    result = asyncio.run(run_experiment(run_id, resumed, store=store, runtime=Classifier(), round_runner=run))
    assert calls[count:] == [(1, 1)]
    assert result["completed_samples"] == 4 and result["failed_samples"] == 0
    assert store.load(run_id, "status")["status"] == "completed"


def test_retrospective_profile_leak_is_rejected(tmp_path):
    store, events = EnsembleStore(tmp_path/"runs"), EventRepository(tmp_path/"events")
    req = request(as_of_month="2020-01")
    req.recent_timeline = [{"timeline_id": "h", "timestamp": "2022-01", "description": "Later history",
                            "confidence": .5, "importance": .5}]
    with pytest.raises(ValueError, match="Retrospective"):
        prepare_experiment(req, events=events, store=store)


def test_invalid_classifier_coverage_does_not_fabricate_a_distribution(tmp_path):
    class Missing(Classifier):
        async def json_call(self, **kwargs):
            return ClassificationBatch(samples=[])
    store, events = EnsembleStore(tmp_path/"runs"), EventRepository(tmp_path/"events")
    run_id, req = prepare_experiment(request(), events=events, store=store)
    result = asyncio.run(run_experiment(run_id, req, store=store, runtime=Missing(), round_runner=runner([])))
    assert result["completed_samples"] == 0
    assert all(c["frequency"] is None for c in result["categories"])
    assert all(s["status"] == "sampled" for s in result["samples"])


def test_wilson_bounds_and_empty_samples():
    assert wilson(0, 0) is None
    low, high = wilson(50, 100)
    assert .40 < low < .41 and .59 < high < .60
    assert wilson(0, 100)[1] > 0


def test_new_baseline_uses_compact_dossier_instead_of_uncalibrated_strength(tmp_path):
    from app.domain import Capability
    principal = actor()
    principal.capabilities = [Capability(name="bargaining_leverage", level=.6),
                              Capability(name="resource_access", level=.4)]
    req = request()
    req.agents = [principal.model_dump()]
    store = EnsembleStore(tmp_path / "runs")
    run_id, _ = prepare_experiment(req, events=EventRepository(tmp_path), store=store)
    baseline = store.load(run_id, "initial_snapshot")
    assert baseline["strength_dimensions"] == []
    assert baseline["agents"][0]["decision_core"]["version"] == "decision-core-1"
    assert len(req.agents[0]["capabilities"]) == 2  # Original input is retained, not rewritten.


@pytest.mark.parametrize('takes_action', [False, True])
def test_real_round_orchestrator_keeps_no_action_paths_and_private_initial_information(tmp_path, takes_action):
    """Use the actual BDI/activation/settlement/transition chain with a scripted model."""
    import re
    from app.domain.ensemble import OutcomeCondition, StateInvariant
    calls = []
    class ScenarioModel:
        recorder = None
        async def json_call(self, **kw):
            calls.append(kw["schema"].__name__)
            kind, prompt = kw["schema"].__name__, kw["prompt"]
            step = int(re.search(r"_round_(\d+)", kw["call_name"])[1])
            ids = {"agent_id": "actor_a", "round_id": step, "snapshot_version": step-1}
            if kind == "InformationReview":
                assert "PRIVATE_INITIAL" in prompt
                return kw["schema"](**ids, processed_observation_ids=["initial_actor_a_0"] if step == 1 else [], assessment="Keep beliefs")
            if kind == "MasterActivationBatch":
                return kw["schema"](round_id=step, snapshot_version=step-1, reviews=[{
                    "agent_id": "actor_a", "master_score": .1, "decision_point_now": False,
                    "inaction_consistency": "consistent", "feasible_action_exists": False,
                    "decision_window": "closed", "cross_agent_dependency": False, "reasons": ["wait"]}])
            if kind == "SelfActivationReview":
                return kw["schema"](**ids, self_score=.1, decision_required=False,
                                    feasible_action_exists=False, preferred_mode="wait", reasons=["No trigger"])
            if kind == 'ActionProposal':
                return kw['schema'](**ids, agent_name='Laboratory', action_id=f'action_{step}_actor_a',
                                    decision='act' if takes_action else 'wait',
                                    action_type='statement' if takes_action else 'no_action',
                                    action='Propose public coordination' if takes_action else 'Wait for new information',
                                    reasoning='No due action; waiting is an explicit decision', confidence=.8)
            if kind == 'AuditBatch':
                return kw['schema'](round_id=step, snapshot_version=step-1,
                                    reviews=[{'agent_id':'actor_a','accepted':True}])
            if kind == "GameMasterSettlement":
                assert "PRIVATE_INITIAL" not in prompt
                data = json.loads(prompt.split("FROZEN PUBLIC SNAPSHOT\n", 1)[1].split("\n\nUSER INTERVENTIONS", 1)[0])
                state = dict(data['world_state'])
                events=[]
                if takes_action and step==1:
                    assert '"decision": "act"' in prompt
                    state['offer_made']=True
                    events=[{'timeline_id':f'sim_{step}_offer','timestamp':'2026-01',
                             'description':'The team proposed public coordination', 'actors':['actor_a'],
                             'confidence':1,'importance':.5,'origin':'simulated','round_id':step}]
                return kw["schema"](state_summary="Offer made" if state['offer_made'] else "No change", next_world_state=state,
                    public_events=events,
                    agent_knowledge_updates=[{"agent_id": "actor_a", "information": []}], analysis="No event is valid")
            raise AssertionError(kind)
    req = request()
    req.event_questions = []
    req.initial_private_information = {"actor_a": ["PRIVATE_INITIAL"]}
    req.initial_world_state = {"stock": 10, "offer_made": False}
    req.state_invariants = [StateInvariant(paths=["stock"], rule="fixed")]
    req.outcome_definitions = [OutcomeDefinition(id="stable", label="Stable", criterion="No offer", conditions=[OutcomeCondition(path="offer_made", operator="eq", value=False)]),
                              OutcomeDefinition(id="changed", label="Changed", criterion="Offer made", conditions=[OutcomeCondition(path="offer_made", operator="eq", value=True)])]
    store, events = EnsembleStore(tmp_path/"runs"), EventRepository(tmp_path/"events")
    run_id, req = prepare_experiment(req, events=events, store=store)
    result = asyncio.run(run_experiment(run_id, req, store=store, runtime=ScenarioModel()))
    assert result["completed_samples"] == 4
    assert result["categories"][int(takes_action)]["count"] == 4
    assert calls.count("GameMasterSettlement") == 8
    assert calls.count('ActionProposal') == 4
    assert result['reliability']['explicit_waits'] == (0 if takes_action else 4)
    assert result['reliability']['external_actions'] == (4 if takes_action else 0)
    assert result['reliability']['deliberation_rounds'] == 4
    assert "InformationReview" in calls and "RelationshipProjection" not in calls
    assert sum(len(r['timeline']) for r in store.sample(run_id,1)['rounds']) == int(takes_action)


def test_unlimited_experiment_completes_and_reports_usage(tmp_path):
    store = EnsembleStore(tmp_path / 'runs')
    req = request(max_model_calls=None, max_total_tokens=None, max_runtime_seconds=None)
    run_id, req = prepare_experiment(req, events=EventRepository(tmp_path / 'events'), store=store)
    result = asyncio.run(run_experiment(run_id, req, store=store, runtime=Classifier(), round_runner=runner([])))
    assert result['completed_samples'] == 4
    assert result['budget']['limits']['max_calls'] is None
    assert result['budget']['limits']['max_total_tokens'] is None
