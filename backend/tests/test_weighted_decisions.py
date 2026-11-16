"""Regression coverage for weighted decisions through audit and round recovery."""

import asyncio
import json
import re

import pytest

from app.domain import SelectedAction, TimelineEvent, WorldSnapshot
from app.domain.decision_core import DecisionCore, DecisionLimit
from app.domain.ensemble import EnsembleRequest, OutcomeCondition, OutcomeDefinition
from app.llm.client import LLMError
from app.simulation.ensemble import prepare_experiment, run_experiment
from app.simulation.weighted_decisions import CandidatePool, VERSION, _build_candidates, produce_weighted_actions
from app.simulation.action_audit import _program_violations
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository
from app.storage.round_stages import RoundStages
from tests.test_ensemble import actor


def option(text, weight, **fields):
    return {
        "decision": "act", "action_type": "statement", "action": text,
        "reasoning": "Scenario-specific rationale", "weight": weight,
    } | fields


def saved_draws(directory):
    payloads = [json.loads(path.read_text())["payload"] for path in directory.rglob("*.json")]
    return [value for value in payloads if isinstance(value, dict) and "audit" in value and "actions" in value]


def test_weighted_candidate_format_repair_preserves_target_boundaries():
    a = actor()
    israel = a.model_copy(update={"agent_id": "agent_israel", "name": "Israel", "aliases": ["IL"]})
    other = a.model_copy(update={"agent_id": "agent_other", "name": "Other", "aliases": ["IL"]})
    snapshot = WorldSnapshot(event_title="Negotiation", agents=[a, israel, other])
    pool = CandidatePool(candidates=[
        option("Publicly request a response", 40, target=" Israel "),
        option("Observe until the deadline", 30, target="Israeli forces", decision="monitor", action_type="ceasefire_monitoring"),
        option("Send a private note", 30, target="IL", visibility="direct"),
    ])
    first, second, third = _build_candidates(pool, snapshot, a, "initial")
    assert first.proposal.target == "agent_israel" and not _program_violations(first.proposal, a, snapshot)
    assert second.proposal.target == "" and "Israeli forces" in second.proposal.action
    assert second.proposal.action_type == "no_action" and not _program_violations(second.proposal, a, snapshot)
    assert "UNKNOWN_TARGET" in _program_violations(third.proposal, a, snapshot)

    contradictions = CandidatePool(candidates=[
        option("Send a message immediately", 50, target="Israel via Qatar", visibility="direct"),
        option("Announce a deal", 50, decision="wait", action_type="announce_deal"),
    ])
    direct, wait = _build_candidates(contradictions, snapshot, a, "initial")
    assert "UNKNOWN_TARGET" in _program_violations(direct.proposal, a, snapshot)
    assert "NO_ACTION_TYPE_REQUIRED" in _program_violations(wait.proposal, a, snapshot)


@pytest.mark.parametrize("text", [
    "Maintain the existing decision and revisit only if the other party offers new terms.",
    "Maintain the existing position; no new outward act before the horizon.",
    "Take no new outward step; the previous decision remains in place.",
    "Maintain the existing decision; issue no new statement or response this round.",
])
def test_keep_existing_position_is_repaired_without_changing_the_initial_distribution(tmp_path, text):
    a = actor()
    snapshot = WorldSnapshot(event_title="Agreement", agents=[a])
    calls = []
    hold = option(text, 80, decision="no_action", action_type="hold_previous_position")
    signal = option("Publicly propose negotiations on revised terms.", 20)

    class Model:
        async def json_call(self, *, schema, call_name, prompt, **kwargs):
            calls.append(schema.__name__)
            if schema is CandidatePool:
                assert call_name.endswith("_initial")
                return schema(candidates=[hold, signal])
            return schema(round_id=1, snapshot_version=0, reviews=[
                {"candidate_id": candidate["candidate_id"], "verdict": "allow"}
                for candidate in json.loads(prompt)["candidates"]
            ])

    _, stats = asyncio.run(produce_weighted_actions(
        snapshot, [a], [], Model(), "test", seed=2026, sample_id=2,
        cache=RoundStages(tmp_path, {"sample": 2}),
    ))
    draw, = saved_draws(tmp_path)
    audit, = draw["audit"]
    assert [row["probability"] for row in audit["final_pool"]] == [0.8, 0.2]
    assert audit["candidates"][0]["repairs"] == ["normalize_non_action_type"]
    assert stats["rejected_candidates"] == stats["supplemented_decisions"] == 0
    assert calls == ["CandidatePool", "CandidateAuditBatch"]


def test_explicitly_deferred_action_is_preserved_as_a_plan():
    actor_profile = actor()
    snap = WorldSnapshot(event_title="Policy review", agents=[actor_profile])
    draft = option("Monitor incoming data, then publish the decision at the meeting's conclusion.", 40,
                   decision="monitor", action_type="monitor_incoming_data", condition="Review at the meeting's conclusion.")
    candidate, = _build_candidates(CandidatePool(candidates=[draft]), snap, actor_profile, "initial")
    assert candidate.proposal.action.splitlines()[0] == "Monitor incoming data"
    assert "publish the decision" not in candidate.proposal.action
    assert candidate.proposal.action_type == "no_action"
    assert "publish the decision" in candidate.proposal.plan_update.description


def test_benchmark_defers_only_when_no_active_response_is_required():
    a = actor()

    class EmptyRuntime:
        async def json_call(self, *, schema, **_):
            assert schema is CandidatePool
            return CandidatePool(candidates=[])

    async def run(world_state):
        snap = WorldSnapshot(event_title="Policy review", agents=[a], world_state=world_state)
        return await produce_weighted_actions(snap, [a], [], EmptyRuntime(), "test", seed=3, sample_id=1,
                                              cache=RoundStages(None, {}), compact_benchmark=True, record_draws=True)

    actions, stats = asyncio.run(run({}))
    assert actions[0].decision == "no_action" and actions[0].visibility == "private"
    assert stats["sampled_choices"][0]["p_selected"] == 1
    with pytest.raises(ValueError, match="no eligible action"):
        asyncio.run(run({"decision_queue": [a.agent_id]}))


def test_maintaining_a_position_cannot_hide_a_new_outward_action():
    a = actor()
    snapshot = WorldSnapshot(event_title="Agreement", agents=[a])
    pool = CandidatePool(candidates=[
        option("Maintain the current position and issue a new public statement; do not sign the agreement.",
               60, decision="no_action", action_type="maintain_position"),
        option("维持现有决定，暂不签署，收到新提案后再复核。",
               40, decision="wait", action_type="保持现状"),
    ])
    outward, waiting = _build_candidates(pool, snapshot, a, "initial")
    assert "NO_ACTION_TYPE_REQUIRED" in _program_violations(outward.proposal, a, snapshot)
    assert waiting.proposal.action_type == "no_action" and not _program_violations(waiting.proposal, a, snapshot)


def test_intentional_single_candidate_does_not_trigger_supplement_or_rescoring(tmp_path):
    a = actor()
    snapshot = WorldSnapshot(event_title="Supply", agents=[a])
    calls = []

    class Model:
        async def json_call(self, *, schema, call_name, prompt, **kwargs):
            calls.append(schema.__name__)
            if schema is CandidatePool:
                assert call_name.endswith("_initial")
                return schema(candidates=[option("Acknowledge the published decision", 25)])
            candidate, = json.loads(prompt)["candidates"]
            return schema(round_id=1, snapshot_version=0,
                          reviews=[{"candidate_id": candidate["candidate_id"], "verdict": "allow"}])

    actions, stats = asyncio.run(produce_weighted_actions(
        snapshot, [a], [], Model(), "test", seed=11, sample_id=1,
        cache=RoundStages(tmp_path, {"sample": 1}),
    ))
    assert [action.action for action in actions] == ["Acknowledge the published decision"]
    assert calls == ["CandidatePool", "CandidateAuditBatch"]
    assert stats["supplemented_decisions"] == 0
    draw, = saved_draws(tmp_path)
    assert draw["audit"][0]["final_pool"][0]["probability"] == 1


def test_candidate_audits_remove_grounded_veto_but_keep_vague_veto_and_rescore(tmp_path):
    a = actor()
    a.decision_core = DecisionCore(limits=[DecisionLimit(
        id="quota", description="The laboratory may dispatch at most one unit per round.",
        provenance="source_statement", claim_scope="structural", timeline_refs=["h1"],
    )])
    history = TimelineEvent(timeline_id="h1", timestamp="2025-01", description="The laboratory charter limits dispatch.",
                            confidence=1, importance=1)
    snapshot = WorldSnapshot(event_title="Supply", agents=[a], timeline=[history])
    audit_batches = []
    scores = []

    class Model:
        async def json_call(self, *, schema, call_name, prompt, **kwargs):
            payload = json.loads(prompt)
            if schema.__name__ == "CandidatePool":
                if call_name.endswith("_initial"):
                    return schema(candidates=[
                        option("Ask the unknown office", 70, target="missing_actor"),
                        option("Dispatch ten units immediately", 25),
                        option("Offer to discuss emergency supplies", 5),
                    ])
                assert [c["candidate_id"] for c in payload["retained_options"]] == ["actor_a:initial:3"]
                assert {item["code"] for item in payload["feedback"]} == {"program_rejected", "model_rejected"}
                return schema(candidates=[option("Offer one unit", 50, condition="After confirmation")])
            if schema.__name__ == "CandidateAuditBatch":
                ids = [c["candidate_id"] for c in payload["candidates"]]
                audit_batches.append(ids)
                if ids == ["actor_a:initial:2", "actor_a:initial:3"]:
                    reviews = [
                        {"candidate_id": ids[0], "verdict": "hard_violation", "evidence_ref": "limits:actor_a:quota",
                         "evidence_quote": "may dispatch at most one unit per round", "reason": "Ten exceeds the limit"},
                        {"candidate_id": ids[1], "verdict": "hard_violation", "evidence_ref": "unknown",
                         "evidence_quote": "Not permitted", "reason": "Uncertain authority"},
                    ]
                else:
                    assert ids == ["actor_a:supplement:1"]
                    reviews = [{"candidate_id": ids[0], "verdict": "allow"}]
                return schema(round_id=1, snapshot_version=0, reviews=reviews)
            if schema.__name__ == "CandidateWeights":
                ids = [c["candidate_id"] for c in payload["options"]]
                scores.append(ids)
                assert ids == ["actor_a:initial:3", "actor_a:supplement:1"]
                # Reversing the scorer's order must not change ID association.
                return schema(weights=[{"candidate_id": ids[1], "weight": 100}, {"candidate_id": ids[0], "weight": 0}])
            raise AssertionError(schema.__name__)

    actions, stats = asyncio.run(produce_weighted_actions(
        snapshot, snapshot.agents, [], Model(), "test", seed=11, sample_id=1,
        cache=RoundStages(tmp_path, {"sample": 1}),
    ))
    assert len(actions) == 1 and actions[0].action.startswith("Offer one unit")
    assert "After confirmation" in actions[0].action
    assert len(audit_batches) == 2 and len(scores) == 1
    assert stats["rejected_candidates"] == 2 and stats["supplemented_decisions"] == 1
    draw, = saved_draws(tmp_path)
    record, = draw["audit"]
    assert [row["probability"] for row in record["final_pool"]] == [0, 1]
    assert record["selected_candidate_id"] == "actor_a:supplement:1"


def test_weighted_settlement_resume_reuses_draw_and_applies_only_selected_action(tmp_path):
    options = [
        option("OPTION_ONE", 40, plan_update={"description": "PLAN_ONE"}),
        option("OPTION_TWO", 35, plan_update={"description": "PLAN_TWO"}),
        option("OPTION_THREE", 25, plan_update={"description": "PLAN_THREE"}),
    ]
    plan_by_action = {candidate["action"]: candidate["plan_update"]["description"] for candidate in options}
    calls = []
    settlement_actions = {}

    class Model:
        recorder = None

        async def json_call(self, *, schema, call_name, prompt, **kwargs):
            name = schema.__name__
            sample = int(re.search(r"_sample_(\d+)", call_name)[1])
            calls.append((sample, name))
            if name == "CandidatePool":
                return schema(candidates=options)
            if name == "CandidateAuditBatch":
                payload = json.loads(prompt)
                return schema(round_id=1, snapshot_version=0, reviews=[
                    {"candidate_id": row["candidate_id"], "verdict": "allow"}
                    for row in payload["candidates"]
                ])
            if name == "GameMasterSettlement":
                frozen = json.loads(prompt.split("FROZEN PUBLIC SNAPSHOT\n", 1)[1].split("\n\nUSER INTERVENTIONS", 1)[0])
                actions = json.loads(prompt.split("AUDITED ACTIONS\n", 1)[1].split("\n\nSTRENGTH DIMENSIONS", 1)[0])
                selected, = actions
                action_text = selected["action"]
                assert action_text in plan_by_action
                assert not any(action in prompt for action in plan_by_action if action != action_text)
                if sample == 1 and sample not in settlement_actions:
                    settlement_actions[sample] = actions
                    raise LLMError("Temporary settlement failure")
                if sample in settlement_actions:
                    assert actions == settlement_actions[sample]
                settlement_actions[sample] = actions
                return schema(
                    state_summary="Selected decision settled",
                    next_world_state=frozen["world_state"] | {"selected": action_text},
                    agent_knowledge_updates=[{"agent_id": "actor_a", "information": []}],
                    analysis="Only the selected decision was resolved",
                )
            raise AssertionError((sample, name))

    request = EnsembleRequest(
        event_title="Supply", model="fixture-model", planning_mode="rules", step_unit="day",
        start_date="2026-01-01", sample_count=2, steps=1, agents=[actor().model_dump()],
        selected_actions=[SelectedAction(agent_id="actor_a", action="Decide on today's supply plan")],
        decision_policy="weighted", decision_seed=0,
        initial_world_state={"selected": ""},
        outcome_definitions=[
            OutcomeDefinition(id="one", label="One", criterion="Option one selected", conditions=[
                OutcomeCondition(path="selected", operator="eq", value="OPTION_ONE")]),
            OutcomeDefinition(id="other_option", label="Another", criterion="Another option selected", conditions=[
                OutcomeCondition(path="selected", operator="ne", value="OPTION_ONE")]),
        ],
    )
    store = EnsembleStore(tmp_path / "runs")
    events = EventRepository(tmp_path / "events")
    run_id, request = prepare_experiment(request, events=events, store=store)
    config = store.load(run_id, "config")
    assert config["decision_sampling_version"] == VERSION and config["request"]["decision_seed"] == 0
    partial = asyncio.run(run_experiment(run_id, request, store=store, runtime=Model()))
    assert partial["completed_samples"] == 1 and partial["failed_samples"] == 1
    directory = store.directory(run_id) / "round_stages"
    first, = saved_draws(directory / "0001")
    second, = saved_draws(directory / "0002")
    assert first["audit"][0]["draw"] != second["audit"][0]["draw"]
    before = len(calls)
    _, resumed = prepare_experiment(EnsembleRequest(
        event_title="Supply", resume_run_id=run_id, decision_policy="single", decision_seed=19,
    ), events=events, store=store)
    assert resumed.decision_policy == "weighted" and resumed.decision_seed == 0
    complete = asyncio.run(run_experiment(run_id, resumed, store=store, runtime=Model()))
    assert calls[before:] == [(1, "GameMasterSettlement")]
    assert complete["completed_samples"] == 2
    assert saved_draws(directory / "0001") == [first]
    for sample in (1, 2):
        result = store.sample(run_id, sample)
        selected = settlement_actions[sample][0]["action"]
        assert result["snapshot"]["world_state"]["selected"] == selected
        assert result["snapshot"]["agents"][0]["current_plan"]["description"] == plan_by_action[selected]
        assert [a["action"] for a in result["rounds"][0]["agent_actions"]] == [selected]
        assert result["rounds"][0]["decision_sampling"]["multi_candidate_decisions"] == 1


def test_resuming_config_saved_before_sampling_fields_defaults_to_single(tmp_path):
    store = EnsembleStore(tmp_path / "runs")
    events = EventRepository(tmp_path / "events")
    original = EnsembleRequest(event_title="Supply", agents=[actor().model_dump()], planning_mode="rules")
    run_id, _ = prepare_experiment(original, events=events, store=store)
    config = store.load(run_id, "config")
    config["request"].pop("decision_policy")
    config["request"].pop("decision_seed")
    store.save(run_id, "config", config)

    _, resumed = prepare_experiment(EnsembleRequest(
        event_title="Supply", resume_run_id=run_id, decision_policy="weighted", decision_seed=3,
    ), events=events, store=store)
    assert resumed.decision_policy == "single"
    assert resumed.decision_seed is None
    assert "decision_sampling_version" not in store.load(run_id, "config")


def test_duplicate_text_cannot_add_weight_to_an_existing_candidate(tmp_path):
    a = actor()
    snapshot = WorldSnapshot(event_title="Supply", agents=[a], world_state={
        "inventory": {"units": 5}, "simulation_clock": {"sequential_harness": "fixture"},
    })
    sent = []

    class Model:
        async def json_call(self, *, schema, call_name, prompt, **kwargs):
            sent.append(schema.__name__)
            payload = json.loads(prompt)
            if schema.__name__ == "CandidatePool":
                assert call_name.endswith("_initial")
                return schema(candidates=[option("Issue a statement", 15), option("Issue a statement.", 80),
                                          option("Request a review", 5)])
            if schema.__name__ == "CandidateAuditBatch":
                ids = [c["candidate_id"] for c in payload["candidates"]]
                assert ids == ["actor_a:initial:1", "actor_a:initial:3"]
                return schema(round_id=1, snapshot_version=0, reviews=[
                    {"candidate_id": cid, "verdict": "allow"} for cid in ids
                ])
            if schema.__name__ == "CandidateWeights":
                assert payload["context"]["public"]["world_state"]["inventory"]["units"] == 5
                assert "round_briefing" in payload["context"]["public"]
                ids = [c["candidate_id"] for c in payload["options"]]
                assert ids == ["actor_a:initial:1", "actor_a:initial:3"]
                return schema(weights=[{"candidate_id": cid, "weight": 1} for cid in ids])
            raise AssertionError(schema.__name__)

    _, stats = asyncio.run(produce_weighted_actions(
        snapshot, snapshot.agents, [], Model(), "test", seed=17, sample_id=1,
        cache=RoundStages(tmp_path, {"sample": 1}),
        compact_benchmark=True,
    ))
    assert sent == ["CandidatePool", "CandidateAuditBatch", "CandidateWeights"]
    assert stats["duplicate_candidates"] == 1
    draw, = saved_draws(tmp_path)
    assert [row["probability"] for row in draw["audit"][0]["final_pool"]] == [0.5, 0.5]


def test_public_candidate_probabilities_preserve_mass_without_exposing_private_options(tmp_path):
    a = actor()
    state = WorldSnapshot(event_title="Supply", agents=[a])

    class Model:
        async def json_call(self, *, schema, prompt, **kwargs):
            if schema.__name__ == "CandidatePool":
                return schema(candidates=[option("Publish the offer", 20, expected_effect="Recipients may respond"),
                                          option("CONFIDENTIAL defer review", 80, visibility="private")])
            assert schema.__name__ == "CandidateAuditBatch"
            return schema(round_id=1, snapshot_version=0, reviews=[
                {"candidate_id": row["candidate_id"], "verdict": "allow"} for row in json.loads(prompt)["candidates"]])

    _, stats = asyncio.run(produce_weighted_actions(
        state, state.agents, [], Model(), "privacy_fixture", seed=19, sample_id=1,
        cache=RoundStages(tmp_path, {"sample": 1}), record_draws=True,
    ))
    assert "CONFIDENTIAL" not in str(stats)
    row, = stats["sampled_choices"][0]["public_options"]
    assert row["probability"] == .2
