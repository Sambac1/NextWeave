import asyncio
import json

import pytest

from app.domain import ForecastRound, TimelineEvent, WorldSnapshot
from app.domain.ensemble import BenchmarkSimulation, MainQuestionSimulation, QuestionSimulation
from app.simulation.clock import align_events
from app.simulation.main_forecast import MainForecastRun, admitted_time, read_deadline
from app.simulation.question_report import build_question_report
from app.simulation.weighted_decisions import produce_weighted_actions
from app.storage.ensemble_store import EnsembleStore
from app.storage.round_stages import RoundStages
from test_ensemble import actor


RUN = "00000000-0000-4000-8000-000000000009"
START = "2026-11-09T15:05:00Z"
END = "2026-11-12T14:29:00Z"
APPROVAL = "The board approved the emergency adjustment after reviewing new figures."
QUESTION = "Will the board publish a rate change before the deadline?"
RULES = {"question": QUESTION, "forecast_criteria": {
    "yes": "The board publishes a rate change.",
    "no": "The complete public record shows that the board did not publish a rate change.",
}, "rules": QUESTION, "as_of": START, "outcome_deadline": END}


def world():
    return WorldSnapshot(event_id="fixture", event_title=QUESTION, agents=[actor()],
                         world_state={"benchmark_contract": RULES})


def test_shared_pool_and_rare_branch_that_still_needs_approval(tmp_path):
    from app.simulation.weighted_decisions import MainCandidateAuditBatch, MainCandidatePool

    state = world()
    plan = MainForecastRun(EnsembleStore(tmp_path / "ensemble"), RUN, 2)
    cache = RoundStages(tmp_path / "shared", {"state": 1}, locks=plan.decision_locks)
    calls = []

    class Runtime:
        async def json_call(self, *, schema, prompt, **kwargs):
            calls.append(schema)
            await asyncio.sleep(.001)
            if schema is MainCandidatePool:
                return schema(candidates=[
                    {"decision": "wait", "action_type": "no_action", "action": "Wait for new information.",
                     "visibility": "private", "reasoning": "No new evidence.", "weight": 90,
                     "execution": {"mode": "direct"}},
                    {"decision": "act", "action_type": "publish", "action": "Publish a rate change.",
                     "reasoning": "Requires an emergency assessment.", "weight": 10,
                     "execution": {"mode": "conditional", "prerequisite": "Board assessment approved."}},
                ])
            assert schema is MainCandidateAuditBatch
            payload = json.loads(prompt)
            return schema(round_id=payload["round_id"], snapshot_version=payload["snapshot_version"], reviews=[
                {"candidate_id": row["candidate_id"], "verdict": "allow"} for row in payload["candidates"]
            ])

    async def both():
        return await asyncio.gather(*(
            produce_weighted_actions(state, state.agents, [], Runtime(), f"sample{number}", seed=12,
                                     sample_id=number, cache=cache, record_draws=True, main_forecast=plan)
            for number in (1, 2)
        ))

    ordinary, rare = asyncio.run(both())
    assert calls == [MainCandidatePool, MainCandidateAuditBatch]
    assert plan.sample_masses() == [.9, .1]
    assert ordinary[0][0].decision == "wait"
    assert rare[0][0].decision == "monitor"
    assert rare[0][0].plan_update.review_when == "Board assessment approved."
    assert rare[1]["sampled_choices"][0]["execution_status"] == "awaiting_prerequisite"
    assert rare[1]["sampled_choices"][0]["p_selected"] == .1
    assert MainForecastRun(plan.store, RUN, 2).sample_masses() == [.9, .1]


@pytest.mark.parametrize(("mode", "ready", "origin", "expected"), [
    ("direct", False, "historical", "act"),
    ("attempt", False, "historical", "act"),
    ("conditional", True, "simulated", "act"),
    ("conditional", True, "historical", "monitor"),
    ("conditional", False, "simulated", "monitor"),
])
def test_execution_gate_preserves_feasible_direct_and_approved_actions(mode, ready, origin, expected):
    from app.simulation.main_forecast import gated_action
    from app.simulation.weighted_decisions import MainCandidatePool, MainCandidateReview, _build_candidates

    state = world()
    state.round_id = 1
    state.timeline.append(TimelineEvent(timeline_id="approval", timestamp="2026-11-09",
                                        description=APPROVAL, origin=origin, round_id=1,
                                        confidence=.9, importance=.9))
    draft = {"decision": "act", "action_type": "publish", "action": "Publish the permitted notice.",
             "weight": 1, "reasoning": "Proceed under the decision rules.", "execution": {"mode": mode}}
    if mode == "conditional":
        draft["execution"].update(prerequisite="Board approval", evidence_ref="approval", evidence_quote=APPROVAL)
    candidate, = _build_candidates(MainCandidatePool(candidates=[draft]), state, state.agents[0], "initial")
    review = MainCandidateReview(candidate_id=candidate.id, verdict="allow", prerequisite_met=ready)
    action, _ = gated_action(candidate, state, state.agents[0], review)
    assert action.decision == expected


def test_auditor_can_find_the_prerequisite_that_the_proposer_omitted():
    from app.simulation.main_forecast import gated_action
    from app.simulation.weighted_decisions import MainCandidatePool, MainCandidateReview, _build_candidates

    state = world()
    candidate, = _build_candidates(MainCandidatePool(candidates=[{
        "decision": "act", "action_type": "publish", "action": "Publish the rate adjustment.", "weight": 1,
        "reasoning": "The board can publish it if its emergency assessment calls for a move.",
        "execution": {"mode": "direct"},
    }]), state, state.agents[0], "initial")
    review = MainCandidateReview(candidate_id=candidate.id, verdict="allow",
                                 missing_prerequisite="Emergency assessment supports an adjustment.")
    action, status = gated_action(candidate, state, state.agents[0], review)
    assert action.decision == "monitor" and status == "awaiting_prerequisite"
    assert action.plan_update.review_when == review.missing_prerequisite


def test_round_bounds_support_a_date_without_inventing_an_hour():
    clock = {"interval_start": "2026-11-09", "interval_end": "2026-11-10",
             "interval_start_utc": START, "interval_end_utc": "2026-11-10T00:00:00Z"}
    coarse = TimelineEvent(timeline_id="announcement", timestamp="2026-11-09", description=APPROVAL,
                           origin="simulated", confidence=.9, importance=.9)
    align_events([coarse], clock)
    assert coarse.date_precision == "interval" and coarse.timestamp == "2026-11-09"
    assert admitted_time(coarse, START, END)
    assert not admitted_time(coarse, "2026-11-09T18:00:00Z", END)


@pytest.mark.parametrize(("verdict", "negative_basis", "coverage", "expected"), [
    ("yes", None, False, 1),
    ("no", None, False, None),
    ("no", "complete_window", False, None),
    ("no", "complete_window", True, 0),
])
def test_deadline_requires_supported_outcome_and_explicit_negative_coverage(verdict, negative_basis, coverage, expected):
    state = world()
    clock = {"interval_start": "2026-11-09", "interval_end": "2026-11-12",
             "interval_start_utc": START, "interval_end_utc": END, "horizon_end_utc": END}
    text = ("The complete register recorded no decision between " + START + " and " + END + "."
            if coverage else "The board published the rate change.")
    event = TimelineEvent(timeline_id="verified", timestamp=END, description=text,
                          origin="simulated", confidence=.9, importance=.9)
    align_events([event], clock)
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1,
                          state_summary=text, timeline=[event])

    class Runtime:
        async def json_call(self, *, schema, **kwargs):
            return schema(verdict=verdict, basis=text, negative_basis=negative_basis,
                          evidence_ref="event_1", evidence_quote=text,
                          covered_from=START if coverage else "", covered_through=END if coverage else "")

    outlook, issue = asyncio.run(read_deadline(state, state, frame, RULES, clock, Runtime(), RoundStages(None, {}), "test"))
    assert (outlook.p_yes if outlook else None) == expected
    assert issue == ("terminal_unresolved" if expected is None else None)


def test_report_does_not_carry_forward_early_estimate_or_weight_rare_branch_twice():
    config = {"request": {"steps": 1, "question_simulation": {"algorithm": "evidence-gated-stratified-v1"},
                          "initial_world_state": {"benchmark_contract": RULES}}}

    def record(probability, kind):
        return {"status": "completed", "rounds": [{"step": 1,
            "decision_sampling": {"audited_silent_round": 1},
            "contract_outlook": {"p_yes": probability, "estimate_kind": kind, "basis": APPROVAL},
        }]}

    yes = record(1, "verified_simulation")
    old = record(.08, "terminal_carried_forward")
    partial = build_question_report(RUN, config, [old, yes], masses=[.9, .1])
    assert partial["p_yes"] is None
    assert partial["bounds"] == [.1, 1]
    assert partial["probability_mass"] == {"yes": .1, "no": 0, "unresolved": .9}
    complete = build_question_report(RUN, config, [record(0, "verified_simulation"), yes], masses=[.9, .1])
    assert complete["p_yes"] == .1
    assert build_question_report(RUN, config, [old, yes])["p_yes"] == pytest.approx(.54)


def test_main_version_is_not_the_benchmark_protocol():
    fields = {"round_boundaries": ["2026-11-09", "2026-11-12"], "resolution_rules": QUESTION,
              "outcome_deadline": END, "sequential_outcome": True}
    main = MainQuestionSimulation(**fields, algorithm="evidence-gated-stratified-v1")
    assert "algorithm" not in QuestionSimulation(**fields).model_dump()
    with pytest.raises(ValueError):
        BenchmarkSimulation.model_validate(main.model_dump())


def test_main_engine_runs_two_strata_with_real_round_orchestration(tmp_path):
    from app.domain.decision_core import DecisionCore, DecisionGoal, DecisionItem, SourceSupport
    from app.domain.ensemble import EnsembleRequest, OutcomeDefinition
    from app.simulation.ensemble import prepare_experiment, run_experiment
    from app.storage.json_store import EventRepository

    calls = []
    announcement = "Laboratory published an official rate change."

    class Runtime:
        recorder = None

        async def json_call(self, *, schema, prompt, **kwargs):
            name = schema.__name__
            calls.append(name)
            if name == "MainCandidatePool":
                return schema(candidates=[
                    {"decision": "act", "action_type": "publish", "action": "Publish the authorized rate change.",
                     "reasoning": "The decision may be published immediately.", "weight": 10,
                     "execution": {"mode": "direct"}},
                    {"decision": "act", "action_type": "publish", "action": "Publish an emergency rate change.",
                     "reasoning": "The emergency assessment is not yet complete.", "weight": 90,
                     "execution": {"mode": "conditional", "prerequisite": "Emergency assessment approved."}},
                ])
            if name == "MainCandidateAuditBatch":
                data = json.loads(prompt)
                return schema(round_id=1, snapshot_version=0, reviews=[
                    {"candidate_id": row["candidate_id"], "verdict": "allow"} for row in data["candidates"]
                ])
            if name == "SequentialGameMasterSettlement":
                selected, = json.loads(prompt.split("AUDITED ACTIONS\n", 1)[1].split("\n\nSTRENGTH DIMENSIONS", 1)[0])
                acted = selected["decision"] == "act"
                event_refs = ["sim_1_announcement"] if acted else []
                return schema(
                    state_summary=announcement if acted else "The prerequisite is still unknown.",
                    analysis="Resolved the authorized step only.",
                    state_changes=[],
                    public_events=[{"timeline_id": "sim_1_announcement", "timestamp": "2026-11-09",
                                    "description": announcement, "actors": ["actor_a"], "confidence": .9,
                                    "importance": .9, "cause_action_ids": [selected["action_id"]],
                                    "executed_decisions": [{"actor_id": "actor_a", "action_id": selected["action_id"],
                                                            "quote": announcement}]}] if acted else [],
                    action_results=[{"action_id": selected["action_id"], "status": "executed" if acted else "waiting",
                                     "summary": announcement if acted else "Waiting for approval.", "event_refs": event_refs}],
                    agent_knowledge_updates=[{"agent_id": "actor_a", "information": []}],
                    question_updates=[
                        {"condition_id": condition, "before": "unknown",
                         "after": "met" if acted and condition == "target_event" else "unknown",
                         "summary": "Only report an observed publication.",
                         "event_refs": event_refs if condition == "target_event" else [],
                         "evidence_quote": announcement if acted and condition == "target_event" else ""}
                        for condition in ("target_event", "negative_outcome")
                    ],
                )
            assert name == "DeadlineAudit"
            data = json.loads(prompt)
            ref, = data["settled_public_events"]
            assert data["settled_public_events"][ref]["text"] == announcement
            return schema(verdict="yes", basis=announcement, evidence_ref=ref, evidence_quote=announcement)

    profile = actor()
    authority = "Laboratory sets and publishes official policy rates."
    source = SourceSupport(evidence_id="standing_rule", quote=authority)
    profile.decision_core = DecisionCore(
        mandate=[DecisionItem(description=authority, provenance="source_statement",
                              claim_scope="structural", source_refs=[source])],
        goals=[DecisionGoal(description="Publish timely policy decisions.", source_refs=[source])],
    )
    request = EnsembleRequest(event_title=QUESTION, model="fixture", agents=[profile.model_dump()],
        decision_policy="weighted", decision_seed=5, planning_mode="rules", start_date="2026-11-09", steps=1,
        step_unit="day", sample_count=2, initial_world_state={"benchmark_contract": RULES},
        outcome_definitions=[OutcomeDefinition(id=key, label=key, criterion=value)
                             for key, value in RULES["forecast_criteria"].items()],
        question_simulation=MainQuestionSimulation(algorithm="evidence-gated-stratified-v1",
            round_boundaries=["2026-11-09", "2026-11-12"], resolution_rules=QUESTION, as_of=START,
            outcome_deadline=END, sequential_outcome=True, bdi_updates=False))
    store = EnsembleStore(tmp_path / "runs")
    run_id, resolved = prepare_experiment(request, events=EventRepository(tmp_path), store=store)
    asyncio.run(run_experiment(run_id, resolved, store=store, runtime=Runtime()))
    report = store.load(run_id, "report")
    assert calls.count("MainCandidatePool") == calls.count("MainCandidateAuditBatch") == 1
    assert calls.count("DeadlineAudit") == 1
    assert report["probability_mass"] == {"yes": .1, "no": 0, "unresolved": .9}
    assert report["bounds"] == [.1, 1] and report["p_yes"] is None
    assert [row["terminal_status"] for row in report["trajectories"]] == ["unresolved", "yes"]
    assert store.sample(run_id, 1)["rounds"][0]["timeline"] == []
