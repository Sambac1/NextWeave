import asyncio
import json

from app.benchmark.contract_mechanisms import ContractUpdate, apply_updates, compile_spec
from app.benchmark.dynamics import discard_unsupported_counts, scheduled_window_event
from app.benchmark.runner import summarize_sequential
from app.domain import ForecastRound, TimelineEvent, WorldSnapshot
from app.domain.forecast import ContractOutlook, GameMasterSettlement
from app.simulation.clock import align_events, explicit_step_clock
from app.simulation.outlook import PublicReadout, read_public_outlook, validate_outlook
from app.simulation.process import suppress_continuations
from app.storage.round_stages import RoundStages


def event(ref, text, *, timestamp="2026-06-30", **kwargs):
    return TimelineEvent(timeline_id=ref, timestamp=timestamp, description=text,
                         confidence=.8, importance=.8, **kwargs)


def frame(timeline=(), **kwargs):
    return ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1,
                         state_summary="Internal summary", timeline=list(timeline), **kwargs)


def test_occurrence_requires_the_right_actor_action_object_and_precise_time():
    spec = compile_spec("Will the minister visit Oslo by July 1?", "An actual visit is required.", scheduled_election=False)
    clock = explicit_step_clock(["2026-06-29", "2026-07-01"], 1, as_of="2026-06-29T12:00:00Z",
                                outcome_deadline="2026-07-01T03:59:00Z")
    rows = [event("weather", "Oslo received heavy rain."),
            event("wrong_actor", "The deputy arrived in Oslo to represent the minister."),
            event("wrong_object", "The minister arrived in Bergen, then read a report from Oslo."),
            event("conditional", "The minister would visit Oslo if authorized."),
            event("denial", "It is not true that the minister arrived in Oslo."),
            event("ambiguous", "The minister arrived in Oslo.", timestamp="2026-07-01"),
            event("too_late", "The minister arrived in Oslo.", timestamp="2026-07-01T04:00:00Z"),
            event("valid", "The minister arrived in Oslo.", timestamp="2026-07-01T03:58:00Z")]
    for row in rows:
        update = ContractUpdate(stage="occurred", event_ref=row.timeline_id, quote=row.description)
        state = apply_updates(spec, {}, [update], [row], clock)
        assert state.get("matched_ref") == ("valid" if row.timeline_id == "valid" else None), row.timeline_id
    align_events([rows[-3], rows[-1]], clock)
    assert rows[-3].date_precision == "interval" and rows[-1].date_precision == "datetime"
    assert rows[-1].timestamp == "2026-07-01T03:58:00+00:00"


def test_meeting_requires_qualifiers_from_rules_and_other_domains_reuse_same_gate():
    clock = {"interval_start": "2026-06-29", "interval_end": "2026-06-30"}
    spec = compile_spec("Will A.B. Chen attend the next US x Example diplomatic meeting?",
                        "Representatives are authorized and the meeting must be in-person and publicly acknowledged.",
                        scheduled_election=False)
    missing = event("missing", "A.B. Chen attended the next U.S.–Example diplomatic meeting by video.")
    supported = event("supported", "A.B. Chen attended the first in-person U.S.–Example diplomatic meeting with authorized representatives and the meeting was publicly acknowledged.")
    assert not apply_updates(spec, {}, [ContractUpdate(stage="occurred", event_ref="missing",
                                                        quote=missing.description)], [missing], clock)
    assert apply_updates(spec, {}, [ContractUpdate(stage="occurred", event_ref="supported",
                                                    quote=supported.description)], [supported], clock)["matched_ref"] == "supported"
    regulation = compile_spec("Will the harbour safety regulation take effect?", "The rule must be in effect.", scheduled_election=False)
    wrong = event("wrong", "The harbour safety regulation was discussed while an unrelated building regulation took effect.")
    assert not apply_updates(regulation, {}, [ContractUpdate(stage="in_force", event_ref="wrong",
                                                             quote=wrong.description)], [wrong], clock)


def test_archived_source_text_is_citable_but_unverified_notes_are_not():
    raw = "The minister led the delegation at the April meeting."
    before = WorldSnapshot(event_title="Talks", timeline=[event("hist_01", "The delegation met in April.", evidence_text=raw)])
    current = frame(continuing_state_notes=["A private intermediary sent a request."], contract_outlook=ContractOutlook(
        p_yes=.4, basis="An earlier meeting establishes precedent.", evidence_ref="hist_01", evidence_quote=raw,
    ))
    result, error = validate_outlook(before, before, current, event_ids={})
    assert error is None and result.evidence_ref == "hist_01.source" and result.evidence_quote == raw
    current.contract_outlook.evidence_ref = "note_1"
    assert validate_outlook(before, before, current, event_ids={}) == (None, "unsupported_quote")


def test_public_readout_isolated_cached_and_unchanged_on_unrelated_activity(tmp_path):
    question = "Will the minister visit Oslo by July 1?"
    contract = {"question": question, "spec": compile_spec(question, "An actual visit is required.",
                                                            scheduled_election=False).model_dump()}
    before = WorldSnapshot(event_title=question, world_state={"contract_progress": "SECRET inbox", "benchmark_contract": contract},
                           timeline=[event("hist_01", "Oslo was the destination for earlier talks.",
                                           evidence_text="The April source confirms the minister led a delegation to Oslo.")])

    class Runtime:
        calls = 0

        async def json_call(self, **kwargs):
            self.calls += 1
            assert "SECRET" not in kwargs["prompt"] and "Internal summary" not in kwargs["prompt"]
            data = json.loads(kwargs["prompt"])
            key = next(key for key, row in data["evidence"].items() if row["kind"] == "historical_source")
            if self.calls == 1:
                key = "invented"
            elif self.calls == 2:
                assert data["previous_validation_error"] == "unsupported_quote"
            probability = .22 if data["final_round"] else .3
            return PublicReadout(outlook=ContractOutlook(p_yes=probability, basis="The prior visit establishes a precedent.",
                evidence_ref=key, evidence_quote="The April source confirms the minister led a delegation to Oslo."))

    async def check():
        cache, runtime, locks = RoundStages(tmp_path, {"model": "fixture"}), Runtime(), {}
        clock = {"step": 1, "horizon_steps": 3, "interval_end": "2026-06-29"}
        first = frame(continuing_state_notes=["SECRET probe was sent."])
        second = frame(continuing_state_notes=["A different SECRET probe was sent."])
        a, b = await asyncio.gather(*(read_public_outlook(before, before, row, contract, clock, runtime, cache, locks, "test")
                                      for row in (first, second)))
        assert a == b and a[0].p_yes == .3 and a[0].evidence_ref == "hist_01.source" and runtime.calls == 2
        before.world_state["contract_outlook"] = a[0].model_dump()
        weather = frame([event("sim_02", "Heavy rain was recorded in Oslo.", origin="simulated")])
        result, issue = await read_public_outlook(before, before, weather, contract, clock | {"step": 2}, runtime, cache, locks, "test")
        assert result.p_yes == .3 and issue is None and runtime.calls == 2
        final, issue = await read_public_outlook(before, before, weather, contract, clock | {"step": 3}, runtime, cache, locks, "test")
        assert issue is None and final.estimate_kind == "terminal_carried_forward" and final.p_yes == .3
        assert runtime.calls == 2

    asyncio.run(check())


def test_terminal_update_must_cite_an_uncredited_settled_development(tmp_path):
    question = "Will the minister visit Oslo by July 1?"
    contract = {"question": question, "spec": compile_spec(question, "An actual visit is required.",
                                                            scheduled_election=False).model_dump()}
    historic = event("hist", "The minister visited Oslo in April.")
    public = event("sim_01", "The minister cancelled the planned visit to Oslo.", origin="simulated", round_id=1)
    prior = ContractOutlook(p_yes=.35, basis="The earlier visit gives a precedent.", evidence_ref="hist",
                            evidence_quote=historic.description)
    before = WorldSnapshot(event_title=question, world_state={"benchmark_contract": contract,
                           "contract_outlook": prior.model_dump()}, timeline=[historic, public])

    class Runtime:
        calls = 0

        async def json_call(self, *, schema, prompt, **kwargs):
            self.calls += 1
            data = json.loads(prompt)
            assert data["previous_estimate"] == {"p_yes": .35}
            if self.calls == 1:
                key, row = next((key, row) for key, row in data["evidence"].items() if row["kind"] == "historical")
                return schema(outlook=ContractOutlook(p_yes=.62, basis="The earlier visit gives a precedent.",
                    evidence_ref=key, evidence_quote=row["text"], change_driver="settled_event"))
            assert data["previous_validation_error"] == "missing_change_driver"
            key, row = next((key, row) for key, row in data["evidence"].items() if row["eligible_update"])
            assert not row["current"] and row["text"] == public.description
            return schema(outlook=ContractOutlook(p_yes=.08, basis="The cancellation reduced the chance of a visit.",
                evidence_ref=key, evidence_quote=row["text"], change_driver="settled_event"))

    async def check():
        runtime, cache = Runtime(), RoundStages(tmp_path, {})
        clock = {"step": 3, "horizon_steps": 3}
        args = (before, before, frame(), contract, clock, runtime, cache, {}, "test")
        estimate, issue = await read_public_outlook(*args)
        assert issue is None and estimate.p_yes == .08 and estimate.evidence_ref == "sim_01"
        assert estimate.estimate_kind == "terminal_estimate" and runtime.calls == 2

        already_cited = [{"contract_outlook": estimate.model_dump()}]
        estimate, issue = await read_public_outlook(*args, prior_rounds=already_cited)
        assert issue is None and estimate.p_yes == .35 and estimate.estimate_kind == "terminal_carried_forward"
        assert runtime.calls == 2

    asyncio.run(check())


def test_program_window_remains_distinct_from_unobserved_count():
    contract = {"scheduled_milestone": {"counting_from": "2026-07-01"}, "spec": {"predicate": "rank"}}
    clock = explicit_step_clock(["2026-06-29", "2026-07-01", "2026-07-03"], 1,
                                as_of="2026-06-29T12:00:00Z", outcome_deadline="2026-07-03T11:00:00Z")
    snapshot = WorldSnapshot(event_title="Election", world_state={"benchmark_contract": contract})
    model_event = event("sim_1_model", "Officials announced the count allocated 42 seats.", change_kind="external_observation")
    settlement = GameMasterSettlement(state_summary="Counting", analysis="Count claim", next_world_state={}, public_events=[model_event])
    discard_unsupported_counts(snapshot, settlement)
    suppress_continuations(snapshot, settlement, [])
    window = scheduled_window_event(contract, clock, 1)
    assert not settlement.public_events and window.event_type == "scheduled_window"
    assert window.timestamp == "2026-07-01T00:00:00+00:00"
    assert scheduled_window_event(contract, clock | {"interval_start_utc": window.timestamp}, 2) is None


def test_audited_silence_is_not_an_invalid_trace():
    report = summarize_sequential([{"status": "completed", "rounds": [{"step": 1,
        "decision_sampling": {"audited_silent_round": 1}, "outlook_issue": "no_terminal_observation"}]}], expected_steps=1)
    assert report["unknown_samples"] == 1 and report["invalid_trace_samples"] == 0 and report["p_yes"] is None


def test_readout_provider_failure_keeps_the_sampled_action_and_committed_round(tmp_path):
    from app.domain.ensemble import BenchmarkSimulation, EnsembleRequest, OutcomeDefinition
    from app.llm.client import LLMError
    from app.simulation.ensemble import prepare_experiment, run_experiment
    from app.storage.ensemble_store import EnsembleStore
    from app.storage.json_store import EventRepository
    from test_ensemble import actor

    question = "Will Laboratory visit Oslo by July 1?"
    rules = "An actual visit is required."
    spec = compile_spec(question, rules, scheduled_election=False)
    observed = "Laboratory announced a visit to Oslo pending approval."
    seen = []

    class Runtime:
        recorder = None

        async def json_call(self, *, schema, prompt, **kwargs):
            name = schema.__name__
            seen.append(name)
            if name == "CandidatePool":
                return schema(candidates=[{"decision": "act", "action_type": "communicate",
                                           "action": "Announce a proposed visit to Oslo.", "reasoning": "Seek approval.",
                                           "weight": 1}])
            if name == "CandidateAuditBatch":
                return schema(round_id=1, snapshot_version=0, reviews=[
                    {"candidate_id": item["candidate_id"], "verdict": "allow"} for item in json.loads(prompt)["candidates"]
                ])
            if name == "SequentialGameMasterSettlement":
                action_rows = json.loads(prompt.split("AUDITED ACTIONS\n", 1)[1].split("\n\nSTRENGTH DIMENSIONS", 1)[0])
                return schema(
                    state_summary="Visit proposed. INTERNAL ONLY.",
                    analysis="Settlement contains an internal observation.",
                    state_changes=[{"path": ["permit_pending"], "operation": "add", "before": None, "after": True,
                                    "reason": "An official announcement described the condition.",
                                    "event_ref": "sim_1_visit", "evidence_quote": observed}],
                    public_events=[event("sim_1_visit", observed, actors=["actor_a"],
                        executed_decisions=[{"actor_id": "actor_a", "action_id": action_rows[0]["action_id"], "quote": observed}])],
                    agent_knowledge_updates=[{"agent_id": "actor_a", "information": ["INTERNAL ONLY"]}],
                    action_results=[{"action_id": action_rows[0]["action_id"], "status": "executed",
                                     "summary": "Announcement published", "event_refs": ["sim_1_visit"]}],
                    question_updates=[{"condition_id": "target_event", "before": "unknown", "after": "pending",
                                       "summary": "Approval remains pending", "event_refs": ["sim_1_visit"],
                                       "evidence_quote": observed, "pending_actor_ids": []}],
                )
            if name == "PublicReadout":
                assert "INTERNAL ONLY" not in prompt
                assert observed in prompt
                raise LLMError("Unavailable provider")
            raise AssertionError(name)

    request = EnsembleRequest(event_title=question, model="fixture", agents=[actor().model_dump()],
        decision_policy="weighted", decision_seed=5, planning_mode="rules", start_date="2026-06-29", steps=1,
        step_unit="day", sample_count=1, update_relationships=False,
        initial_world_state={"benchmark_contract": {"question": question, "rules": rules, "spec": spec.model_dump()}},
        outcome_definitions=[OutcomeDefinition(id="yes", label="Yes", criterion="Actual visit."),
                             OutcomeDefinition(id="no", label="No", criterion="Visit did not occur.")],
        benchmark=BenchmarkSimulation(round_boundaries=["2026-06-29", "2026-07-01"],
            resolution_rules=rules, as_of="2026-06-29T12:00:00Z", outcome_deadline="2026-07-01T03:59:00Z",
            sequential_outcome=True, bdi_updates=False))
    store = EnsembleStore(tmp_path / "runs")
    run_id, request = prepare_experiment(request, events=EventRepository(tmp_path), store=store)
    asyncio.run(run_experiment(run_id, request, store=store, runtime=Runtime()))
    sample = store.sample(run_id, 1)
    assert seen == ["CandidatePool", "CandidateAuditBatch", "SequentialGameMasterSettlement", "PublicReadout"]
    assert sample["status"] == "completed" and sample["snapshot"]["world_state"]["permit_pending"] is True
    round_result, = sample["rounds"]
    assert round_result["outlook_issue"] == "outlook_call_unavailable"
    assert round_result["sampled_choices"][0]["p_selected"] == 1.0
    report = summarize_sequential([sample], expected_steps=1)
    assert report["p_yes"] is None and report["unknown_samples"] == 1 and report["failed_samples"] == 0
    assert report["resume_available"]

    class ResumeRuntime:
        recorder = None

        async def json_call(self, *, schema, prompt, **kwargs):
            seen.append(schema.__name__)
            data = json.loads(prompt)
            assert schema is PublicReadout and data["final_round"] is True
            key = next(key for key, row in data["evidence"].items() if row["text"] == observed)
            return schema(outlook=ContractOutlook(p_yes=.32, basis="The announced visit still needed approval.",
                                                 evidence_ref=key, evidence_quote=observed))

    asyncio.run(run_experiment(run_id, request, store=store, runtime=ResumeRuntime()))
    assert seen[-2:] == ["PublicReadout", "PublicReadout"] and len(seen) == 5
    resumed = store.sample(run_id, 1)
    assert resumed["rounds"][0]["sampled_choices"] == round_result["sampled_choices"]
    assert resumed["rounds"][0]["timeline"] == round_result["timeline"]
    assert summarize_sequential([resumed], expected_steps=1)["p_yes"] == .32
