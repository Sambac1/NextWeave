from __future__ import annotations

import asyncio
from datetime import date
import json

from fastapi.testclient import TestClient
import pytest

from app.api import benchmark as api
from app.benchmark import runner as benchmark
from app.benchmark.settlement import MarketJudgments
from app.domain.ensemble import BenchmarkSimulation, EnsembleRequest, OutcomeDefinition
from app.main import app
from app.simulation.ensemble import prepare_experiment, run_experiment
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository
from test_ensemble import actor, runner


def market():
    return {
        "case_id": "market_" + "a" * 24, "group_id": "supply",
        "question": "Will the laboratory announce delivery?", "rules": "Yes if it announces delivery.",
        "as_of": "2025-01-01T12:00:00+00:00", "outcome_deadline": "2025-01-08T12:00:00+00:00",
        "evidence": [{"title": "Supply", "url": "https://en.wikipedia.org/wiki/Supply",
                      "revision_id": 123, "revision_at": "2024-12-31T00:00:00+00:00",
                      "text": "The laboratory announced earlier work on this project."}],
    }


def test_market_report_allows_resuming_a_current_harness_checkpoint(monkeypatch, tmp_path):
    cases, runs = EnsembleStore(tmp_path / "cases"), EnsembleStore(tmp_path / "runs")
    monkeypatch.setattr(api, "stores", lambda: (cases, runs))
    run_id = "6f94ca11-308c-4fa3-8eaf-87f8586f285c"
    runs.save(run_id, "status", {"run_id": run_id, "status": "partial", "stage": "finished",
        "methods": ["trajectory_sampling"], "results": [{"status": "failed", "resume_available": True}]})
    runs.save(run_id, "contract_spec", {"trajectory_protocol": benchmark.HARNESS_VERSION})
    assert api.get_benchmark(run_id).get("resume_available") is not False
    runs.save(run_id, "contract_spec", {"trajectory_protocol": "old-harness"})
    assert api.get_benchmark(run_id)["resume_available"] is False


def test_freezes_evidence_and_same_horizon_across_depths():
    case = market()
    evidence = benchmark.frozen_sources(case, "expanded")
    assert evidence[0]["revision_id"] == 123
    assert "announced earlier" in evidence[0]["content"]
    one = benchmark.round_boundaries(date(2025, 1, 1), date(2025, 1, 8), 1)
    multiple = benchmark.round_boundaries(date(2025, 1, 1), date(2025, 1, 8), 3)
    assert one == [multiple[0], multiple[-1]]
    assert multiple == ["2025-01-01", "2025-01-03", "2025-01-05", "2025-01-08"]
    case["evidence"][0]["revision_at"] = "2025-01-02T00:00:00+00:00"
    with pytest.raises(ValueError, match="outside the cutoff"):
        benchmark.frozen_sources(case, "expanded")


def test_contract_clock_uses_rules_without_using_the_observed_result():
    from app.benchmark.dynamics import contract_state, milestone_phase, simulation_case

    case = market() | {
        "question": "Will the Northern Alliance win the fourth most seats?",
        "rules": "Parliamentary elections are scheduled to be held on 8 January 2025.\n\n"
                 "The market resolves by seats. If voting does not occur by March 1, 2025, 11:59 PM ET, resolve Other.",
        "outcome_deadline": "2025-01-08T00:00:00+00:00",
        "deadline_basis": "market_end_date_requires_rule_review",
    }
    adjusted = simulation_case(case)
    assert adjusted["outcome_deadline"] == "2025-01-15T00:00:00+00:00"
    assert adjusted["nominal_market_end"] == case["outcome_deadline"]
    frozen = contract_state(adjusted, benchmark.frozen_sources(case, "expanded"))
    assert frozen["historical_evidence"][0]["revision_id"] == 123
    assert milestone_phase(frozen, {"interval_start": "2025-01-04", "interval_end": "2025-01-08"}) == "pre_count"
    assert milestone_phase(frozen, {"interval_start": "2025-01-08", "interval_end": "2025-01-15"}) == "count_window_opens"
    meeting = case | {"question": "Will a meeting occur?",
                      "rules": 'This market will resolve to "Yes" if a meeting occurs by June 30, 2025, 11:59 PM ET.'}
    assert simulation_case(meeting)["outcome_deadline"] == "2025-07-01T03:59:00+00:00"
    assert simulation_case(case | {"deadline_basis": "confirmed_for_run"})["outcome_deadline"] == case["outcome_deadline"]


def test_outcome_start_is_distinct_from_forecast_cutoff_and_market_creation():
    from app.benchmark.dynamics import outcome_window
    from app.simulation.clock import on_or_after

    metadata = {"startDate": "2025-04-29T18:00:00Z", "createdAt": "2025-04-28T00:00:00Z"}
    explicit = 'This market will resolve to "Yes" if a tariff is announced on or after May 1, 2025, 9:00 AM ET.'
    assert outcome_window(explicit, metadata)["outcome_window_start"] == "2025-05-01T13:00:00+00:00"
    assert outcome_window('The new tariff must be announced by the deadline.', metadata) == {}
    rules = "Any tariff officially announced within the market's time frame qualifies."
    start = outcome_window(rules, metadata)["outcome_window_start"]
    assert outcome_window(rules, {"createdAt": metadata["createdAt"]}) == {}
    assert start == "2025-04-29T18:00:00+00:00"
    assert on_or_after("2025-04-04", start) is False
    assert on_or_after("2025-04-29", start) is None
    assert on_or_after("2025-04-29T18:00:00Z", start) is True


def test_adaptive_round_plan_respects_horizon_and_frozen_milestones():
    case = market() | {
        "rules": "The hearing is scheduled for December 20, 2024. The hearing is scheduled for March 20, 2025. "
                 "Use the decision after the June 11, 2025 board meeting. The hearing is scheduled for January 2, 2026.",
        "outcome_deadline": "2026-01-01T16:00:00Z",
    }
    plan = benchmark.simulation_round_plan(case, "trajectory_sampling", None, {})
    assert plan["mode"] == "adaptive" and plan["steps"] == 8
    assert plan["milestones"] == ["2025-03-20", "2025-06-11"]
    assert plan["boundaries"] == sorted(set(plan["boundaries"]))
    assert len(plan["boundaries"]) == 9
    assert benchmark.simulation_round_plan(case, "one_step", None, {})["boundaries"] == [
        plan["boundaries"][0], plan["boundaries"][-1],
    ]
    fixed = benchmark.simulation_round_plan(case, "trajectory_sampling", 3, {})
    assert fixed["boundaries"] == benchmark.round_boundaries(date(2025, 1, 1), date(2026, 1, 1), 3)
    assert benchmark.simulation_round_plan(market(), "trajectory_sampling", None, {})["steps"] == 3


def test_adaptive_round_plan_can_execute_an_exact_intraday_window():
    from app.simulation.clock import event_in_window, explicit_step_clock

    case = market() | {"outcome_deadline": "2025-01-01T18:00:00+00:00"}
    plan = benchmark.simulation_round_plan(case, "trajectory_sampling", None, {})
    assert plan["steps"] == 1
    request = EnsembleRequest(
        event_title=case["question"], start_date="2025-01-01", steps=plan["steps"], decision_policy="weighted",
        outcome_definitions=[OutcomeDefinition(id="yes", label="Yes", criterion="Delivery announced"),
                             OutcomeDefinition(id="no", label="No", criterion="Delivery not announced")],
        benchmark=BenchmarkSimulation(round_boundaries=plan["boundaries"], resolution_rules=case["rules"],
                                      as_of=case["as_of"], outcome_deadline=case["outcome_deadline"]),
    )
    clock = explicit_step_clock(request.benchmark.round_boundaries, 1, as_of=request.benchmark.as_of,
                                outcome_deadline=request.benchmark.outcome_deadline)
    assert event_in_window("2025-01-01T15:00:00Z", clock) is True
    assert event_in_window("2025-01-01T18:01:00Z", clock) is False
    with pytest.raises(ValueError, match="at least one UTC calendar day"):
        benchmark.simulation_round_plan(case, "trajectory_sampling", 3, {})


def test_trajectory_defaults_increase_capacity_and_preserve_explicit_limits():
    from app.benchmark.evaluation import EvaluationRequest

    options = {"case_id": market()["case_id"], "methods": ["trajectory_sampling"]}
    adaptive = benchmark.BenchmarkRequest(**options)
    assert adaptive.steps is None and adaptive.sample_count == 8
    assert (adaptive.max_model_calls, adaptive.max_total_tokens) == benchmark.sampling_allowance(8, None)
    saved = benchmark.BenchmarkRequest(**options, steps=3, sample_count=2, max_model_calls=150, max_total_tokens=1_800_000)
    restored = benchmark.BenchmarkRequest.model_validate(saved.model_dump())
    assert restored.steps == 3 and restored.sample_count == 2 and restored.max_model_calls == 150
    cohort = EvaluationRequest(methods=["multi_trajectory_simulation"])
    assert cohort.steps is None and cohort.sample_count == 8


def test_benchmark_persists_and_reuses_each_methods_round_plan(tmp_path, monkeypatch):
    store = EnsembleStore(tmp_path / "runs")
    case = market() | {"outcome_deadline": "2025-07-01T12:00:00Z"}
    request = benchmark.BenchmarkRequest(case_id=case["case_id"], methods=["one_step", "trajectory_sampling"])
    run_id = benchmark.new_run(request, store)["run_id"]
    seen = []

    async def compile_contract(contract, *args, **kwargs):
        return contract

    async def prepare(*args, **kwargs):
        return "fixture-event", {"status": "completed"}

    async def run(method, *args, round_plan, **kwargs):
        seen.append((method, round_plan))
        complete = method == "one_step" or len(seen) == 3
        return {"status": "completed" if complete else "partial", "p_yes": .6 if complete else None}

    monkeypatch.setattr(benchmark, "compile_additional_predicate", compile_contract)
    monkeypatch.setattr(benchmark, "prepare_case", prepare)
    monkeypatch.setattr(benchmark, "run_simulation", run)
    asyncio.run(benchmark.run_benchmark(run_id, request, case, store=store))
    saved = store.load(run_id, "status")["round_plans"]
    assert [saved[method]["steps"] for method in request.methods] == [1, 6]

    def refuse_replan(*args, **kwargs):
        pytest.fail("A saved round plan should not be recalculated when resuming")

    monkeypatch.setattr(benchmark, "simulation_round_plan", refuse_replan)
    asyncio.run(benchmark.run_benchmark(run_id, request, case, store=store))
    assert seen[-1] == seen[-2] == ("trajectory_sampling", saved["trajectory_sampling"])
    assert store.load(run_id, "status")["status"] == "completed"


def test_benchmark_keeps_qa_but_skips_simulation_when_the_initiator_is_missing(tmp_path, monkeypatch):
    store = EnsembleStore(tmp_path / "runs")
    events = EventRepository(tmp_path)
    case = market() | {
        "question": "Will Atlas invade Boreal before 2027?",
        "rules": "Yes if Atlas commences a military offensive intended to establish control over part of Boreal.",
    }
    request = benchmark.BenchmarkRequest(case_id=case["case_id"], methods=["direct_qa", "trajectory_sampling"])
    run_id = benchmark.new_run(request, store)["run_id"]

    async def compile_contract(contract, *args, **kwargs):
        return contract

    async def qa(*args, **kwargs):
        return {"status": "completed", "p_yes": 0.2, "coverage": 1}

    async def extract(request, events):
        events.save_artifact(request.event_id, "decision_seat_plan", {
            "status": "limited", "seats": [], "requirements": [
                {"decision_id": "initiate", "kind": "decision", "owner": None,
                 "description": "Atlas leadership authorizes the invasion of Boreal."},
            ],
        })

    async def unexpected_simulation(*args, **kwargs):
        pytest.fail("An unbound initiator must prevent sampling")

    monkeypatch.setattr(benchmark, "compile_additional_predicate", compile_contract)
    monkeypatch.setattr(benchmark, "run_qa", qa)
    monkeypatch.setattr(benchmark, "extract_figures", extract)
    monkeypatch.setattr(benchmark, "run_simulation", unexpected_simulation)
    asyncio.run(benchmark.run_benchmark(run_id, request, case, store=store, events=events))
    saved = store.load(run_id, "status")
    assert [(row["status"], row["p_yes"]) for row in saved["results"]] == [("completed", 0.2), ("unsupported", None)]
    assert saved["preparation"]["actor_coverage"]["missing_essential"] == ["Atlas leadership authorizes the invasion of Boreal."]


def test_retrieves_direct_history_late_in_section_with_revision_identity():
    case = market()
    case["question"] = "Will J.D. Vance attend the next US-Iran diplomatic meeting?"
    case["rules"] = "Yes if Vance attends the next diplomatic meeting between the US and Iran."
    precedent = "Vance was the head of the U.S. delegation at the April negotiations with Iran in Islamabad."
    source = case["evidence"][0]
    source.update(title="JD Vance", url="https://en.wikipedia.org/wiki/JD_Vance",
                  text="== Vice presidency ==\n" + "Biographical background unrelated to this case. " * 100 + precedent)
    packet = benchmark.frozen_sources(case, "expanded")
    assert packet[0]["revision_id"] == 123
    assert precedent in packet[0]["content"]
    assert sum(len(page["content"]) for page in packet) <= 16000


def test_frozen_evidence_keeps_numeric_table_rows_with_their_headers():
    case = market()
    case["question"] = "Will the Northern Alliance win the most seats in the 2025 regional election?"
    case["rules"] = "Yes if the Northern Alliance finishes first by seats."
    source = case["evidence"][0]
    header = '{| class="wikitable"\n|-\n! Poll date\n! NA\n! C'
    rows = ["|-\n| 28 December\n| 41%\n| 36%"] + ["|-\n| Earlier poll\n| 35%\n| 40%"] * 550
    source.update(title="Polling for the 2025 regional election",
                  text="== Opinion polls ==\n=== 2025 ===\n" + header + "\n" + "\n".join(rows) + "\n|}")
    packet = benchmark.frozen_sources(case, "expanded")[0]["content"]
    assert "! Poll date\n! NA\n! C" in packet
    assert "| 28 December\n| 41%\n| 36%" in packet


@pytest.mark.parametrize("quote,expected", [("A recorded laboratory move", "yes"), ("Fabricated terminal quote", "other")])
def test_single_market_trajectory_uses_verified_contract_judgment(tmp_path, quote, expected):
    class Model:
        recorder = None

        async def json_call(self, **kwargs):
            assert kwargs["schema"] is MarketJudgments
            payload = json.loads(kwargs["prompt"])
            item = payload["samples"][0]
            return MarketJudgments(samples=[{"sample_id": item["sample_id"], "answer": "yes",
                                              "refs": list(item["public_evidence"]), "quote": quote}])

    status_store = EnsembleStore(tmp_path / "runs")
    events = EventRepository(tmp_path)
    params = EnsembleRequest(
        event_title="Laboratory", model="fixture", agents=[actor().model_dump()],
        decision_policy="weighted", decision_seed=1, planning_mode="rules",
        start_date="2025-01-01", steps=1, step_unit="day", sample_count=1,
        outcome_definitions=[OutcomeDefinition(id=x, label=x, criterion=x) for x in ("yes", "no")],
        benchmark=BenchmarkSimulation(round_boundaries=["2025-01-01", "2025-01-08"],
                                      resolution_rules="Fixture rule", outcome_deadline="2025-01-08"),
    )
    run_id, request = prepare_experiment(params, events=events, store=status_store)
    delegate = runner([])

    async def round_runner(snapshot, interventions, prefix):
        state, result = await delegate(snapshot, interventions, prefix)
        result.timeline[0].description = "A recorded laboratory move"
        state.timeline[-1].description = "A recorded laboratory move"
        return state, result

    result = asyncio.run(run_experiment(run_id, request, store=status_store, runtime=Model(),
                                       round_runner=round_runner))
    assert result["samples"][0]["classification"]["category_id"] == expected
    assert result["categories"][0]["sampling_interval_95"] is None
    assert status_store.sample(run_id, 1)["rounds"][0]["simulation_interval"]["interval_end"] == "2025-01-08"


def test_sequential_rounds_pass_grounded_outlook_forward_without_terminal_judge(tmp_path):
    from app.domain.forecast import ActionProposal, ContractOutlook

    store = EnsembleStore(tmp_path / "runs")
    params = EnsembleRequest(
        event_title="Trade talks", model="fixture", agents=[actor().model_dump()],
        decision_policy="weighted", decision_seed=1, planning_mode="rules",
        start_date="2025-01-01", steps=2, step_unit="day", sample_count=1,
        outcome_definitions=[OutcomeDefinition(id=x, label=x, criterion=x) for x in ("yes", "no")],
        benchmark=BenchmarkSimulation(round_boundaries=["2025-01-01", "2025-01-02", "2025-01-03"],
                                      resolution_rules="Agreement signed", outcome_deadline="2025-01-03",
                                      sequential_outcome=True),
    )
    run_id, params = prepare_experiment(params, events=EventRepository(tmp_path), store=store)
    delegate = runner([])

    async def round_runner(snapshot, interventions, prefix):
        step = snapshot.round_id + 1
        if step == 2:
            assert snapshot.world_state["contract_outlook"]["p_yes"] == .3
        state, frame = await delegate(snapshot, interventions, prefix)
        quote = ["The counterpart received the offer.", "The counterpart agreed to the terms."][step - 1]
        frame.timeline[0].description = state.timeline[-1].description = quote
        frame.contract_outlook = ContractOutlook(p_yes=.3 if step == 1 else .8,
                                                basis="Public negotiations progressed.", evidence_quote=quote)
        frame.active_agent_ids = ["actor_a"]
        frame.agent_actions = [ActionProposal(
            agent_id="actor_a", agent_name="Laboratory", action_id=f"action_{step}", round_id=step,
            snapshot_version=step - 1, decision="act", action_type="communicate", action="Communicate terms",
            reasoning="Respond to the current offer", confidence=.5,
        )]
        frame.sampled_choices = [{"actor_id": "actor_a", "candidate_id": f"candidate_{step}",
                                  "p_selected": .5 if step == 1 else .6}]
        return state, frame

    class NoTerminalCall:
        async def json_call(self, **kwargs):
            pytest.fail("Sequential aggregation must not call a terminal judge")

    asyncio.run(run_experiment(run_id, params, store=store, runtime=NoTerminalCall(), round_runner=round_runner))
    result = benchmark.summarize_sequential([store.sample(run_id, 1)], expected_steps=2)
    assert result["p_yes"] == .8
    assert result["trajectories"][0]["action_path_probability"] == .3


def test_an_old_quote_does_not_justify_a_new_probability():
    from app.domain import ForecastRound, WorldSnapshot
    from app.domain.forecast import ContractOutlook
    from app.simulation.outlook import validate_outlook

    quote = "The parties have previously exchanged letters."
    before = WorldSnapshot(event_title="Talks", world_state={
        "benchmark_contract": {"historical_evidence": [{"excerpt": quote}]}, "contract_outlook": {"p_yes": .3},
    })
    after = before.model_copy(deep=True)
    frame = ForecastRound(step=2, input_snapshot_version=1, output_snapshot_version=2,
                          state_summary="No public change", analysis="No new facts",
                          contract_outlook=ContractOutlook(p_yes=.8, basis="The parties have talked before.", evidence_quote=quote))
    assert validate_outlook(before, after, frame, event_ids={}) == (None, "missing_change_driver")
    frame.contract_outlook.p_yes = 1
    assert validate_outlook(before, after, frame, event_ids={}) == (None, "unverified_certainty")


def test_exact_event_reference_does_not_override_a_wrong_quote():
    from app.domain import ForecastRound, TimelineEvent, WorldSnapshot
    from app.domain.forecast import ContractOutlook
    from app.simulation.outlook import validate_outlook

    before = WorldSnapshot(event_title="Talks", world_state={"contract_outlook": {"p_yes": .3}})
    event = TimelineEvent(timeline_id="normalized", timestamp="2025-01-02",
                          description="The chair is ready to lead if the president designates him.",
                          confidence=.5, importance=.5)
    frame = ForecastRound(step=2, input_snapshot_version=1, output_snapshot_version=2,
                          state_summary="New public statement", analysis="An appointment is pending",
                          timeline=[event], contract_outlook=ContractOutlook(
                              p_yes=.32, basis="The chair signaled readiness but is not yet appointed.",
                              evidence_ref="original", evidence_quote="if the president designates me",
                              change_driver="settled_event",
                          ))
    assert validate_outlook(before, before, frame, event_ids={"original": "normalized"}) == (None, "unsupported_quote")
    frame.contract_outlook.evidence_quote = event.description
    outlook, issue = validate_outlook(before, before, frame, event_ids={"original": "normalized"})
    assert not issue and outlook.evidence_ref == "normalized" and outlook.evidence_quote == event.description
    frame.contract_outlook.evidence_ref = "invented"
    assert validate_outlook(before, before, frame, event_ids={}) == (None, "unsupported_quote")


def test_api_import_and_qa_persist_without_credentials(tmp_path, monkeypatch):
    case_store = EnsembleStore(tmp_path / "cases")
    run_store = EnsembleStore(tmp_path / "runs")
    seen = []

    async def importer(**kwargs):
        return market()

    class FakeModel:
        def __init__(self, options):
            from app.llm.budget import ModelBudget
            self.budget = ModelBudget(options)

        async def json_call(self, **kwargs):
            seen.append(json.loads(kwargs["prompt"]))
            return benchmark.DirectForecast(p_yes=0.67)

    monkeypatch.setattr(api, "stores", lambda: (case_store, run_store))
    monkeypatch.setattr(api, "import_market", importer)
    monkeypatch.setattr(benchmark, "LLMClient", FakeModel)
    with TestClient(app) as client:
        imported = client.post("/api/benchmark/cases", json={"market_url": "https://polymarket.com/market/example"})
        assert imported.status_code == 200
        data = imported.json()
        assert "text" not in data["evidence"][0]
        unconfirmed = client.post("/api/benchmark/runs", json={"case_id": data["case_id"], "methods": ["one_step"]})
        assert unconfirmed.status_code == 422
        submitted = client.post("/api/benchmark/runs", json={
            "case_id": data["case_id"], "methods": ["direct_qa"], "api_key": "fixture-key",
        })
        assert submitted.status_code == 202
        run_id = submitted.json()["run_id"]
        for _ in range(15):
            status = client.get(f"/api/benchmark/runs/{run_id}").json()
            if status["status"] == "completed":
                break
        assert status["results"][0]["p_yes"] == 0.67
        assert seen[0]["sources"][0]["revision_id"] == 123
        assert "api_key" not in run_store.load(run_id, "config")


def test_qa_resume_reuses_completed_method_and_sample(tmp_path, monkeypatch):
    from app.llm.budget import BudgetExceeded, ModelBudget

    store = EnsembleStore(tmp_path / "runs")
    request = benchmark.BenchmarkRequest(
        case_id=market()["case_id"], methods=["direct_qa", "qa_ensemble"], qa_samples=2,
    )
    seen = []

    class FakeModel:
        def __init__(self, options):
            self.budget = ModelBudget(options)

        async def json_call(self, **kwargs):
            call = kwargs["call_name"].split("_", 1)[1]
            seen.append(call)
            if call == "qa_ensemble_2" and seen.count(call) == 1:
                raise BudgetExceeded("Fixture stops after one QA-ensemble answer")
            return benchmark.DirectForecast(p_yes=0.55)

    monkeypatch.setattr(benchmark, "LLMClient", FakeModel)
    run_id = benchmark.new_run(request, store)["run_id"]
    asyncio.run(benchmark.run_benchmark(run_id, request, market(), store=store))
    assert store.load(run_id, "status")["status"] == "partial"
    asyncio.run(benchmark.run_benchmark(run_id, request, market(), store=store))
    assert store.load(run_id, "status")["status"] == "completed"
    assert seen == ["direct_qa_1", "qa_ensemble_1", "qa_ensemble_2", "qa_ensemble_2"]


@pytest.mark.parametrize("projected,settled", [(False, False), (True, False), (True, True)])
def test_market_terminal_branches_keep_probability_and_resume_only_missing_work(tmp_path, projected, settled):
    from app.benchmark.branch_probability import summarize_branches
    from app.domain import SelectedAction
    from app.llm.client import LLMError
    from test_weighted_decisions import option

    texts = ["Confirm delivery completed", "Confirm the delivery was cancelled", "Delivery remains unresolved"]
    calls = []
    adjudicated = []
    forecasts = []

    class Model:
        recorder = None

        async def json_call(self, *, schema, prompt, **kwargs):
            if schema.__name__ == "CandidatePool":
                return schema(candidates=[option(text, weight) for text, weight in zip(texts, (45, 35, 20))])
            if schema.__name__ == "CandidateAuditBatch":
                return schema(round_id=1, snapshot_version=0, reviews=[
                    {"candidate_id": row["candidate_id"], "verdict": "allow"}
                    for row in json.loads(prompt)["candidates"]
                ])
            if schema.__name__ == "GameMasterSettlement":
                frozen = json.loads(prompt.split("FROZEN PUBLIC SNAPSHOT\n", 1)[1].split("\n\nUSER INTERVENTIONS", 1)[0])
                selected, = json.loads(prompt.split("AUDITED ACTIONS\n", 1)[1].split("\n\nSTRENGTH DIMENSIONS", 1)[0])
                text = selected["action"]
                assert all(other not in prompt for other in texts if other != text)
                calls.append(text)
                if text == texts[1] and calls.count(text) == 1:
                    raise LLMError("Fixture branch temporarily unavailable")
                world = frozen["world_state"] | ({"contract_progress": text, "benchmark_contract": "mutated"} if settled else {})
                return schema(state_summary=text, next_world_state=world, analysis=text,
                              agent_knowledge_updates=[{"agent_id": "actor_a", "information": []}],
                              public_events=[{"timeline_id": "sim_1_fixture", "timestamp": "2025-01-05",
                                              "description": text, "actors": ["actor_a"],
                                              "confidence": .8, "importance": .8}])
            if schema.__name__ == "BranchForecastBatch":
                payload = json.loads(prompt)
                assert "public_evidence" not in payload and "state_before_actions" in payload
                ids = [branch["branch_id"] for branch in payload["branches"]]
                if settled:
                    assert payload["state_before_actions"]["world_state"]["benchmark_contract"]["historical_evidence"]
                    for branch in payload["branches"]:
                        outcome = branch["settled_outcome"]
                        expected = texts[branch["branch_id"] - 1]
                        assert outcome["progress"] == expected
                        assert [row["description"] for row in outcome["public_evidence"].values()] == [expected]
                        assert outcome["replay_verdict"] == "no"
                forecasts.append(ids)
                return schema(forecasts=[{"branch_id": i, "p_yes": (.1, .2, .3)[i - 1],
                                          "yes_path": "Delivery succeeds", "no_path": "Delivery fails"}
                                         for i in ids])
            assert schema is MarketJudgments
            rows = []
            for row in json.loads(prompt)["samples"]:
                ref, event = next(iter(row["public_evidence"].items()))
                text = event["description"]
                adjudicated.append(text)
                rows.append({"sample_id": row["sample_id"], "refs": [ref], "quote": text,
                             "answer": "no" if projected else {texts[0]: "yes", texts[1]: "no", texts[2]: "unknown"}[text]})
            return schema(samples=rows)

    store, events = EnsembleStore(tmp_path / "runs"), EventRepository(tmp_path)
    params = EnsembleRequest(
        event_title="Laboratory", model="fixture", agents=[actor().model_dump()],
        decision_policy="weighted", decision_seed=1, planning_mode="rules",
        start_date="2025-01-01", steps=1, step_unit="day", sample_count=1,
        initial_world_state={"benchmark_contract": {"historical_evidence": [{"excerpt": "Shipping records"}]}} if settled else {},
        selected_actions=[SelectedAction(agent_id="actor_a", action="Consider what to announce")],
        outcome_definitions=[OutcomeDefinition(id=x, label=x, criterion=x) for x in ("yes", "no")],
        benchmark=BenchmarkSimulation(round_boundaries=["2025-01-01", "2025-01-08"],
                                      resolution_rules="Yes if delivery completed", outcome_deadline="2025-01-08",
                                      terminal_branching=True, terminal_probability="forecast" if projected else "replay"),
    )
    run_id, request = prepare_experiment(params, events=events, store=store)
    first = asyncio.run(run_experiment(run_id, request, store=store, runtime=Model()))
    mode = request.benchmark.terminal_probability
    partial = summarize_branches(first["samples"], 1, .5, mode=mode, settled_context=settled)
    assert partial["p_yes"] == pytest.approx(.28 if settled else .175 if projected else .725)
    assert partial["unknown_bounds"] == pytest.approx([.105, .455] if settled else [.175, .175] if projected else [.45, 1.0])
    assert partial["trajectories"][0]["branches"][1]["error"]
    assert store.sample(run_id, 1)["rounds"] == []
    assert adjudicated == [texts[0], texts[2]]

    _, resumed = prepare_experiment(params.model_copy(update={"resume_run_id": run_id}), events=events, store=store)
    result = asyncio.run(run_experiment(run_id, resumed, store=store, runtime=Model()))
    summary = summarize_branches(result["samples"], 1, .5, mode=mode, settled_context=settled)
    assert calls == [*texts, texts[1]]
    assert adjudicated == [texts[0], texts[2], texts[1]]
    assert summary["p_yes"] == pytest.approx(.175 if projected else .55)
    assert summary["coverage"] == pytest.approx(1 if projected else .8)
    assert summary["settled_replay_coverage"] == pytest.approx(1 if projected else .8)
    assert summary["unknown_bounds"] == pytest.approx([.175, .175] if projected else [.45, .65])
    if projected:
        assert forecasts == ([[1, 3], [2]] if settled else [[1, 2, 3]])
        assert all(row["answer"] == "no" for row in summary["trajectories"][0]["branches"])
    assert summary["trajectories"][0]["branches"][0]["evidence"][0]["text"] == texts[0]
    assert store.sample(run_id, 1)["status"] == "completed"


def test_joint_tail_mass_and_equal_prefix_aggregation():
    from app.benchmark.branch_probability import summarize_branches, terminal_branches

    pools = [[{"id": str(i), "probability": p} for i, p in enumerate(probabilities)]
             for probabilities in ((.6, .4), (.75, .25))]
    branches = terminal_branches(pools, seed=9, sample_id=3)
    assert [b["mass"] for b in branches] == pytest.approx([.45, .30, .25])
    assert branches[2]["choices"][1]["id"] == "1"
    assert branches[2]["joint_action_probability"] in (.6 * .25, .4 * .25)

    def record(index, yes):
        return {"sample_id": index, "status": "completed", "market_probability":
                {"yes_mass": yes, "no_mass": 1 - yes, "unknown_mass": 0, "branches": []}}

    samples = [record(i, .3 if i <= 6 else .7) for i in range(1, 11)]
    assert summarize_branches(samples, 10, .5)["p_yes"] == pytest.approx(.46)
    assert summarize_branches([], 2, .5)["p_yes"] is None


def test_archived_rescore_uses_run_id_without_replacing_original(tmp_path, monkeypatch):
    from app.storage.ensemble_store import EnsembleStore

    stores = (EnsembleStore(tmp_path / "cases"), EnsembleStore(tmp_path / "runs"))
    monkeypatch.setattr(api, "stores", lambda: stores)
    client = TestClient(app)
    response = client.get("/api/benchmark/runs/506bdcaa-8ffd-5a94-9970-385edb8b148d")
    assert response.status_code == 200
    run = response.json()
    assert run["source_run_id"] != run["run_id"] and run["resume_available"] is False
    assert run["case_snapshot"]["case_id"] == run["case_id"]
    assert run["market_reference"]["label"] == "Later market snapshot"
    assert [row["p_yes"] for row in run["results"]] == pytest.approx([.08, .065, .1009075, .0653625])
    assert [len(row["rounds"]) for row in run["results"][-1]["trajectories"]] == [2, 2]
    assert client.get("/api/benchmark/runs/not-a-run").status_code == 422


def test_recent_runs_and_active_simulation_progress(tmp_path, monkeypatch):
    from app.storage.ensemble_store import EnsembleStore

    cases = EnsembleStore(tmp_path / "cases")
    runs = EnsembleStore(tmp_path / "runs")
    monkeypatch.setattr(api, "stores", lambda: (cases, runs))
    run_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    simulation_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    monkeypatch.setitem(api._tasks, run_id, object())
    runs.save(run_id, "status", {"run_id": run_id, "status": "running",
                                "stage": "trajectory_sampling", "methods": ["trajectory_sampling"]})
    runs.save(run_id, "simulation_ids/trajectory_sampling", simulation_id)
    EnsembleStore(runs.directory(run_id) / "simulations").save(simulation_id, "status", {
        "stage": "sampling", "sample_count": 4, "completed_samples": 2,
        "completed_rounds": 6, "failed_samples": 0,
    })
    client = TestClient(app)
    recent = client.get("/api/benchmark/runs").json()
    assert [row["run_id"] for row in recent] == [run_id]
    assert "results" not in recent[0]
    progress = client.get(f"/api/benchmark/runs/{run_id}").json()["progress"]
    assert (progress["completed_samples"], progress["sample_count"], progress["completed_rounds"]) == (2, 4, 6)
