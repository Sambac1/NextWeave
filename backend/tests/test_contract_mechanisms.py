import asyncio
from types import SimpleNamespace

from app.benchmark.contract_mechanisms import CompiledPredicate, ContractUpdate, apply_updates, compile_spec
from app.benchmark.runner import BenchmarkRequest, compile_additional_predicate, new_run, run_benchmark
from app.domain.common import LLMOptions
from app.benchmark.settlement import MarketJudgments, settle_records
from app.domain.ensemble import BenchmarkSimulation
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository


def test_rank_contract_runs_sequential_actor_methods_without_poll_projection(tmp_path, monkeypatch):
    from app.benchmark import runner
    text = """== Electoral system ==
The Assembly is made up of 50 seats elected by proportional representation with an electoral threshold of 4% of the national vote.

== 2025 ==
{| class="wikitable"
!Polling firm
!Fieldwork date
!Sample size
|-
!A
!B
!C
|-
|Institute One
|10–14 August
|1000
|46
|29
|25
|-
|Institute Two
|10–16 August
|1000
|43
|32
|25
|}
"""
    case = {
        "case_id": "fixture", "question": "Will Alliance (A) win the first most seats?",
        "rules": "Elections are scheduled on 9 September 2025. The contract resolves based on ranking by seats.",
        "as_of": "2025-08-20T00:00:00Z", "outcome_deadline": "2025-09-16T00:00:00Z",
        "evidence": [{"url": "https://en.wikipedia.org/wiki/Fixture", "title": "Fixture 2025 election",
                      "revision_id": 42, "revision_at": "2025-08-17T00:00:00Z", "text": text}],
    }
    store = EnsembleStore(tmp_path / "runs")
    calls = []

    async def prepare(*args, **kwargs):
        calls.append("actor_preparation")
        return "fixture-event", {"status": "completed"}

    async def simulation(method, request, case, event_id, *, contract, **kwargs):
        assert event_id == "fixture-event" and contract["spec"]["predicate"] == "rank" and "projection" not in contract
        calls.append(method)
        return {"status": "completed", "p_yes": .27, "coverage": 1}

    async def qa(method, *args, **kwargs):
        calls.append(method)
        return {"status": "completed", "p_yes": .35, "coverage": 1, "calls": 1}

    monkeypatch.setattr(runner, "prepare_case", prepare)
    monkeypatch.setattr(runner, "run_simulation", simulation)
    monkeypatch.setattr(runner, "run_qa", qa)
    request = BenchmarkRequest(case_id="fixture", methods=["direct_qa", "one_step", "trajectory_sampling"],
                               steps=3, sample_count=3)
    run = new_run(request, store)["run_id"]
    asyncio.run(run_benchmark(run, request, case, store=store, events=EventRepository(tmp_path)))
    direct, single, multi = store.load(run, "status")["results"]
    assert calls == ["direct_qa", "actor_preparation", "one_step", "trajectory_sampling"] and direct["p_yes"] == .35
    for result in (single, multi):
        assert result["status"] == "completed" and result["p_yes"] == .27
    assert "projection" not in store.load(run, "contract_spec")


def test_sequential_outlooks_use_sampled_likelihood_only_for_diagnostics():
    from app.benchmark.runner import summarize_sequential

    def record(draws, probability):
        return {"status": "completed", "rounds": [
            {"step": i, "sampled_choices": [{"p_selected": draw}],
             "contract_outlook": {"p_yes": probability if i == len(draws) else .2}}
            for i, draw in enumerate(draws, 1)
        ]}

    records = [record([.5, .55, .65], .8), record([.3, .25, .5], .1)]
    result = summarize_sequential(records)
    assert result["p_yes"] == .45
    assert result["trajectories"][0]["action_path_probability"] == .5 * .55 * .65
    missing = summarize_sequential([*records, {"status": "failed"}])
    assert missing["p_yes"] is None and missing["coverage"] == 2 / 3
    assert missing["failed_samples"] == 1 and missing["unknown_bounds"][0] == .9 / 3


def test_sequential_coverage_requires_a_valid_trace_for_each_trajectory():
    from app.benchmark.runner import summarize_sequential

    settled = {"step": 1, "contract_outlook": {"p_yes": .4}, "active_agent_ids": ["actor_a"],
               "agent_actions": [{"agent_id": "actor_a", "decision": "wait"}],
               "sampled_choices": [{"actor_id": "actor_a", "candidate_id": "wait", "p_selected": 1.0}]}
    no_trace = {"step": 1, "contract_outlook": {"p_yes": .9}, "sampled_choices": []}
    wrong_actor = settled | {"sampled_choices": [{"actor_id": "actor_b", "candidate_id": "move", "p_selected": .7}]}
    rows = [{"status": "completed", "rounds": [frame]} for frame in (settled, no_trace, wrong_actor)]
    report = summarize_sequential(rows, expected_steps=1)
    assert report["p_yes"] is None and report["covered_p_yes"] == .4
    assert report["coverage"] == 1 / 3 and report["invalid_trace_samples"] == 2
    assert report["unknown_bounds"] == [.4 / 3, (.4 + 2) / 3]
    assert report["resume_available"] is False


def test_two_round_sampling_matches_the_exact_conditional_expectation():
    from app.benchmark.runner import summarize_sequential
    from app.simulation.weighted_decisions import _draw

    records = []
    for sample_id in range(1, 1025):
        first = [.6, .4]
        route, _ = _draw(123, sample_id, 1, "actor_a", first)
        second = [.8, .2] if route == 0 else [.25, .75]
        finish, _ = _draw(123, sample_id, 2, "actor_b", second)
        rounds = [
            {"step": step, "active_agent_ids": [actor], "agent_actions": [{"agent_id": actor}],
             "sampled_choices": [{"actor_id": actor, "candidate_id": str(choice), "p_selected": p}]}
            for step, actor, choice, p in ((1, "actor_a", route, first[route]), (2, "actor_b", finish, second[finish]))
        ]
        rounds[-1]["contract_outlook"] = {"p_yes": .9 if finish == 0 else .2}
        records.append({"status": "completed", "rounds": rounds})
    exact = .6 * (.8 * .9 + .2 * .2) + .4 * (.25 * .9 + .75 * .2)
    report = summarize_sequential(records, expected_steps=2)
    assert report["coverage"] == 1
    assert abs(report["p_yes"] - exact) < .035


def test_visit_cancelled_then_rescheduled_requires_actual_occurrence():
    spec = compile_spec("Will the minister visit Oslo by Friday?", "Actual arrival is required.", scheduled_election=False)
    assert spec.route == "actor" and spec.subject == "the minister"
    clock = {"interval_start": "2025-01-01", "interval_end": "2025-01-05"}
    events = [SimpleNamespace(timeline_id=str(i), timestamp=f"2025-01-0{i + 2}", description=text)
              for i, text in enumerate(["The minister announced a visit to Oslo.", "The minister cancelled the Oslo visit.",
                                        "The minister arrived in Oslo."])]
    updates = [ContractUpdate(stage=stage, event_ref=event.timeline_id, quote=event.description)
               for stage, event in zip(("planned", "cancelled", "occurred"), events)]
    prior = apply_updates(spec, {}, updates[:2], events, clock)
    assert prior["stage"] == "cancelled" and "matched_ref" not in prior
    assert apply_updates(spec, prior, updates[2:], events, clock)["matched_ref"] == "2"


def test_ceasefire_duration_requires_explicit_contiguous_observation():
    spec = compile_spec("Will the ceasefire last at least 4 days?", "Continuous coverage is required.", scheduled_election=False)
    assert spec.route == "hybrid" and spec.duration_days == 4

    def advance(prior, stage, day, left, coverage=None):
        text = "Both parties signed the ceasefire." if stage == "signed" else "The ceasefire remained in force."
        if coverage:
            text = f"The ceasefire remained in force continuously from {coverage[0]} through {day}."
        event = SimpleNamespace(timeline_id=day, timestamp=day, description=text)
        update = ContractUpdate(stage=stage, event_ref=day, quote=event.description,
                                covered_from=coverage[0] if coverage else None, covered_through=day if coverage else None)
        return apply_updates(spec, prior, [update], [event], {"interval_start": left, "interval_end": day})

    signed = advance({}, "signed", "2025-01-02", "2025-01-01")
    effective = advance(signed, "in_force", "2025-01-04", "2025-01-02", ("2025-01-02",))
    gap = advance(effective, "in_force", "2025-01-06", "2025-01-04")
    assert "matched_ref" not in gap
    assert "matched_ref" not in advance(gap, "in_force", "2025-01-08", "2025-01-06", ("2025-01-06",))
    assert advance(effective, "in_force", "2025-01-06", "2025-01-04", ("2025-01-04",))["matched_ref"] == "2025-01-06"


def test_semantic_judge_cannot_count_announcement_as_structured_occurrence():
    event = {"timeline_id": "sim_1_visit", "timestamp": "2025-01-02", "description": "The visit was publicly announced.",
             "origin": "simulated", "round_id": 1}
    record = {"sample_id": 1, "rounds": [{"step": 1, "timeline": [event],
              "contract_observation": {"kind": "structured_stage", "stage": "planned"}}]}
    spec = BenchmarkSimulation(round_boundaries=["2025-01-01", "2025-01-03"],
                               resolution_rules="Actual arrival is required.", outcome_deadline="2025-01-03")

    class Judge:
        async def json_call(self, **kwargs):
            return MarketJudgments(samples=[{"sample_id": 1, "answer": "yes", "refs": [event["timeline_id"]],
                                             "quote": event["description"]}])

    verdict, = asyncio.run(settle_records([record], spec, Judge(), "test", "Will the minister visit Oslo?"))
    assert verdict.category_id == "other"


def test_open_domain_compiler_binds_observable_predicate_to_original_rules(tmp_path, monkeypatch):
    from app.benchmark import runner

    rules = "Resolve Yes only when the new maritime regulation is published in the official register."
    question = "Will the maritime authority publish its new register notice?"
    default = compile_spec(question, rules, scheduled_election=False)
    assert default.route == "review"
    plan = CompiledPredicate(route="hybrid", predicate="stage", subject="maritime authority",
                             stages=["draft", "published"], terminal_stage="published", rule_quote=rules)

    class Compiler:
        def __init__(self, options):
            assert options.max_model_calls == 1

        async def json_call(self, **kwargs):
            assert kwargs["schema"] is CompiledPredicate and rules in kwargs["prompt"]
            return plan

    monkeypatch.setattr(runner, "LLMClient", Compiler)
    request = BenchmarkRequest(case_id="fixture", methods=["one_step"])
    store = EnsembleStore(tmp_path)
    result = asyncio.run(compile_additional_predicate({"spec": default.model_dump()},
        {"question": question, "rules": rules}, request, LLMOptions(),
        store=store, run_id=new_run(request, store)["run_id"]))
    assert result["spec"]["terminal_stage"] == "published"
    assert result["rule_quote"] == rules
    assert plan.model_copy(update={"rule_quote": "A fabricated rule from outside the archive."}).bind(rules) is None
