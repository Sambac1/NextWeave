import asyncio
from datetime import datetime, timezone
import json
import uuid

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api import benchmark as benchmark_api
from app.benchmark import cohort, evaluation
from app.benchmark.markets import MarketSelectionRequired
from app.benchmark.runner import forecast_inputs, frozen_sources
from app.benchmark.tree_search import search_probability_tree
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import write_json

NOW = datetime(2026, 10, 25, tzinfo=timezone.utc)
MARKET = {
    "id": "42", "conditionId": "condition-42", "slug": "senate-election",
    "question": "Will the Senate election result be certified?", "description": "Yes if certified before the deadline.",
    "createdAt": "2026-09-01T00:00:00Z", "updatedAt": "2026-10-24T00:00:00Z",
    "endDate": "2026-11-10T00:00:00Z", "outcomes": '["No","Yes"]', "outcomePrices": '["0.7","0.3"]',
    "closed": False, "events": [{"slug": "senate-election-event", "category": "Politics"}],
}
RESOLVED = MARKET | {"closed": True, "outcomePrices": '["0","1"]', "umaResolutionStatus": "resolved",
                     "closedTime": "2026-11-10T10:00:00Z", "updatedAt": "2026-11-10T10:00:00Z"}


def test_comparison_reads_archive_without_accessing_working_stores(tmp_path, monkeypatch):
    archive = tmp_path / "comparison"
    monkeypatch.setattr(benchmark_api, "COMPARISON_ARCHIVE", archive)

    def runtime_store_called():
        pytest.fail("Comparison must not read the working evaluation or run stores")

    monkeypatch.setattr(benchmark_api, "stores", runtime_store_called)
    monkeypatch.setattr(benchmark_api, "read_evaluation", runtime_store_called)
    app = FastAPI()
    app.include_router(benchmark_api.router)
    with TestClient(app) as client:
        for kind in ("evaluations", "runs"):
            identifier = str(uuid.uuid4())
            archived = {"run_id": identifier, "results": ["saved"]}
            write_json(archive / kind / f"{identifier}.json", archived)
            endpoint = f"/api/benchmark/comparison/{kind}"
            assert client.get(f"{endpoint}/{identifier}").json() == archived
            assert client.get(f"{endpoint}/{uuid.uuid4()}").status_code == 404
            assert client.get(f"{endpoint}/invalid").status_code == 422


def test_historical_discovery_uses_only_old_revision_and_separates_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(cohort, "_now", lambda: NOW)
    write_json(tmp_path / "daily/2026-10-25.json", {"observed_at": NOW.isoformat(), "markets": [MARKET]})
    seen = []

    def respond(request):
        params = request.url.params
        if request.url.path == "/tags/slug/politics":
            return httpx.Response(200, json={"id": "9"})
        if request.url.path == "/markets":
            assert params["closed"] == "true" and params["tag_id"] == "9"
            return httpx.Response(200, json=[RESOLVED, RESOLVED | {"conditionId": "no-archive", "slug": "no-archive",
                                  "question": "Will the president resign?"}])
        if params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [
                {"title": "Senate election", "snippet": "The result was certified in November."},
                {"title": "Timeline of Senate election", "snippet": "New article after the cutoff"},
            ]}})
        seen.append(params["rvstart"])
        if params["titles"] == "Timeline of Senate election":
            return httpx.Response(200, json={"query": {"pages": [{"title": "Timeline of Senate election", "ns": 0}]}})
        return httpx.Response(200, json={"query": {"pages": [{
            "title": "Senate election", "ns": 0, "pageid": 4, "revisions": [{"revid": 13,
                "timestamp": "2026-10-25T00:00:00Z",
                "slots": {"main": {"content": "Election officials described the procedures for certifying the Senate election."}}}]}]}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await cohort.collect_cohort("historical", 2, tmp_path, client=client)

    cases, labels, selection = asyncio.run(run())
    assert len(cases) == 1
    case = cases[0]
    assert case["as_of"] == "2026-10-27T00:00:00+00:00"
    assert seen == [case["as_of"], case["as_of"]]
    assert labels[case["case_id"]]["observed_yes"] is True
    assert labels[case["case_id"]]["market_probability"] is None
    assert selection["excluded"][0]["reason"] == "Cutoff-valid Market Rules Unavailable"
    prompt = json.dumps(forecast_inputs(case, frozen_sources(case, "expanded")))
    assert "certified in November" not in prompt and "observed_yes" not in prompt and "outcomePrices" not in prompt
    assert [item["revision_id"] for item in case["evidence"]] == [13]


def test_live_top_30_and_monitor_never_rewrite_forecasts(tmp_path, monkeypatch):
    monkeypatch.setattr(cohort, "_now", lambda: NOW)
    store = EnsembleStore(tmp_path / "evaluations")
    cases_store, child_store = EnsembleStore(tmp_path / "cases"), EnsembleStore(tmp_path / "runs")
    monkeypatch.setattr(evaluation, "stores", lambda: (cases_store, child_store))

    async def wikipedia(*args, **kwargs):
        assert kwargs["as_of"] == NOW and kwargs["title_discovery"]
        return ([{"url": "https://en.wikipedia.org/wiki/Election", "title": "Election",
                  "revision_at": "2026-10-24T00:00:00Z", "page_id": 8, "revision_id": 12, "sha256": __import__("hashlib").sha256(
                      b"Election administrators announced certification procedures.").hexdigest(),
                  "text": "Election administrators announced certification procedures."}], [])

    monkeypatch.setattr(cohort, "discover_wikipedia", wikipedia)

    def respond(request):
        if request.url.path == "/markets":
            assert request.url.params["limit"] == "30" and request.url.params["order"] == "volume24hr"
            return httpx.Response(200, json=[MARKET, MARKET | {"id": "43", "conditionId": "not-politics",
                                  "question": "Will it rain?", "events": [{"category": "Weather"}]}])
        return httpx.Response(200, json=MARKET)

    async def collect(*args, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await cohort.collect_cohort(*args, **kwargs, client=client)

    monkeypatch.setattr(evaluation, "collect_cohort", collect)

    async def unchanged_market(*args):
        return MARKET

    monkeypatch.setattr(evaluation, "_gamma", unchanged_market)

    async def forecast(child_id, request, case, *, store):
        assert set(request.methods) == set(evaluation.METHODS.values())
        assert "observed_yes" not in case and "market_probability" not in case
        assert request.qa_samples == 3 and request.single_step_samples == 4
        assert request.sample_count == 8 and request.steps is None
        status = store.load(child_id, "status")
        status.update(status="completed", results=[
            {"method": method, "p_yes": None if method == "mcts_simulation" else .7,
             "forecast_created_at": NOW.isoformat(), "status": "completed", "calls": 1, "tokens": 10}
            for method in request.methods])
        store.save(child_id, "status", status)

    monkeypatch.setattr(evaluation, "run_benchmark", forecast)
    request = evaluation.EvaluationRequest(mode="live", model="fixture", methods=list(evaluation.METHODS),
                                          qa_samples=3, single_step_samples=4)
    run_id = "11111111-1111-4111-8111-111111111111"
    store.save(run_id, "status", {"run_id": run_id, "mode": "live", "status": "Collecting", "methods": request.methods})

    async def run():
        await evaluation.run_evaluation(run_id, request, store)
        pending = evaluation.read_evaluation(run_id, store)
        assert pending["status"] == "Awaiting Resolution" and pending["scores"] is None
        assert pending["selection"]["excluded"][0]["reason"] == "Outside Politics"
        assert pending["questions"][0]["quote"] == .3
        assert pending["method_usage"]["mcts_simulation"] == {
            "calls": {"total": 1, "count": 1},
            "tokens": {"total": 10, "count": 1},
            "elapsed_seconds": {"total": 0, "count": 0},
            "unknown_usage_requests": {"total": 0, "count": 0},
        }
        path = child_store.directory(pending["questions"][0]["run_id"]) / "status.json"
        before = path.read_bytes()

        async def resolved_market(*args):
            return RESOLVED

        monkeypatch.setattr(evaluation, "_gamma", resolved_market)
        await evaluation.refresh_settlements(run_id, store, object())
        assert path.read_bytes() == before
        result = evaluation.read_evaluation(run_id, store)
        assert result["status"] == "Resolved"
        assert result["method_usage"] == pending["method_usage"]
        assert result["scores"]["case_count"] == 1
        assert result["scores"]["metrics"]["direct_qa"]["brier"] == pytest.approx(.09)
        assert result["scores"]["metrics"]["direct_qa"]["ece"] == pytest.approx(.3)
        assert result["scores"]["metrics"]["mcts_simulation"]["brier"] == .25
        assert result["scores"]["metrics"]["mcts_simulation"]["fallback_count"] == 1

    asyncio.run(run())


def test_resolution_rejects_ambiguous_or_unverified_markets():
    assert cohort.official_outcome(RESOLVED)["observed_yes"] is True
    for change in ({"outcomePrices": '["0.5","0.5"]'}, {"umaResolutionStatus": None}, {"closedTime": None},
                   {"resolutionStatus": "cancelled"}):
        assert cohort.official_outcome(RESOLVED | change)["status"] == "Review Required"


def test_mcts_uses_residual_mass_and_not_visits_for_aggregation():
    async def expand(path, _):
        if not path:
            return [(.6, .1, "first"), (.25, .8, "second"), (.15, None, None)]
        assert path == (1,)  # First expansion follows prior mass.
        return [(.75, .2, None), (.25, None, None)]

    result = asyncio.run(search_probability_tree("root", expand, depth=2, iterations=2, fallback=.5))
    assert result["search_iterations"] == 2
    assert result["p_yes"] == pytest.approx(.6 * .75 * .2 + .25 * .8 + .3 * .5)
    assert result["unexplored_mass"] == pytest.approx(.3)
    assert result["tree"][1]["visits"] == 1
    assert sum(node["mass"] for node in result["tree"] if node["frontier"]) == pytest.approx(1)


def test_mcts_adapter_expands_world_state_with_normal_round_contract(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.benchmark import mcts
    from app.domain import ForecastRound, WorldSnapshot
    from app.domain.ensemble import BenchmarkSimulation, EnsembleRequest

    store = EnsembleStore(tmp_path / "sims")
    sim_id = "22222222-2222-4222-8222-222222222222"
    world = WorldSnapshot(event_title="Election")
    store.save(sim_id, "initial_snapshot", world.model_dump())
    store.save(sim_id, "config", {"knowledge": [], "knowledge_version": "fixture"})
    request = EnsembleRequest(event_title="Election", model="fixture", api_key="fixture", steps=2, sample_count=1,
                              decision_policy="weighted", decision_seed=7, start_date="2026-10-25",
                              outcome_definitions=[{"id": "yes", "label": "Yes", "criterion": "Certified"},
                                                   {"id": "no", "label": "No", "criterion": "Not certified"}],
                              benchmark=BenchmarkSimulation(round_boundaries=["2026-10-25", "2026-11-02", "2026-11-10"],
                                  resolution_rules="Yes if certified", outcome_deadline="2026-11-10"))
    stages = []
    monkeypatch.setattr(mcts, "attach_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(mcts, "DecisionEvidenceRetriever", lambda *args, **kwargs: None)

    async def estimates(plan, working, *args):
        stages.append((working.round_id, working.world_state.get("from_parent")))
        return {row["branch_id"]: SimpleNamespace(p_yes=.2 if working.round_id == 0 else .7) for row in plan}

    monkeypatch.setattr(mcts, "forecast_branches", estimates)

    class Round:
        def __init__(self, *args, **kwargs):
            assert kwargs["decision_seed"] == 7

        async def run(self, snapshot, interventions, prefix, *, terminal_resolver):
            async def resolve(actions, index):
                after = snapshot.model_copy(deep=True)
                after.round_id += 1
                after.snapshot_version += 1
                after.world_state["from_parent"] = True
                return after, ForecastRound(step=after.round_id, input_snapshot_version=snapshot.snapshot_version,
                                            output_snapshot_version=after.snapshot_version, state_summary="Fixture")
            return await terminal_resolver([], resolve)

    monkeypatch.setattr(mcts, "ForecastRoundOrchestrator", Round)
    result = asyncio.run(mcts.run_mcts(sim_id, request, store, 2, .5))
    assert result["status"] == "completed", result["error"]
    assert result["p_yes"] == .7 and stages == [(0, None), (1, True)]
    assert store.load(sim_id, "result")["tree"][-1]["depth"] == 2


def test_cutoff_precedes_actual_close_when_nominal_deadline_is_later(tmp_path):
    early = RESOLVED | {"endDate": "2029-01-19T23:59:00Z", "closedTime": "2026-08-08 20:27:18+00",
                        "updatedAt": "2026-08-09T16:31:55Z"}
    _, original, cutoff, _, label = next(cohort._historical_contracts([early], tmp_path))
    assert cutoff == "2026-07-25T20:27:18+00:00"
    assert label["status"] == "Resolved"
    assert original is None  # Post-close rules cannot be used as the pre-close contract snapshot.


def test_published_archive_extracts_only_market_rules_from_pinned_rows():
    from app.benchmark.historical_archive import QuestionPrefix

    name = "2026-09-13-llm.json"
    reader = QuestionPrefix(name)
    identifier = "0x" + "a" * 64
    payload = json.dumps({"question_set": name, "questions": [
        {"source": "kalshi", "question": "Other source"},
        {"source": "polymarket", "id": identifier, "question": "Will the candidate win?",
         "background": "Yes if certified.", "freeze_datetime_value": "0.4", "observed_yes": True},
        {"source": "acled", "question": "Subsequent dataset"},
    ]})
    assert not reader.feed(payload[:75])
    assert reader.feed(payload[75:])
    assert reader.rows == [{"id": identifier, "question": "Will the candidate win?", "background": "Yes if certified."}]


def test_pause_keeps_checkpoint_and_resume_reuses_run_id(tmp_path, monkeypatch):
    store = EnsembleStore(tmp_path / "evaluations")
    cases = EnsembleStore(tmp_path / "cases")
    runs = EnsembleStore(tmp_path / "runs")
    monkeypatch.setattr(evaluation, "stores", lambda: (cases, runs))
    original_collect = asyncio.Event()
    collections = []

    async def collect(*args, **kwargs):
        collections.append(1)
        original_collect.set()
        if len(collections) == 1:
            await asyncio.Future()
        return [], {}, {"accepted": 0, "excluded": []}

    monkeypatch.setattr(evaluation, "collect_cohort", collect)
    request = evaluation.EvaluationRequest(mode="historical", model="fixture", methods=["direct_qa"])

    async def run():
        started = evaluation.queue_evaluation(request, store=store)
        run_id = started["run_id"]
        await asyncio.wait_for(original_collect.wait(), 2)
        store.save(run_id, "checkpoint", {"collected": 1})
        await evaluation.pause_evaluation(run_id, store)
        paused = evaluation.read_evaluation(run_id, store)
        assert paused["status"] == "Paused"
        assert paused["scores"] is None
        assert store.load(run_id, "checkpoint") == {"collected": 1}
        assert not (store.directory(run_id) / ".running").exists()
        resumed = evaluation.queue_evaluation(request, store=store, run_id=run_id)
        assert resumed["run_id"] == run_id
        await asyncio.wait_for(evaluation._jobs[run_id], 2)
        assert evaluation.read_evaluation(run_id, store)["status"] == "Review Required"
        assert store.load(run_id, "control") == {"pause_requested": False}

    asyncio.run(run())


@pytest.mark.parametrize(("mode", "quick_test", "expected_limit"), [
    ("historical", True, 1), ("historical", False, 5), ("live", True, 1), ("live", False, 30),
])
def test_new_evaluation_uses_supported_methods_and_correct_cohort_size(tmp_path, monkeypatch, mode, quick_test, expected_limit):
    store = EnsembleStore(tmp_path / "evaluations")
    monkeypatch.setattr(evaluation, "evaluation_store", lambda: store)
    methods = ["direct_qa", "qa_ensemble", "single_trajectory_simulation", "mcts_simulation", "multi_trajectory_simulation"]
    seen = []

    async def collect(mode, limit, *args, **kwargs):
        seen.append((mode, limit))
        return [], {}, {"accepted": 0, "excluded": []}

    monkeypatch.setattr(evaluation, "collect_cohort", collect)
    request = evaluation.EvaluationRequest(mode=mode, quick_test=quick_test, question_count=5,
                                          model="fixture", api_key="fixture", methods=methods)

    async def run():
        for removed in ("single_step_simulation", "single_step_ensemble"):
            with pytest.raises(HTTPException) as rejected:
                await benchmark_api.create_evaluation(request.model_copy(update={"methods": [*methods, removed]}))
            assert rejected.value.status_code == 422
        started = await benchmark_api.create_evaluation(request)
        assert started["methods"] == methods
        assert started["quick_test"] is quick_test
        await asyncio.wait_for(evaluation._jobs[started["run_id"]], 2)
        assert seen == [(mode, expected_limit)]
        assert store.load(started["run_id"], "status")["status"] == "Review Required"

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["historical", "live"])
def test_contract_url_runs_only_the_selected_market_with_frozen_inputs(tmp_path, monkeypatch, mode):
    store = EnsembleStore(tmp_path / "evaluations")
    cases, runs = EnsembleStore(tmp_path / "cases"), EnsembleStore(tmp_path / "runs")
    monkeypatch.setattr(evaluation, "evaluation_store", lambda: store)
    monkeypatch.setattr(evaluation, "stores", lambda: (cases, runs))
    monkeypatch.setattr(cohort, "_now", lambda: NOW)
    market = MARKET | {"question": "Will the laboratory acquisition close?", "events": [{"slug": "acquisition"}]}
    settled = market | {key: RESOLVED[key] for key in ("closed", "outcomePrices", "umaResolutionStatus", "closedTime", "updatedAt")}
    cutoff = NOW if mode == "live" else datetime(2026, 10, 27, tzinfo=timezone.utc)
    write_json(store.root / "daily/2026-10-25.json", {"observed_at": NOW.isoformat(), "markets": [market]})
    url = f"https://polymarket.com/market/{market['slug']}"
    seen = []

    async def wikipedia(_client, **options):
        assert options["title_discovery"] and options["seed_urls"] == []
        assert options["as_of"] == cutoff
        text = "The laboratory published its acquisition and closing procedures."
        return ([{"url": "https://en.wikipedia.org/wiki/Laboratory", "title": "Laboratory",
                  "revision_at": "2026-10-24T00:00:00Z", "page_id": 7, "revision_id": 21,
                  "sha256": __import__("hashlib").sha256(text.encode()).hexdigest(), "text": text}], [])

    monkeypatch.setattr(cohort, "discover_wikipedia", wikipedia)

    def respond(request):
        assert request.url.path == f"/markets/slug/{market['slug']}"
        return httpx.Response(200, json=market if mode == "live" else settled)

    async def collect(*args, **kwargs):
        assert kwargs["market_url"] == url and args[1] == 1
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await cohort.collect_cohort(*args, **kwargs, client=client)

    async def verify_live(*args):
        assert mode == "live"
        return market

    async def forecast(child_id, request, case, *, store):
        seen.append(case["case_id"])
        assert case["condition_id"] == market["conditionId"]
        assert case["as_of"] == cutoff.isoformat()
        assert "market_probability" not in case and "observed_yes" not in case
        assert request.methods == ["direct_qa", "trajectory_sampling"]
        status = store.load(child_id, "status")
        status.update(status="completed", results=[
            {"method": method, "p_yes": .6, "status": "completed", "forecast_created_at": NOW.isoformat()}
            for method in request.methods
        ])
        store.save(child_id, "status", status)

    monkeypatch.setattr(evaluation, "collect_cohort", collect)
    monkeypatch.setattr(evaluation, "_gamma", verify_live)
    monkeypatch.setattr(evaluation, "run_benchmark", forecast)

    async def run():
        request = evaluation.EvaluationRequest(mode=mode, model="fixture", api_key="fixture", market_url=url,
                                              methods=["direct_qa", "multi_trajectory_simulation"])
        started = await benchmark_api.create_evaluation(request)
        run_id = started["run_id"]
        assert started["market_url"] == url
        await asyncio.wait_for(evaluation._jobs[run_id], 3)
        report = evaluation.read_evaluation(run_id, store)
        assert len(seen) == len(report["questions"]) == report["selection"]["accepted"] == 1
        assert report["questions"][0]["forecasts"]["multi_trajectory_simulation"] == .6
        if mode == "live":
            assert report["status"] == "Awaiting Resolution"
            assert report["questions"][0]["quote"] == .3 and report["scores"] is None
        else:
            assert report["status"] == "Completed"
            assert report["questions"][0]["observed_yes"] is True and report["scores"]["case_count"] == 1

    asyncio.run(run())


def test_contract_url_returns_submarket_choices_and_rejects_foreign_hosts(tmp_path, monkeypatch):
    store = EnsembleStore(tmp_path / "evaluations")
    monkeypatch.setattr(evaluation, "evaluation_store", lambda: store)
    choices = [{"slug": "one", "question": "First contract", "url": "https://polymarket.com/event/event?marketSlug=one"}]

    async def collect(*args, **kwargs):
        raise MarketSelectionRequired(choices)

    monkeypatch.setattr(evaluation, "collect_cohort", collect)

    async def run():
        request = evaluation.EvaluationRequest(mode="live", model="fixture", api_key="fixture",
                                              market_url="https://polymarket.com/event/event", methods=["direct_qa"])
        with pytest.raises(HTTPException) as rejected:
            await benchmark_api.create_evaluation(request.model_copy(update={"market_url": "https://example.com/event/event"}))
        assert rejected.value.status_code == 422
        started = await benchmark_api.create_evaluation(request)
        run_id = started["run_id"]
        await asyncio.wait_for(evaluation._jobs[run_id], 2)
        status = store.load(run_id, "status")
        assert status["status"] == "Review Required" and status["market_choices"] == choices
        assert store.load(run_id, "contract_runs") is None

    asyncio.run(run())
