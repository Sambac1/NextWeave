from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import events, experiments, knowledge
from app.domain import ForecastRound, TimelineEvent
from app.domain.decision_core import DecisionCore, DecisionGoal, DecisionItem, SourceSupport
from app.knowledge import extraction
from app.knowledge.timeline_agent import TimelineAgentOutcome
from app.llm.client import LLMError
from app.settings import get_settings
from app.simulation.ensemble import run_experiment
from app.simulation.outlook import PublicEvidence, verified_condition
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository, GraphRepository, SnapshotRepository
from test_benchmark_sources import EVENT, MARKET, page
from test_ensemble import actor

AUTHORITY = "The laboratory board has authority to approve the acquisition and publish the closing notice."
GOAL = "The laboratory board aims to secure regulatory approval before closing the acquisition."


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    repository = EventRepository()
    monkeypatch.setattr(events, "events", repository)
    monkeypatch.setattr(events, "graphs", GraphRepository(repository))
    monkeypatch.setattr(events, "snapshots", SnapshotRepository(repository))
    monkeypatch.setattr(knowledge, "events", repository)
    app = FastAPI()
    for router in (events.router, experiments.router, knowledge.router):
        app.include_router(router)
    try:
        with TestClient(app) as client:
            yield client, repository
    finally:
        get_settings.cache_clear()


def test_market_preview_selects_contract_before_search(workspace, monkeypatch):
    client, _ = workspace
    calls = []
    expected_deadline = datetime.now(timezone.utc) + timedelta(days=30)
    east = expected_deadline.strftime("%B %d, %Y")
    market = MARKET | {
        "description": f"This market resolves to Yes if the laboratory acquisition closes by {east}, 6:00 PM ET.",
    }

    def handler(request):
        calls.append(request.url.host)
        assert request.url.host == "gamma-api.polymarket.com"
        return httpx.Response(200, json={"markets": [market, MARKET | {"slug": "another-market"}]})

    monkeypatch.setattr(events, "forecast_http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    selection = client.post("/api/events/forecast-preview", json={"query": EVENT}).json()
    assert selection["status"] == "select_contract"
    specific = client.post("/api/events/forecast-preview", json={"query": selection["choices"][0]["url"]})
    assert specific.status_code == 200
    data = specific.json()
    assert data["spec"]["kind"] == "market"
    assert data["spec"]["deadline_basis"] == "explicit_contract_deadline"
    assert data["spec"]["rules"] == market["description"]
    assert "outcomePrices" not in specific.text
    changed = client.post("/api/events/forecast", json={
        "preview_id": data["preview_id"], "outcome_deadline": (expected_deadline + timedelta(days=1)).isoformat(),
    })
    assert changed.status_code == 422 and "explicit contract rule" in changed.json()["detail"]
    assert calls == ["gamma-api.polymarket.com"] * 2


@pytest.mark.parametrize("fields", [{}, {"yes_criterion": "  The board publishes a signed notice.  ", "no_criterion": " "},
                                    {"yes_criterion": None, "no_criterion": "The public register shows no closing."}])
def test_question_confirmation_fills_only_blank_criteria(workspace, monkeypatch, fields):
    client, repository = workspace
    titles = ["Laboratory acquisition", "Laboratory acquisition approval"]

    def handler(request):
        assert request.url.host == "en.wikipedia.org"
        params = request.url.params
        if params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [{"title": title} for title in titles]}})
        title = params["titles"]
        return httpx.Response(200, json=page(title, titles.index(title) + 1, f"{AUTHORITY} {GOAL}"))

    monkeypatch.setattr(events, "forecast_http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    question = "Will the laboratory acquisition close?"
    preview = client.post("/api/events/forecast-preview", json={"query": question}).json()
    deadline = datetime.fromisoformat(preview["spec"]["as_of"]) + timedelta(days=30)
    request = {"preview_id": preview["preview_id"], "outcome_deadline": deadline.isoformat(), **fields}
    response = client.post("/api/events/forecast", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    spec = result["forecast_spec"]
    for name, opposite in [("yes_criterion", "No"), ("no_criterion", "Yes")]:
        supplied = (fields.get(name) or "").strip()
        if supplied:
            assert spec[name] == supplied
        else:
            assert "full forecast window" in spec[name]
            assert (f"stated {opposite} condition" if fields else question) in spec[name]
    assert spec["yes_criterion"] != spec["no_criterion"]
    assert datetime.fromisoformat(spec["outcome_deadline"]) == deadline
    assert repository.load_artifact(result["event_id"], "forecast_spec", {}) == spec
    assert client.post("/api/events/forecast", json=request).json()["event_id"] == result["event_id"]


def test_automatic_criteria_need_an_observable_binary_target():
    from app.application.forecast_task import question_criteria

    with pytest.raises(ValueError, match="Use a yes/no question"):
        question_criteria("How will the negotiations unfold?")
    with pytest.raises(ValueError, match="Enter distinct conditions"):
        question_criteria("Will the talks end?", "Talks end.", "talks  end.")


def test_main_forecast_accepts_fixed_time_steps_and_keeps_the_deadline(workspace, monkeypatch):
    client, repository = workspace
    question = "Will the laboratory acquisition close?"
    url = "https://en.wikipedia.org/wiki/Laboratory_acquisition"
    event_id = repository.create(question, [url], None)["event_id"]
    repository.save_sources(event_id, [{"url": url, "title": "Laboratory acquisition", "content": AUTHORITY}])
    spec = {"kind": "question", "question": question, "as_of": "2026-11-10T12:00:00Z",
            "outcome_deadline": "2026-11-20T16:00:00Z",
            "yes_criterion": "The signed closing notice is published.",
            "no_criterion": "A complete register shows the acquisition did not close."}
    repository.save_artifact(event_id, "forecast_spec", spec)
    repository.save_artifact(event_id, "scheduled_milestones", [{"date": "2026-11-17"}])

    async def queue(request):
        return {"steps": request.steps, "protocol": request.question_simulation.model_dump()}

    monkeypatch.setattr(experiments, "_queue_experiment", queue)
    endpoint = "/api/experiments/from-question"
    auto = client.post(endpoint, json={"event_id": event_id})
    assert auto.status_code == 202 and auto.json()["steps"] == 3
    fixed = client.post(endpoint, json={"event_id": event_id, "steps": 5})
    assert fixed.status_code == 202 and fixed.json()["steps"] == 5
    protocol = fixed.json()["protocol"]
    assert protocol["round_boundaries"][0] == "2026-11-10"
    assert protocol["round_boundaries"][-1] == "2026-11-20"
    assert "2026-11-17" in protocol["round_boundaries"][1:-1]
    assert datetime.fromisoformat(protocol["outcome_deadline"]) == datetime.fromisoformat(spec["outcome_deadline"])
    assert client.post(endpoint, json={"event_id": event_id, "steps": 13}).status_code == 422

    repository.save_artifact(event_id, "forecast_spec", spec | {"outcome_deadline": "2026-11-10T16:00:00Z"})
    assert client.post(endpoint, json={"event_id": event_id, "steps": 1}).json()["steps"] == 1
    invalid = client.post(endpoint, json={"event_id": event_id, "steps": 2})
    assert invalid.status_code == 422 and "at most 1 time step" in invalid.json()["detail"]


def test_question_api_keeps_undated_evidence_and_resumes_only_readout(workspace, monkeypatch):
    client, repository = workspace
    revision_requests = []
    searched = []

    def handler(request):
        assert request.url.host == "en.wikipedia.org"
        params = request.url.params
        if params.get("list") == "search":
            searched.append(params["srsearch"])
            titles = ["Laboratory acquisition"] if len(searched) == 1 else ["Laboratory acquisition", "Laboratory acquisition approval"]
            return httpx.Response(200, json={"query": {"search": [{"title": title} for title in titles]}})
        revision_requests.append((params["titles"], params["rvstart"]))
        if params["titles"] == "Laboratory acquisition approval":
            return httpx.Response(200, json=page("Laboratory acquisition approval", 2, GOAL))
        return httpx.Response(200, json=page("Laboratory acquisition", 1,
                                           AUTHORITY + " " + GOAL + " Polymarket assigned a 75% probability to closing."))

    monkeypatch.setattr(events, "forecast_http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    preview = client.post("/api/events/forecast-preview", json={"query": "Will the laboratory acquisition close?"}).json()
    assert preview["status"] == "confirm" and not revision_requests
    assert client.post("/api/events/forecast", json={"preview_id": preview["preview_id"]}).status_code == 422
    as_of = datetime.fromisoformat(preview["spec"]["as_of"])
    deadline = as_of + timedelta(minutes=30)
    confirm = {"preview_id": preview["preview_id"], "outcome_deadline": deadline.isoformat(),
               "yes_criterion": "the board publishes the signed closing notice.",
               "no_criterion": "the official closing register establishes no closing by the deadline."}
    created = client.post("/api/events/forecast", json=confirm)
    assert created.status_code == 200, created.text
    event_id = created.json()["event_id"]
    assert client.post("/api/events/forecast", json=confirm).json()["event_id"] == event_id
    assert len(searched) == 4
    assert all("mandate authority" not in query for query in searched)
    assert [title for title, _ in revision_requests] == ["Laboratory acquisition", "Laboratory acquisition approval"]
    assert all(datetime.fromisoformat(cutoff) == as_of for _, cutoff in revision_requests)
    sources = repository.load_sources(event_id)
    assert {row["revision_id"] for row in sources} == {101, 102}
    source = next(row for row in sources if row["revision_id"] == 101)
    assert source["revision_url"].endswith("oldid=101")
    assert AUTHORITY in source["content"] and "75%" not in source["content"]

    async def empty_coverage(*args, **kwargs):
        return TimelineAgentOutcome(applied=True, reason="Undated institutional source", historical=[], stats={}, artifact={})

    async def grounded_profile(*, request, timeline, events):
        assert not timeline and request.evidence_cutoff == as_of.date().isoformat()
        mandate = DecisionItem(description=AUTHORITY, provenance="source_statement", claim_scope="structural",
                               source_refs=[SourceSupport(evidence_id="public_fixture", quote=AUTHORITY)])
        goal = DecisionGoal(description=GOAL, provenance="source_statement",
                            source_refs=[SourceSupport(evidence_id="public_goal", quote=GOAL)])
        grounded = actor().model_copy(update={"decision_core": DecisionCore(mandate=[mandate], goals=[goal])})
        participant = actor().model_copy(update={"agent_id": "participant", "name": "Evidence Board",
                                                  "decision_core": DecisionCore(mandate=[mandate])})
        return {"agents": [grounded, participant], "name_mapping": {}, "quality": {"key_figures_selected": 2}}

    monkeypatch.setattr(extraction, "extract_coverage", empty_coverage)
    monkeypatch.setattr(extraction, "run_layer_stage", grounded_profile)
    from app.knowledge import operations

    brief_calls = []

    async def generate_brief(request, *, events):
        brief_calls.append(request.event_id)
        return "The board has authority to approve the acquisition."

    monkeypatch.setattr(operations, "combine_summary", generate_brief)
    prep = client.post("/api/llm/figures-pipeline", json={
        "event_id": event_id, "event_title": "Client title override", "api_key": "fixture", "model": "fixture",
    })
    assert prep.status_code == 200, prep.text
    saved = client.get(f"/api/llm/figures-pipeline/{event_id}/result")
    assert saved.status_code == 200, saved.text
    assert saved.json()["timeline"] == []
    assert [row["forecast_role"] for row in saved.json()["figures"]] == ["decision_actor", "evidence_participant"]
    assert brief_calls == [event_id]

    rounds, estimates = [], []
    observed = "The laboratory board published the signed closing notice for the acquisition."

    async def settle(snapshot, interventions, prefix):
        number = int(prefix.split("_sample_")[1].split("_")[0])
        rounds.append((number, snapshot.round_id))
        assert [a.agent_id for a in snapshot.agents] == ["actor_a"]
        version = snapshot.snapshot_version
        snapshot.round_id += 1
        snapshot.snapshot_version += 1
        clock = snapshot.world_state["simulation_clock"]
        event = TimelineEvent(timeline_id="result", description=observed, timestamp=clock["interval_end_utc"],
                              origin="simulated", round_id=snapshot.round_id, confidence=.9, importance=.8)
        snapshot.timeline.append(event)
        return snapshot, ForecastRound(step=snapshot.round_id, input_snapshot_version=version, output_snapshot_version=version+1,
                                       timeline=[event.model_copy(deep=True)], state_summary=observed,
                                       decision_sampling={"audited_silent_round": 1})

    class Runtime:
        recorder = None
        available = False

        async def json_call(self, *, schema, prompt, **kwargs):
            import json

            assert schema.__name__ == "DeadlineAudit"
            data = json.loads(prompt)
            estimates.append(data["window"])
            if not self.available:
                raise LLMError("Fixture provider unavailable")
            key = next(key for key, row in data["settled_public_events"].items() if row["text"] == observed)
            return schema(verdict="yes", basis="The board published the signed closing notice.",
                          evidence_ref=key, evidence_quote=observed)

    runtime = Runtime()

    async def run(run_id, request):
        await run_experiment(run_id, request, runtime=runtime, round_runner=settle)

    monkeypatch.setattr(experiments, "run_experiment", run)

    def wait(run_id, expected):
        import time

        for _ in range(100):
            value = client.get(f"/api/experiments/{run_id}").json()
            if value.get("status") == expected:
                return value
            if value.get("status") == "failed":
                pytest.fail(str(value.get("error")))
            time.sleep(.02)
        pytest.fail(str(value))

    started = client.post("/api/experiments/from-question", json={
        "event_id": event_id, "sample_count": 2, "api_key": "fixture", "model": "fixture",
    })
    assert started.status_code == 202, started.text
    run_id = started.json()["run_id"]
    assert started.json()["steps"] == 1
    first = wait(run_id, "partial")["report"]
    assert first["p_yes"] is None and first["resume_available"]
    assert first["bounds"] == [0, 1]
    assert len(rounds) == 2

    runtime.available = True
    resumed = client.post("/api/experiments", json={"event_title": "Case", "resume_run_id": run_id, "api_key": "fixture", "model": "fixture"})
    assert resumed.status_code == 202, resumed.text
    completed = wait(run_id, "completed")
    assert completed["event_id"] == event_id and completed["question_forecast"]
    assert client.get("/api/events/context", params={"event_id": event_id}).json()["event_title"] == preview["spec"]["question"]
    report = completed["report"]
    assert report["status"] == "complete" and report["p_yes"] == 1
    assert report["algorithm"] == "conditional-stratified-v2"
    assert report["probability_mass"] == {"yes": 1, "no": 0, "unresolved": 0}
    assert report["estimated_trajectories"] == 2 and report["knowledge_version"]
    assert {row["revision_id"] for row in report["sources"]} == {101, 102}
    assert all(datetime.fromisoformat(row["rounds"][-1]["interval"]["interval_end_utc"]) == deadline for row in report["trajectories"])
    assert len(rounds) == 2 and all(estimates)
    persisted = EnsembleStore().load(run_id, "config")["request"]
    assert "api_key" not in persisted
    assert client.post("/api/experiments", json=persisted).status_code == 422


@pytest.mark.parametrize("condition_id", ["target_event", "negative_outcome"])
def test_verified_outcome_requires_public_observation_within_the_window(condition_id):
    from app.domain import WorldSnapshot

    quote = "The official register confirmed no acquisition closed by the deadline."
    snapshot = WorldSnapshot(event_title="Forecast", world_state={
        "benchmark_contract": {"forecast_criteria": {"no": quote}, "as_of": "2026-01-01T12:00:00Z"},
        "question_states": {condition_id: {"status": "met", "event_refs": ["obs"], "evidence_quote": quote}},
    })
    clock = {"step": 1, "horizon_steps": 1, "interval_end_utc": "2026-01-01T14:00:00Z"}
    condition = PublicEvidence(f"condition:{condition_id}", "simulated_condition", quote)
    catalog = {condition.ref: condition}
    assert verified_condition(snapshot, catalog, clock, condition_id) is None
    catalog["obs"] = PublicEvidence("obs", "simulated", quote, "2026-01-01T15:00:00Z")
    assert verified_condition(snapshot, catalog, clock, condition_id) is None
    catalog["obs"] = PublicEvidence("obs", "simulated", quote, "2026-01-01T14:00:00Z")
    assert verified_condition(snapshot, catalog, clock, condition_id) == condition
    assert verified_condition(snapshot, catalog, clock | {"horizon_steps": 2}, condition_id) is None


def test_question_packet_prefers_named_scope_and_retains_dated_schedule():
    from app.application.forecast_task import source_milestones
    from app.domain.forecast_task import ForecastSpec
    from app.knowledge.forecast_evidence import frozen_sources
    from app.simulation.question_schedule import simulation_round_plan

    question = "Will the Bank of Canada change its policy interest rate?"
    planned = "The Bank of Canada policy meeting is scheduled for 2027-01-14."
    spec = ForecastSpec(kind="question", question=question,
                        as_of="2027-01-01T12:00:00Z", outcome_deadline="2027-01-31T18:00:00Z",
                        yes_criterion="a policy rate change is announced.", no_criterion="the policy rate is held.")
    pages = [{"url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}", "title": title,
              "text": body, "revision_id": index, "revision_at": "2026-12-31T12:00:00Z",
              "discovered_from": "wiki_search" if index == 1 else "https://en.wikipedia.org/w/index.php?oldid=100"}
             for index, (title, body) in enumerate([
                 ("Bank of Canada", f"The Bank of Canada sets its policy interest rate independently. {planned}"),
                 ("European Central Bank", "The European Central Bank sets its policy interest rate independently."),
             ], 1)]
    case = spec.contract_case()
    sources = frozen_sources(case | {"evidence": pages}, "expanded", require_relevant=True)
    assert [row["title"] for row in sources] == ["Bank of Canada"]
    milestones = source_milestones(spec, sources)
    assert milestones[0]["quote"] == planned
    plan = simulation_round_plan(case, "trajectory_sampling", None, {"source_milestones": milestones})
    assert "2027-01-14" in plan["boundaries"]
