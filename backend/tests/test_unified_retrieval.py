"""Retrieval quality, frozen-release safety and real round integration (no API)."""
import asyncio
from copy import deepcopy
import json

import pytest

from app.domain import ActionProposal, TimelineEvent, WorldSnapshot
from app.knowledge.okf import KnowledgeStore, digest
from app.knowledge.retrieval import EvidenceIndex
from app.simulation.action_audit import _program_violations
from app.simulation.context import actor_context, compact_json, information_context, public_context
from app.simulation.ensemble import attach_evidence, prepare_experiment, run_experiment
from app.simulation.evidence import DecisionEvidenceRetriever, declared_documents, evidence_card
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository
from test_ensemble import actor, request, runner, Classifier
from test_okf import corpus


def document(identifier, text, *, timestamp="2020-01", actors=(), status="supported", **metadata):
    return {"id": identifier, "kind": "timeline", "path": identifier + ".md", "body": text,
            "metadata": {"title": text, "description": text, "timestamp": timestamp,
                         "occurred_month": timestamp[:7], "actor_ids": list(actors),
                         "origin": "historical", "date_validation": {"status": status}, **metadata}}


def snapshot_for(documents):
    return WorldSnapshot(event_title="Supply", agents=[actor()], timeline=[TimelineEvent(
        timeline_id=d["id"], timestamp=d["metadata"]["timestamp"],
        description=d["metadata"]["description"], actors=d["metadata"].get("actor_ids", []),
        source_sections=d["metadata"].get("source_sections", []),
        evidence_text="An existing source statement.", confidence=.5, importance=.5)
        for d in documents if d.get("kind") == "timeline"])


def test_multilingual_queries_match_terms_without_requiring_whole_sentence():
    index = EvidenceIndex([document("a", "The laboratory tested alternative sources of supply."),
                           document("b", "实验室探索替代方案，保证供应稳定。"),
                           document("c", "A sporting event.")])
    assert index.search("alternative supply", kind="timeline")[0]["id"] == "a"
    assert index.search("供应替代", kind="timeline")[0]["id"] == "b"
    assert not index.search("unseenkeyword", kind="timeline")


def test_page_and_simulation_share_the_same_frozen_index_and_rebuild(tmp_path):
    events, case_id = corpus(tmp_path)
    store = KnowledgeStore(events)
    first = store.publish(case_id)
    version, docs = store.snapshot(case_id, version=first["version"])
    frozen = EvidenceIndex(docs, version=version)
    expected = frozen.search("供应试验", kind="timeline")
    assert expected == store.search(case_id, "供应试验", kind="timeline", version=version)
    assert any(d["kind"] == "section" for d in docs)
    # Publishing an updated corpus does not change the pinned result.
    events.save_sources(case_id, [{"url": "https://example.org/new", "content": "New source text."}])
    second = store.publish(case_id)
    assert second["version"] != version
    store.database.unlink()
    assert store.search(case_id, "供应试验", kind="timeline", version=version) == expected
    assert frozen.search("供应试验", kind="timeline") == expected
    with pytest.raises(ValueError, match="belong"):
        store.snapshot(case_id, version="a" * 24)


def test_dates_unknown_and_future_events_do_not_enter_decision_evidence():
    docs = [document("old", "Supply agreement", timestamp="2020-01-05"),
            document("future", "Supply agreement", timestamp="2021-01-05"),
            document("unverified", "Supply agreement", status="needs_review"),
            document("month", "Supply agreement", timestamp="2020-01")]
    snapshot = snapshot_for(docs)
    retrieval = DecisionEvidenceRetriever(snapshot, docs, cutoff_date="2020-01-15")
    evidence, audit = retrieval.retrieve(snapshot, snapshot.agents[0], "Supply", 6)
    assert [e["timeline_id"] for e in evidence] == ["old"]
    assert audit["model_calls"] == 0
    working = snapshot.model_copy(deep=True)
    attach_evidence(working, snapshot, docs, "Supply", retriever=retrieval)
    assert {e["timeline_id"] for e in public_context(working, "decision", working.agents[0])["timeline"]} == {"old"}
    # Even an already-cited belief cannot override the date boundary.
    from app.domain.agents import SituationView
    snapshot.agents[0].beliefs.situation_views = [SituationView(
        topic="Supply", assessment="Future claim", confidence=.5, timeline_refs=["future"])]
    with pytest.raises(ValueError, match="boundary"):
        retrieval.retrieve(snapshot, snapshot.agents[0], "Supply")


def test_chapter_matches_recall_linked_validated_events_not_undated_claims():
    docs = [document("old", "An agreement was signed.", source_sections=["source_01/Terms"]),
            document("bad", "A proposed agreement.", status="needs_review", source_sections=["source_01/Terms"]),
            {"id": "section", "kind": "section", "path": "section.md",
             "body": "A uniquekeyword condition. UNDATED_UNSUPPORTED_CLAIM.",
             "metadata": {"title": "Terms", "section_path": "source_01/Terms"}}]
    snapshot = snapshot_for(docs)
    evidence, _ = DecisionEvidenceRetriever(snapshot, docs).retrieve(snapshot, snapshot.agents[0], "uniquekeyword")
    assert [e["timeline_id"] for e in evidence] == ["old"]
    assert "section_link" in evidence[0]["retrieval_reasons"]
    assert "UNDATED_UNSUPPORTED_CLAIM" not in json.dumps(evidence)


def test_frozen_original_passage_reaches_actor_without_timeline_and_respects_exact_cutoff(tmp_path):
    from app.simulation.context import cited_public_context

    events = EventRepository(tmp_path / "events")
    event_id = events.create("Supply", [], None)["event_id"]
    original = "== Supply terms ==\nThe laboratory may request an exception to the supply agreement."
    pages = [
        {"url": "https://en.wikipedia.org/wiki/Supply", "title": "Supply",
         "content": original, "revision_at": "2025-01-01T11:59:59+00:00", "revision_id": 10},
        {"url": "https://en.wikipedia.org/wiki/Supply_Later", "title": "Later",
         "content": "The laboratory supply agreement authorizes a FUTURE_DETAIL.",
         "revision_at": "2025-01-01T12:00:01+00:00", "revision_id": 11},
        {"url": "https://en.wikipedia.org/wiki/Unknown_Revision", "title": "Unknown",
         "content": "The laboratory supply agreement authorizes an UNKNOWN_REVISION.",
         "retrieved_at": "2025-01-01T11:00:00+00:00"},
    ]
    events.save_sources(event_id, pages)
    knowledge = KnowledgeStore(events)
    release = knowledge.publish(event_id)
    _, docs = knowledge.snapshot(event_id, version=release["version"])
    frozen = snapshot_for(docs)
    working = frozen.model_copy(deep=True)
    retrieve = DecisionEvidenceRetriever(frozen, docs, cutoff_date="2025-01-01",
                                         cutoff_at="2025-01-01T12:00:00+00:00")
    attach_evidence(working, frozen, docs, "laboratory supply agreement", retriever=retrieve)
    a = working.agents[0]
    decision = public_context(working, "decision", a)
    assert not decision["timeline"]
    passage, = decision["source_passages"]
    assert passage["revision_id"] == 10 and passage["available_at"] == pages[0]["revision_at"]
    assert original[passage["span_start"]:passage["span_end"]] == passage["quote"]
    assert information_context(working, a)["allowed_timeline_refs"] == []
    assert "source_passages" not in public_context(working, "activation")
    proposal = ActionProposal(agent_id=a.agent_id, agent_name=a.name, round_id=1, snapshot_version=0,
        action_id="a1", decision="wait", action_type="no_action", action="Wait", reasoning="Wait", confidence=.5)
    settlement = cited_public_context(working, "settlement", [proposal])
    assert settlement["source_passages"] == [passage]
    assert "queries" not in settlement["source_passages"][0]


def test_goals_constraints_and_counterpart_context_have_retrieval_opportunities():
    docs = [document(f"filler{i}", "Laboratory supply update", actors=["actor_a"]) for i in range(20)]
    docs += [document("constraint", "An exclusive contract restricts purchases", actors=["actor_a"]),
             document("counterpart", "Another supplier offers alternative supply", actors=["actor_b"]),
             document("goal", "A research expansion agreement", actors=["actor_a"])]
    snapshot = snapshot_for(docs)
    from app.domain.agents import Desire
    a = snapshot.agents[0]
    a.desires = [Desire(desire_id="g", description="research expansion", priority=1)]
    a.constraints = ["exclusive contract restricts purchases"]
    evidence, audit = DecisionEvidenceRetriever(snapshot, docs).retrieve(snapshot, a, "supply", 6)
    ids = {e["timeline_id"] for e in evidence}
    assert {"constraint", "counterpart", "goal"} <= ids
    assert len(ids) <= 6 and audit["queries"]["objectives"] == "research expansion"


def test_retrieved_sources_reach_prompts_and_other_actors_do_not_receive_private_queries():
    proof = {"source_id": "source", "snapshot_sha256": "frozen", "url": "https://example.org/source",
             "section_paths": ["source_01/Terms"], "span_start": 10, "span_end": 50,
             "quote": "The contract restricts purchases.", "attribution": "exact_span"}
    docs = [document("constraint", "Contract restrictions", actors=["actor_a"], evidence_refs=[proof])]
    docs += [document(f"filler{i}", "Unrelated update") for i in range(20)]
    snapshot = snapshot_for(docs)
    from app.domain.agents import Desire
    snapshot.agents[0].desires = [Desire(desire_id="g", description="PRIVATE_GOAL_A", priority=1)]
    other = actor().model_copy(update={"agent_id": "actor_b", "name": "Other actor"})
    snapshot.agents.append(other)
    attach_evidence(snapshot, snapshot.model_copy(deep=True), docs, "contract restrictions", 3)
    decision = public_context(snapshot, "decision", snapshot.agents[0])
    assert "constraint" in {e["timeline_id"] for e in decision["timeline"]}
    assert decision["evidence_bundle"][0]["sources"][0]["quote"] == proof["quote"]
    assert "constraint" in information_context(snapshot, snapshot.agents[0])["allowed_timeline_refs"]
    other_context = compact_json(public_context(snapshot, "decision", other)) + compact_json(actor_context(other))
    assert "PRIVATE_GOAL_A" not in other_context
    assert "PRIVATE_GOAL_A" in json.dumps(snapshot.agents[0].retrieval_context)


def test_frozen_index_is_not_mutated_by_trajectory_or_returned_results():
    docs = [document("old", "Supply evidence", actors=["actor_a"])]
    baseline = snapshot_for(docs)
    retriever = DecisionEvidenceRetriever(baseline, docs, version="fixed")
    before = deepcopy(docs)
    one, two = baseline.model_copy(deep=True), baseline.model_copy(deep=True)
    one.timeline.append(TimelineEvent(timeline_id="sim_other", timestamp="2020-02", description="SECRET_OTHER_PATH",
                                      origin="simulated", confidence=.5, importance=.5))
    attach_evidence(one, baseline, docs, "supply", retriever=retriever)
    one.agents[0].evidence_context[0]["description"] = "MUTATED"
    attach_evidence(two, baseline, docs, "supply", retriever=retriever)
    assert two.agents[0].evidence_context[0]["description"] == "Supply evidence"
    assert "SECRET_OTHER_PATH" not in compact_json(two.model_dump())
    assert docs == before


def test_source_excerpt_is_not_cut_in_the_middle_of_a_sentence():
    quote = "A complete first sentence. " + "x" * 900 + "."
    card = evidence_card({"timeline_id": "e", "evidence": [{"quote": quote}]})
    assert card["sources"][0]["quote"] == "A complete first sentence."
    assert card["sources"][0]["excerpt_shortened"]
    long_only = evidence_card({"timeline_id": "e", "evidence": [{"quote": "x" * 900 + "."}]})
    assert long_only["sources"][0]["excerpt_omitted"]


def test_citations_to_known_but_unsupplied_events_are_rejected():
    docs = [document(f"e{i}", "Unrelated update") for i in range(30)]
    docs.append(document("chosen", "Supply condition", actors=["actor_a"]))
    snapshot = snapshot_for(docs)
    attach_evidence(snapshot, snapshot.model_copy(deep=True), docs, "supply", 3)
    visible = {e["timeline_id"] for e in public_context(snapshot, "decision", snapshot.agents[0])["timeline"]}
    hidden = next(e.timeline_id for e in snapshot.timeline if e.timeline_id not in visible)
    proposal = ActionProposal(agent_id="actor_a", agent_name="Laboratory", round_id=1, snapshot_version=0,
        action_id="action_1_actor_a", decision="wait", action_type="no_action", action="Wait", reasoning="Wait",
        confidence=.5, timeline_refs=[hidden])
    assert "REFERENCE_NOT_IN_SUPPLIED_EVIDENCE" in _program_violations(proposal, snapshot.agents[0], snapshot)


def test_rounds_reuse_one_frozen_index_and_persist_per_path_audit(tmp_path, monkeypatch):
    events, store = EventRepository(tmp_path / "events"), EnsembleStore(tmp_path / "runs")
    req = request(recent_timeline=[{"timeline_id": "history", "timestamp": "2020-01", "description": "Supply trial",
                                  "actors": ["actor_a"], "confidence": .5, "importance": .5}])
    rid, req = prepare_experiment(req, events=events, store=store)
    config = store.load(rid, "config")
    assert config["retrieval"]["corpus_sha256"] == digest(config["knowledge"])
    built = []
    original = EvidenceIndex.__init__

    def record(self, *args, **kwargs):
        built.append(True)
        original(self, *args, **kwargs)

    monkeypatch.setattr(EvidenceIndex, "__init__", record)
    result = asyncio.run(run_experiment(rid, req, store=store, runtime=Classifier(), round_runner=runner([])))
    assert result["completed_samples"] == req.sample_count and len(built) == 1
    for number in range(1, req.sample_count + 1):
        audit = store.load(rid, f"retrieval/{number:04d}/001")
        assert audit["actor_a"]["selected_ids"] == ["history"]
        assert audit["actor_a"]["model_calls"] == 0


def test_changed_frozen_documents_cannot_silently_resume(tmp_path):
    events, store = EventRepository(tmp_path / "events"), EnsembleStore(tmp_path / "runs")
    rid, req = prepare_experiment(request(), events=events, store=store)
    config = store.load(rid, "config")
    config["knowledge"][0]["metadata"]["description"] = "Changed"
    store.save(rid, "config", config)
    with pytest.raises(ValueError, match="knowledge documents changed"):
        asyncio.run(run_experiment(rid, req, store=store, runtime=Classifier(), round_runner=runner([])))


def test_short_actor_alias_does_not_match_inside_an_unrelated_word():
    docs = [document("e", "Unrelated event", actors=["us"]),
            {"id": "us", "kind": "actor", "body": "", "metadata": {"title": "US", "actor_id": "us"}}]
    assert not EvidenceIndex(docs).search("industry", kind="timeline")


def test_settlement_keeps_action_evidence_even_outside_recent_context():
    from app.simulation.context import cited_public_context
    docs = [document(f"e{i}", "Unrelated update") for i in range(30)]
    docs[0] = document("old", "Supply obligation", evidence_refs=[{
        "source_id": "source", "url": "https://example.org/contract", "quote": "An existing obligation.",
        "attribution": "exact_span"}])
    snapshot = snapshot_for(docs)
    attach_evidence(snapshot, snapshot.model_copy(deep=True), docs, "obligation")
    proposal = ActionProposal(agent_id="actor_a", agent_name="Laboratory", round_id=1, snapshot_version=0,
        action_id="action_1_actor_a", decision="wait", action_type="no_action", action="Wait", reasoning="Obligation",
        confidence=.5, timeline_refs=["old"])
    payload = cited_public_context(snapshot, "settlement", [proposal])
    assert "old" in {e["timeline_id"] for e in payload["timeline"]}
    assert payload["evidence_bundle"][0]["sources"][0]["quote"] == "An existing obligation."
    assert "queries" not in compact_json(payload)


def test_old_experiments_keep_legacy_retrieval_when_resumed(tmp_path, monkeypatch):
    events, store = EventRepository(tmp_path / "events"), EnsembleStore(tmp_path / "runs")
    rid, req = prepare_experiment(request(), events=events, store=store)
    config = store.load(rid, "config")
    config["engine_version"] = "ensemble-6"
    config.pop("retrieval")
    store.save(rid, "config", config)

    def forbidden(*args, **kwargs):
        raise AssertionError("A legacy experiment must not switch retrieval methods")

    monkeypatch.setattr(DecisionEvidenceRetriever, "__init__", forbidden)
    result = asyncio.run(run_experiment(rid, req, store=store, runtime=Classifier(), round_runner=runner([])))
    assert result["completed_samples"] == req.sample_count


def test_public_api_pins_search_record_and_graph_to_the_same_release(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api import experiments
    events, case_id = corpus(tmp_path)
    store = KnowledgeStore(events)
    version = store.publish(case_id)["version"]
    events.save_sources(case_id, [{"url": "https://example.org/changed", "content": "Different source."}])
    assert store.publish(case_id)["version"] != version
    monkeypatch.setattr(experiments, "KnowledgeStore", lambda: store)
    app = FastAPI()
    app.include_router(experiments.router)
    with TestClient(app) as client:
        search = client.get(f"/api/knowledge/{case_id}/search", params={"q": "供应", "version": version})
        assert search.status_code == 200 and search.json()["manifest"]["version"] == version
        record = client.get(f"/api/knowledge/{case_id}/records/hist_1", params={"version": version})
        assert record.status_code == 200 and record.json()["metadata"]["date_validation"]["status"] == "supported"
        graph = client.get(f"/api/knowledge/{case_id}/graph", params={"version": version})
        assert graph.status_code == 200 and graph.json()["version"] == version
        assert client.get(f"/api/knowledge/{case_id}/search", params={"version": "../current"}).status_code == 422
