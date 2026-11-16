from __future__ import annotations

from app.knowledge.okf import KnowledgeStore
from app.storage.json_store import EventRepository


def corpus(tmp_path):
    events = EventRepository(tmp_path / "events")
    case = events.create("供应链", ["https://example.org/wiki/supply"], "")
    identifier = case["event_id"]
    events.save_sources(identifier, [{"title": "Supply", "url": "https://example.org/wiki/supply",
                                      "content": "== January ==\n2020年1月，实验室启动供应试验。\n"}])
    historical = {"timeline_id": "hist_1", "timestamp": "2020-01", "description": "实验室启动供应试验",
                  "actors": ["actor_a"], "source_sections": ["source_01/January"],
                  "evidence_text": "2020年1月，实验室启动供应试验。", "origin": "historical"}
    events.save_artifact(identifier, "timeline", [historical, historical | {"timeline_id": "sim_1", "origin": "simulated"}])
    events.save_artifact(identifier, "agents", [{"agent_id": "actor_a", "name": "实验室", "aliases": ["Lab"],
        "description": "公开档案", "beliefs": {"private": "PRIVATE_SENTINEL"},
        "public_state": {"accidental_secret": "PRIVATE_SENTINEL"}}])
    return events, identifier


def test_okf_evidence_graph_and_privacy_survive_reindex(tmp_path):
    events, case_id = corpus(tmp_path)
    store = KnowledgeStore(events)
    manifest = store.publish(case_id)
    assert manifest["timeline_count"] == 1
    text = "\n".join(p.read_text() for p in store.root.rglob("*.md"))
    assert "PRIVATE_SENTINEL" not in text and "sim_1" not in text
    hits = store.search(case_id, "供应", kind="timeline")
    assert len(hits) == 1
    proof = hits[0]["metadata"]["evidence_refs"][0]
    assert proof["attribution"] == "exact_span" and proof["span_start"] >= 0
    assert len(store.search(case_id, "Lab", kind="actor")) == 1
    graph = store.graph(case_id)
    assert {n["type"] for n in graph["nodes"]} == {"timeline", "actor", "source"}
    assert len(graph["edges"]) == 2
    store.database.unlink()
    assert store.search(case_id, "供应", kind="timeline")[0]["id"] == "hist_1"
    assert store.publish(case_id)["version"] == manifest["version"]


def test_source_changes_keep_immutable_snapshots_and_do_not_claim_exact_evidence(tmp_path):
    events, case_id = corpus(tmp_path)
    store = KnowledgeStore(events)
    first = store.publish(case_id)
    events.save_sources(case_id, [{"url": "https://example.org/wiki/supply", "content": "Updated page without the old quotation"}])
    second = store.publish(case_id)
    assert first["version"] != second["version"]
    assert len(list((store.root / "references").rglob("*.txt"))) == 2
    event = store.search(case_id, kind="timeline")[0]["metadata"]
    assert event["status"] == "draft"
    assert event["evidence_refs"][0]["attribution"] == "page_only"
    assert "verified" not in event


def test_coarse_source_reports_stay_searchable_outside_timeline_and_respect_cutoff(tmp_path):
    from app.domain import WorldSnapshot
    from app.simulation.evidence import DecisionEvidenceRetriever
    from test_ensemble import actor

    events, case_id = corpus(tmp_path)
    quote = "In 2020, the Laboratory surveyed supply operators about delivery."
    interval = "Between late-February and early-March 2021, the Laboratory received a shipment."
    plan = "The Laboratory plans to open a new facility in 2023."
    body = "\n\n".join([quote, interval, plan])
    events.save_sources(case_id, [{"url": "https://en.wikipedia.org/wiki/Laboratory", "title": "Laboratory",
        "content": body, "revision_at": "2022-01-01T12:00:00+00:00", "revision_id": "12"}])
    from app.knowledge.extraction import _sections
    section, = _sections(events.load_sources(case_id))
    def rejected(text, reason, date_text):
        return {"reason": reason, "candidate": {"source_sections": [section["path"]],
            "description": "UNVERIFIED MODEL PARAPHRASE", "date_original": date_text, "evidence_text": text}}
    events.save_artifact(case_id, "timeline_evidence_review", {"rejected": [
        rejected(quote, "no_unambiguous_source_month", "In 2020"),
        rejected(interval, "multiple_event_months_require_narrower_evidence", "Between late-February and early-March 2021"),
        rejected(plan, "announcement_act_not_supported", "2023"),
        rejected("Quote not in snapshot", "no_unambiguous_source_month", ""),
    ]})
    store = KnowledgeStore(events)
    manifest = store.publish(case_id)
    version, documents = store.snapshot(case_id)
    items = {row["body"]: row["metadata"] for row in documents if row["kind"] == "background"}
    assert manifest["background_count"] == 3
    assert items[quote]["date_precision"] == "year"
    assert items[interval]["reported_bounds"] == ["2021-02", "2021-03"]
    assert items[plan]["fact_status"] == "reported_plan"
    assert "UNVERIFIED MODEL PARAPHRASE" not in str(items)
    assert not store.search(case_id, "surveyed", kind="timeline")
    assert store.search(case_id, "surveyed", kind="background")
    assert not store.search(case_id, "surveyed", kind="background", from_month="2020-01")
    frozen = WorldSnapshot(event_title="Laboratory supply", agents=[actor()])
    for cutoff, expected in [("2022-01-01T11:59:59+00:00", False), ("2022-01-01T12:00:00+00:00", True)]:
        retriever = DecisionEvidenceRetriever(frozen, documents, version=version, cutoff_at=cutoff)
        _, audit = retriever.retrieve(frozen, frozen.agents[0], "Laboratory surveyed supply operators")
        assert any(row.get("kind") == "background" for row in audit.get("source_passages", [])) == expected


def test_fallback_evidence_cites_the_actual_excerpt_url(tmp_path):
    events, case_id = corpus(tmp_path)
    events.save_sources(case_id, [{"title": "Synthetic", "url": "https://original.example/", "sdk_fallback": True,
        "content": "A reconstructed synopsis", "sections": [{"url": "https://real.example/quote",
            "content": "2020年1月，实验室启动供应试验。", "title": "January", "section_id": "excerpt1"}]}])
    store = KnowledgeStore(events)
    store.publish(case_id)
    proof = store.search(case_id, kind="timeline")[0]["metadata"]["evidence_refs"][0]
    assert proof["url"] == "https://real.example/quote"


def test_unversioned_search_migrates_old_manifest_even_when_index_exists(tmp_path):
    from app.storage.json_store import write_json
    events, case_id = corpus(tmp_path)
    store = KnowledgeStore(events)
    manifest = store.publish(case_id)
    old = dict(manifest)
    old.pop("schema_version")
    old.pop("requires_reanalysis")
    write_json(store._case(case_id) / "current.json", old)
    # A warm SQLite index must not bypass schema migration.
    assert store.search(case_id, kind="timeline")
    assert store.manifest(case_id)["schema_version"] == 2
    assert store.manifest(case_id)["requires_reanalysis"]
