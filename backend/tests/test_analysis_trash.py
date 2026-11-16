import uuid
import pytest

from app.storage.analysis_trash import AnalysisTrash
from app.storage.ensemble_store import EnsembleStore
from app.knowledge.okf import KnowledgeStore
from app.storage.json_store import BENCHMARK_EVENT_ORIGIN, EventRepository
from test_okf import corpus


def test_benchmark_preparation_is_not_removed_with_analyses(tmp_path):
    events = EventRepository(tmp_path)
    event = events.create("Market question", [], None, origin=BENCHMARK_EVENT_ORIGIN)
    with pytest.raises(ValueError, match="belongs to a Market Evaluation"):
        AnalysisTrash(events).delete(event["event_id"])
    assert events.directory(event["event_id"]).is_dir()


def test_delete_and_restore_analysis_knowledge_and_experiments(tmp_path):
    events, case_id = corpus(tmp_path)
    knowledge = KnowledgeStore(events); knowledge.publish(case_id)
    store = EnsembleStore(events.root.parent / "ensemble_runs")
    run_id = str(uuid.uuid4())
    store.save(run_id, "config", {"request": {"event_id": case_id}})
    store.save(run_id, "initial_snapshot", {"private": "kept"})
    trash = AnalysisTrash(events)
    entry = trash.delete(case_id)
    assert not (events.root / case_id).exists()
    assert not store.directory(run_id).exists()
    assert len(trash.list()) == 1
    assert list((knowledge.root / "references").rglob("*.txt"))
    with knowledge._connection() as db:
        assert db.execute("SELECT COUNT(*) FROM docs WHERE case_id=?", (case_id,)).fetchone()[0] == 0
    restored = trash.restore(entry["trash_id"])
    assert restored["status"] == "restored" and not trash.list()
    assert store.load(run_id, "initial_snapshot") == {"private": "kept"}
    assert knowledge.search(case_id, "供应", kind="timeline")[0]["id"] == "hist_1"


def test_running_experiment_blocks_delete_and_conflicting_restore_never_overwrites(tmp_path):
    events, case_id = corpus(tmp_path)
    store = EnsembleStore(events.root.parent / "ensemble_runs")
    run_id = str(uuid.uuid4())
    store.save(run_id, "config", {"request": {"event_id": case_id}})
    trash = AnalysisTrash(events)
    with pytest.raises(ValueError, match="still running"):
        trash.delete(case_id, active_runs={run_id})
    assert events.directory(case_id).exists()
    entry = trash.delete(case_id)
    occupied = events.root / case_id
    occupied.mkdir()
    (occupied / "keep.txt").write_text("Do not overwrite")
    with pytest.raises(ValueError, match="destination is occupied"):
        trash.restore(entry["trash_id"])
    assert (occupied / "keep.txt").read_text() == "Do not overwrite"


def test_invalid_identifiers_cannot_escape_the_recycle_bin(tmp_path):
    events, _ = corpus(tmp_path)
    trash = AnalysisTrash(events)
    with pytest.raises(ValueError): trash.restore("../../elsewhere")
    with pytest.raises(ValueError): trash.delete("../../elsewhere")
    with pytest.raises(ValueError): trash.purge("../../elsewhere")


def test_purge_removes_all_versions_sources_caches_and_experiments(tmp_path):
    events, case_id = corpus(tmp_path)
    knowledge = KnowledgeStore(events)
    knowledge.publish(case_id)
    events.save_sources(case_id, [{"url": "https://example.org/wiki/supply", "content": "A new source revision"}])
    knowledge.publish(case_id)
    assert len(list((knowledge.root / "references").rglob("*.txt"))) == 2
    events.save_artifact(case_id, "timeline_cache", {"cached": True})
    store = EnsembleStore(events.root.parent / "ensemble_runs")
    run_id = str(uuid.uuid4())
    store.save(run_id, "config", {"request": {"event_id": case_id}})
    store.save(run_id, "initial_snapshot", {"private": "delete me"})
    untouched = events.root.parent / "keep.txt"
    untouched.write_text("another file")
    trash = AnalysisTrash(events)
    entry = trash.delete(case_id)
    assert trash.purge(entry["trash_id"])["status"] == "purged"
    assert not (trash.root / entry["trash_id"]).exists()
    assert not (events.root / case_id).exists()
    assert not (knowledge.root / "cases" / case_id).exists()
    assert not store.directory(run_id).exists()
    assert not list((knowledge.root / "references").rglob("*.txt"))
    assert not list((knowledge.root / "sources").rglob("*.md"))
    assert not trash.list()
    assert untouched.read_text() == "another file"
    with knowledge._connection() as db:
        assert db.execute("SELECT COUNT(*) FROM docs WHERE case_id=?", (case_id,)).fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM search_fts").fetchone()[0] == 0
    with pytest.raises(FileNotFoundError): trash.restore(entry["trash_id"])
    with pytest.raises(FileNotFoundError): trash.purge(entry["trash_id"])


@pytest.mark.parametrize("other_in_trash", [False, True])
def test_purge_preserves_shared_sources_until_last_owner_is_deleted(tmp_path, other_in_trash):
    events, case_id = corpus(tmp_path)
    _, other_id = corpus(tmp_path)
    knowledge = KnowledgeStore(events)
    knowledge.publish(case_id)
    knowledge.publish(other_id)
    trash = AnalysisTrash(events)
    entry = trash.delete(case_id)
    other_entry = trash.delete(other_id) if other_in_trash else None
    trash.purge(entry["trash_id"])
    assert list((knowledge.root / "references").rglob("*.txt"))
    assert list((knowledge.root / "sources").rglob("*.md"))
    if other_entry:
        trash.restore(other_entry["trash_id"])
    assert knowledge.search(other_id, "供应", kind="timeline")[0]["id"] == "hist_1"
    other_entry = trash.delete(other_id)
    trash.purge(other_entry["trash_id"])
    assert not list((knowledge.root / "references").rglob("*.txt"))
    assert not list((knowledge.root / "sources").rglob("*.md"))


def test_purge_keeps_sources_used_by_unpublished_analysis(tmp_path):
    events, case_id = corpus(tmp_path)
    _, other_id = corpus(tmp_path)
    knowledge = KnowledgeStore(events)
    knowledge.publish(case_id)
    trash = AnalysisTrash(events)
    entry = trash.delete(case_id)
    trash.purge(entry["trash_id"])
    assert list((knowledge.root / "references").rglob("*.txt"))
    knowledge.publish(other_id)
    assert knowledge.search(other_id, "供应", kind="timeline")


def test_purge_protects_other_analysis_historical_releases(tmp_path):
    events, case_id = corpus(tmp_path)
    _, other_id = corpus(tmp_path)
    knowledge = KnowledgeStore(events)
    knowledge.publish(case_id)
    knowledge.publish(other_id)
    events.save_sources(other_id, [{"url": "https://example.org/wiki/supply", "content": "A changed revision"}])
    knowledge.publish(other_id)
    trash = AnalysisTrash(events)
    trash.purge(trash.delete(case_id)["trash_id"])
    assert len(list((knowledge.root / "references").rglob("*.txt"))) == 2
    trash.purge(trash.delete(other_id)["trash_id"])
    assert not list((knowledge.root / "references").rglob("*.txt"))


def test_purge_removes_fallback_excerpt_snapshots(tmp_path):
    events, case_id = corpus(tmp_path)
    events.save_sources(case_id, [{"url": "https://example.org/background", "content": "Synthetic summary",
        "sdk_fallback": True, "sections": [{"section_id": "excerpt1", "title": "January", "url": "https://example.org/excerpt", "content": "2020年1月，实验室启动供应试验。"}]}])
    knowledge = KnowledgeStore(events)
    knowledge.publish(case_id)
    assert list((knowledge.root / "references").rglob("*.txt"))
    trash = AnalysisTrash(events)
    trash.purge(trash.delete(case_id)["trash_id"])
    assert not list((knowledge.root / "references").rglob("*.txt"))
    assert not list((knowledge.root / "sources").rglob("*.md"))


def test_partial_purge_stays_visible_and_can_be_retried(tmp_path, monkeypatch):
    import app.storage.analysis_trash as module
    events, case_id = corpus(tmp_path)
    knowledge = KnowledgeStore(events)
    knowledge.publish(case_id)
    trash = AnalysisTrash(events)
    entry = trash.delete(case_id)
    original = module.shutil.rmtree

    def interrupted(path):
        original(path)
        raise OSError("interrupted after removing a directory")

    monkeypatch.setattr(module.shutil, "rmtree", interrupted)
    with pytest.raises(OSError): trash.purge(entry["trash_id"])
    assert trash.list()[0]["status"] == "purging"
    with pytest.raises(ValueError, match="Permanent deletion has already started"): trash.restore(entry["trash_id"])
    monkeypatch.setattr(module.shutil, "rmtree", original)
    assert trash.purge(entry["trash_id"])["status"] == "purged"
    assert not trash.list()


def test_purge_removes_owned_external_logs_and_keeps_unrelated_logs(tmp_path):
    from app.storage.json_store import write_json
    events, case_id = corpus(tmp_path)
    data = events.root.parent
    request_run = str(uuid.uuid4())
    request_log = data / "request_logs" / "request"
    model_log = data / "knowledge_runs" / "model"
    direct_log = data / "event_logs" / "collection"
    other_log = data / "request_logs" / "unrelated"
    write_json(request_log / "0001.json", {"metadata": {"run_id": request_run}})
    write_json(request_log / "0002.json", {"data": {"body": {"event_id": case_id}}})
    write_json(model_log / "0001.json", {"metadata": {"run_id": request_run}, "data": {"prompt": "case source content"}})
    write_json(direct_log / "0001.json", {"metadata": {"event_id": case_id}})
    write_json(other_log / "0001.json", {"metadata": {"run_id": "another"}, "data": {"path": f"/api/events/{case_id}-suffix"}})
    trash = AnalysisTrash(events)
    trash.purge(trash.delete(case_id)["trash_id"])
    assert not request_log.exists() and not model_log.exists() and not direct_log.exists()
    assert (other_log / "0001.json").exists()


def test_purge_api_only_deletes_recycled_entries(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    import app.api.experiments as api

    events, case_id = corpus(tmp_path)
    trash = AnalysisTrash(events)
    monkeypatch.setattr(api, "AnalysisTrash", lambda: trash)
    client = TestClient(app)
    assert client.delete(f"/api/knowledge/trash/{case_id}").status_code == 404
    assert events.directory(case_id).exists()
    entry = trash.delete(case_id)
    assert client.get("/api/knowledge/trash").json()["entries"][0]["trash_id"] == entry["trash_id"]
    assert client.post(f"/api/knowledge/trash/{entry['trash_id']}/restore").status_code == 200
    assert client.delete(f"/api/knowledge/trash/{entry['trash_id']}").status_code == 404
    entry = trash.delete(case_id)
    response = client.delete(f"/api/knowledge/trash/{entry['trash_id']}")
    assert response.status_code == 200 and response.json()["status"] == "purged"
    assert client.get("/api/knowledge/trash").json() == {"entries": []}
    assert client.delete("/api/knowledge/trash/not-a-valid-id").status_code == 409
