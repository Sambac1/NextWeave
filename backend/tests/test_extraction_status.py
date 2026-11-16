from __future__ import annotations

from app.storage.extraction_status import ExtractionStatusRepository
from app.storage.json_store import EventRepository


def test_figures_result_from_artifacts_reads_saved_agents(tmp_path) -> None:
    from app.knowledge.extraction import figures_result_from_artifacts

    events = EventRepository(tmp_path)
    event_id = events.create("Event", ["https://example.test"], None)["event_id"]
    assert figures_result_from_artifacts(event_id, events) is None

    events.save_artifact(event_id, "agents", [{"agent_id": "a1", "name": "Alice"}])
    events.save_artifact(event_id, "timeline", [{"timeline_id": "t1", "description": "Alice acted"}])
    events.save_artifact(event_id, "name_mapping", {"Ali": "Alice"})
    result = figures_result_from_artifacts(event_id, events)
    assert result is not None
    assert result["figures"][0]["name"] == "Alice"
    assert result["timeline"][0]["timeline_id"] == "t1"
    assert result["name_mapping"]["Ali"] == "Alice"


def test_extraction_status_is_persisted_and_can_fail(tmp_path) -> None:
    events = EventRepository(tmp_path)
    event_id = events.create("Event", ["https://example.test"], None)["event_id"]
    statuses = ExtractionStatusRepository(events)

    running = statuses.update(
        event_id,
        run_id="run-1",
        stage="extracting_timeline",
        progress=45,
        message="Extracting timeline...",
        completed_items=4,
        total_items=10,
    )
    assert statuses.get(event_id) == running
    assert running["started_at"]

    failed = statuses.fail(event_id, "provider timeout")
    assert failed["status"] == "failed"
    assert failed["stage"] == "extracting_timeline"
    assert failed["progress"] == 45
    assert failed["error"] == "provider timeout"
