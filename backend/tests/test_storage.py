from __future__ import annotations

import pytest

from app.storage.json_store import EventRepository, GraphRepository, SnapshotRepository


def test_graph_and_snapshot_survive_repository_recreation(tmp_path) -> None:
    events = EventRepository(tmp_path)
    event_id = events.create("Persistent event", ["https://example.test"], None)["event_id"]
    graph = GraphRepository(events)
    graph.add_node(event_id, {"id": "a1", "label": "Actor", "type": "agent", "properties": {}})
    SnapshotRepository(events).commit(event_id, {"event_title": "Persistent event", "snapshot_version": 1}, expected_version=0)
    reloaded_events = EventRepository(tmp_path)
    assert GraphRepository(reloaded_events).get(event_id)["nodes"][0]["data"]["label"] == "Actor"
    assert SnapshotRepository(reloaded_events).get(event_id)["snapshot_version"] == 1


def test_snapshot_commit_rejects_stale_writer_and_can_be_cleared(tmp_path) -> None:
    events = EventRepository(tmp_path)
    event_id = events.create("Event", ["https://example.test"], None)["event_id"]
    snapshots = SnapshotRepository(events)
    snapshots.commit(event_id, {"snapshot_version": 1}, expected_version=0)
    with pytest.raises(RuntimeError, match="expected 0, current 1"):
        snapshots.commit(event_id, {"snapshot_version": 1}, expected_version=0)
    snapshots.clear(event_id)
    assert snapshots.get(event_id) is None
