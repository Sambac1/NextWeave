from app.knowledge.registry import normalize_agent_registry
from app.storage.json_store import EventRepository, GraphRepository


def _candidate(candidate_id, name, entity_type, importance, *, affiliation="", title=""):
    return {
        "candidate_id": candidate_id,
        "canonical_name": name,
        "aliases": [],
        "entity_type": entity_type,
        "importance_score": importance,
        "evidence_count": 2,
        "actions_initiated_count": 1,
        "actions_received_count": 0,
        "latest_timestamp": "2026-08-05",
        "affiliation": affiliation,
        "title": title,
    }


def test_original_registry_normalization_merges_executive_decision_centers() -> None:
    candidates = [
        _candidate("candidate_001", "China", "country", 1.0),
        _candidate("candidate_002", "US", "country", 0.58),
        _candidate("candidate_003", "Trump", "person", 0.49),
        _candidate("candidate_004", "Donald Trump", "person", 0.38),
        _candidate("candidate_005", "US Department of Commerce", "government", 0.33, affiliation="US"),
        _candidate("candidate_006", "Trump administration", "government", 0.32),
        _candidate("candidate_007", "United States", "country", 0.31),
        _candidate("candidate_008", "White House", "government", 0.28, affiliation="United States"),
        _candidate("candidate_014", "Nvidia", "company", 0.24, affiliation="US"),
        _candidate("candidate_015", "Chinese companies", "group", 0.22, affiliation="China"),
    ]
    parsed = {
        "agents": [
            {"agent_id": f"agent_{index:03d}", "canonical_name": candidate["canonical_name"],
             "agent_type": candidate["entity_type"], "decision_subject": candidate["canonical_name"],
             "member_candidate_ids": [candidate["candidate_id"]], "reason": "selected"}
            for index, candidate in enumerate(candidates, 1)
        ],
        "historical": [], "context": [], "uncertain": [],
    }
    result = normalize_agent_registry(parsed, candidates, 8, "2026-08-05")
    by_name = {item["canonical_name"]: item for item in result["agents"]}
    assert set(by_name) == {"China", "Trump administration", "Nvidia", "Chinese companies"}
    assert set(by_name["Trump administration"]["member_candidate_ids"]) == {
        "candidate_002", "candidate_003", "candidate_004", "candidate_005",
        "candidate_006", "candidate_007", "candidate_008",
    }
    assert result["name_mapping"]["US"] == "Trump administration"
    assert result["name_mapping"]["Trump"] == "Trump administration"


def test_registry_refresh_removes_stale_agent_nodes_and_incident_edges(tmp_path) -> None:
    events = EventRepository(tmp_path)
    event_id = events.create("Event", ["https://example.test"], None)["event_id"]
    graph = GraphRepository(events)
    graph.add_node(event_id, {"id": "agent_keep", "label": "Keep", "type": "agent"})
    graph.add_node(event_id, {"id": "agent_old", "label": "Old", "type": "agent"})
    graph.add_edge(event_id, {"source": "agent_keep", "target": "agent_old", "label": "old"})
    removed = graph.remove_unregistered_agents(event_id, {"agent_keep"})
    refreshed = graph.get(event_id)
    assert removed == ["agent_old"]
    assert {item["data"]["id"] for item in refreshed["nodes"]} == {"agent_keep"}
    assert refreshed["edges"] == []
