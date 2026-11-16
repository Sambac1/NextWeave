from __future__ import annotations

import json

from pydantic import BaseModel

from app.agents import AgentSDKClient
from app.domain import Relationship, RelationshipsRequest
from app.domain.relationship_endpoints import clean_relationships, describe_agents
from app.knowledge.operations import make_client
from app.storage.json_store import EventRepository
from app.storage.run_logger import audit_finish, audit_record


class RelationshipResult(BaseModel):
    relationships: list[Relationship]


async def extract_relationships(request: RelationshipsRequest, events: EventRepository) -> list[dict]:
    _, recorder = make_client(request, "knowledge_runs")
    client = AgentSDKClient(request, recorder)
    timeline = events.load_artifact(request.event_id, "timeline", []) if request.event_id else []
    action_evidence = events.load_artifact(request.event_id, "relationship_action_evidence", {}) if request.event_id else {}
    # Members are appended to each card for the same reason as the runtime
    # pass: the timeline names them, and a model that cannot see them writes a
    # name the endpoint check then has to reject.
    public_agents = describe_agents(request.agents, members=True)
    known_ids = {str(item.get("agent_id")) for item in public_agents if item.get("agent_id")}
    audit_record(
        recorder,
        source="relationship-extraction",
        round_id="extraction",
        agent_id="all",
        content_type="relationship-graph-input",
        status="info",
        data={
            "agents": public_agents,
            "timeline": timeline,
            "directed_action_evidence": action_evidence,
        },
    )
    result = await client.json_call(
        call_name="agent_relationship_extraction",
        system="Infer relationships conservatively from supplied timeline and directed-action evidence. Relationship endpoints must use canonical agent_id values, never display names.",
        prompt=(
            f"EVENT: {request.event_title}\nAGENTS\n{json.dumps(public_agents, ensure_ascii=False, indent=2)}\n\n"
            f"TIMELINE\n{json.dumps(timeline, ensure_ascii=False, indent=2)}\n\n"
            f"DIRECTED ACTION EVIDENCE\n{json.dumps(action_evidence, ensure_ascii=False, indent=2)}\n\n"
            "Return only decision-relevant relationships. Every relationship must cite timeline evidence and distinguish historical facts from simulated projections."
        ),
        schema=RelationshipResult,
        temperature=0.1,
        max_tokens=7000,
    )
    known_timeline = {str(item.get("timeline_id")) for item in timeline}
    # Same rule as the runtime pass: resolve member names onto their principal
    # and drop what cannot be resolved, rather than failing the extraction.
    kept = clean_relationships(
        result.relationships,
        agents=request.agents,
        known_timeline=known_timeline,
        call_name="agent_relationship_extraction",
    )
    relationships = [item.model_dump() for item in kept]
    audit_record(
        recorder,
        source="relationship-extraction",
        round_id="extraction",
        agent_id="all",
        content_type="relationship-graph-result",
        status="success",
        data={"nodes": public_agents, "edges": relationships},
    )
    if request.event_id:
        events.save_artifact(request.event_id, "relationships", relationships)
    audit_finish(
        recorder,
        success=True,
        data={"relationship_count": len(relationships)},
    )
    return relationships
