from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.storage.json_store import EventRepository, GraphRepository
from app.storage.run_logger import RunLogger


router = APIRouter(prefix="/api/agents", tags=["graph"])
events = EventRepository()
graphs = GraphRepository(events)


class NodeRequest(BaseModel):
    event_id: str
    node_id: str
    label: str
    node_type: str = "agent"
    properties: dict[str, Any] = Field(default_factory=dict)


class IdEdgeRequest(BaseModel):
    event_id: str
    source_id: str
    target_id: str
    label: str = "relationship"
    edge_type: str = "relationship"
    properties: dict[str, Any] = Field(default_factory=dict)
    weight: float = 0.5


@router.post("/node")
async def add_node(request: NodeRequest) -> dict[str, Any]:
    logger = RunLogger("graph_logs", operation="graph-node-upsert", event_id=request.event_id)
    try:
        node = graphs.add_node(
            request.event_id,
            {
                "id": request.node_id,
                "label": request.label,
                "type": request.node_type,
                "properties": request.properties,
            },
        )
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Event not found") from None
    graph = graphs.get(request.event_id)
    logger.record(
        source="graph-api",
        round_id="current",
        agent_id=request.node_id,
        content_type="graph-after-node-upsert",
        status="success",
        data={"changed_node": node, "graph": graph},
    )
    logger.finish(success=True, data={"node_count": len(graph["nodes"]), "edge_count": len(graph["edges"])})
    return {"status": "ok", "node": node}


@router.get("/{event_id}/graph")
async def get_graph(event_id: str) -> dict[str, Any]:
    logger = RunLogger("graph_logs", operation="graph-read", event_id=event_id)
    try:
        graph = graphs.get(event_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Graph not found") from None
    logger.record(
        source="graph-api",
        round_id="current",
        agent_id="all",
        content_type="persisted-graph-snapshot",
        status="success",
        data=graph,
    )
    logger.finish(success=True, data={"node_count": len(graph["nodes"]), "edge_count": len(graph["edges"])})
    return graph


def _edge_payload(request: IdEdgeRequest, source_id: str, target_id: str) -> dict[str, Any]:
    return {
        "source": source_id,
        "target": target_id,
        "label": request.label,
        "type": request.edge_type,
        "edge_type": request.edge_type,
        "description": str(request.properties.get("description") or request.label),
        "properties": request.properties,
        "weight": max(0, min(1, request.weight)),
    }


@router.post("/edge-by-id")
async def add_id_edge(request: IdEdgeRequest) -> dict[str, Any]:
    logger = RunLogger("graph_logs", operation="graph-edge-upsert", event_id=request.event_id)
    try:
        graph = graphs.get(request.event_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Event not found") from None
    node_ids = {str(item.get("data", {}).get("id")) for item in graph["nodes"]}
    if request.source_id not in node_ids or request.target_id not in node_ids:
        raise HTTPException(status_code=422, detail="source or target agent_id was not found in the event graph")
    edge = graphs.add_edge(
        request.event_id,
        _edge_payload(request, request.source_id, request.target_id),
    )
    updated_graph = graphs.get(request.event_id)
    logger.record(
        source="graph-api",
        round_id="current",
        agent_id=f"{request.source_id}-to-{request.target_id}",
        content_type="graph-after-edge-upsert",
        status="success",
        data={"changed_edge": edge, "graph": updated_graph},
    )
    logger.finish(
        success=True,
        data={"node_count": len(updated_graph["nodes"]), "edge_count": len(updated_graph["edges"])},
    )
    return {"status": "ok", "edge": edge}
