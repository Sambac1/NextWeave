"""Relationship functional Agent for initialization and runtime projection."""

from __future__ import annotations

import json
from typing import Any

from app.domain import (
    GameMasterSettlement,
    RelationshipProjection,
    RelationshipsRequest,
    WorldSnapshot,
)
from app.domain.relationship_endpoints import clean_relationships, describe_agents
from app.knowledge.relationships import extract_relationships
from app.llm.client import LLMClient
from app.storage.json_store import EventRepository


class RelationshipAgent:
    """Build and incrementally maintain relationships from public evidence."""

    def __init__(self, runtime: LLMClient | None = None) -> None:
        self.runtime = runtime

    async def initialize(
        self,
        request: RelationshipsRequest,
        events: EventRepository,
    ) -> list[dict[str, Any]]:
        return await extract_relationships(request, events)

    async def update(
        self,
        snapshot: WorldSnapshot,
        settlement: GameMasterSettlement,
        call_prefix: str,
    ) -> RelationshipProjection:
        if self.runtime is None:
            raise RuntimeError("RelationshipAgent.update requires an injected Agent runtime")
        public_agents = describe_agents(snapshot.agents, members=True)
        result = await self.runtime.json_call(
            call_name=f"{call_prefix}_relationship_update",
            system=(
                "You maintain the public relationship graph from authoritative settled public events. Return only "
                "material new or changed relationships, use canonical agent_id endpoints, and cite supplied timeline IDs. "
                "Each agent card lists the members it acts through -- when an event names a member, use that member's "
                "principal agent_id as the endpoint. Do not use or infer private Agent information."
            ),
            prompt=(
                f"EVENT: {snapshot.event_title}\nROUND: {snapshot.round_id + 1}\n\n"
                f"AGENTS\n{json.dumps(public_agents, ensure_ascii=False, indent=2)}\n\n"
                f"EXISTING RELATIONSHIPS\n{json.dumps([item.model_dump() for item in snapshot.relationships], ensure_ascii=False, indent=2)}\n\n"
                f"SETTLED PUBLIC EVENTS\n{json.dumps([item.model_dump() for item in settlement.public_events], ensure_ascii=False, indent=2)}"
            ),
            schema=RelationshipProjection,
            temperature=0.1,
            max_tokens=5000,
        )
        known_timeline = {item.timeline_id for item in snapshot.timeline} | {
            item.timeline_id for item in settlement.public_events
        }
        result.relationship_updates = clean_relationships(
            result.relationship_updates,
            agents=snapshot.agents,
            known_timeline=known_timeline,
            call_name=f"{call_prefix}_relationship_update",
        )
        return result
