from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.agents import AgentProfile
from app.domain.activities import ActivityAnnotation
from app.domain.process import ProcessIssue


class ExecutedDecision(BaseModel):
    actor_id: str
    action_id: str
    quote: str = Field(min_length=8, max_length=500)


class TimelineEvent(BaseModel):
    timeline_id: str
    timestamp: str
    description: str
    actors: list[str] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    event_type: str = "event"
    activity_annotations: list[ActivityAnnotation] = Field(default_factory=list, max_length=8)
    cause_action_ids: list[str] = Field(default_factory=list, max_length=8)
    executed_decisions: list[ExecutedDecision] = Field(default_factory=list, max_length=8)
    change_kind: Literal['unspecified', 'new_action', 'state_change', 'continuation', 'external_observation'] = 'unspecified'
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(ge=0, le=1)
    date_precision: str = "unknown"
    date_original: str = ""
    time_role: str = "occurrence"
    modality: str = "occurred"
    date_evidence: dict[str, Any] = Field(default_factory=dict)
    date_range: dict[str, str] = Field(default_factory=dict)
    evidence_text: str = ""
    source_sections: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    source_event_count: int = 0
    origin: Literal["historical", "projected", "simulated"] = "historical"
    round_id: int | None = None
    retrieval_actors: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_scores: dict[str, Any] = Field(default_factory=dict)


class RelationshipEvidence(BaseModel):
    timeline_id: str
    timestamp: str = ""
    action: str = ""
    description: str = ""
    source_sections: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    source: str
    target: str
    relation_type: str
    description: str
    weight: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: list[RelationshipEvidence] = Field(default_factory=list)
    origin: Literal["historical", "simulated"] = "historical"


class RoundManifest(BaseModel):
    round_id: int
    input_snapshot_version: int
    output_snapshot_version: int
    active_agent_ids: list[str] = Field(default_factory=list)
    action_ids: list[str] = Field(default_factory=list)
    status: Literal["committed"] = "committed"


class WorldSnapshot(BaseModel):
    event_id: str | None = None
    event_title: str
    background: str = ""
    state_summary: str = ""
    world_state: dict[str, Any] = Field(default_factory=dict)
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    process_issues: list[ProcessIssue] = Field(default_factory=list, max_length=20)
    agents: list[AgentProfile] = Field(default_factory=list)
    archived_agents: list[AgentProfile] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    # What the last registry pass actually did (dedupe merges, capacity
    # archives), written by ``maintain_registry`` for the audit trail.
    registry_changes: list[dict[str, Any]] = Field(default_factory=list)
    # The event's shared strength dimensions.
    strength_dimensions: list[str] = Field(default_factory=list)
    round_manifests: list[RoundManifest] = Field(default_factory=list)
    round_id: int = 0
    snapshot_version: int = 0
