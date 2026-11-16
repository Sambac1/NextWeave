from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from app.domain.behavior import BehavioralModel
from app.domain.decision_core import DecisionCore, CurrentPlan


class SituationView(BaseModel):
    topic: str
    assessment: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    timeline_refs: list[str] = Field(default_factory=list)
    constraint_ref: str | None = None


class ActorModel(BaseModel):
    agent_name: str
    assessment: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    timeline_refs: list[str] = Field(default_factory=list)
    constraint_ref: str | None = None


class Beliefs(BaseModel):
    situation_views: list[SituationView] = Field(default_factory=list)
    actor_models: list[ActorModel] = Field(default_factory=list)
    desires: list[Desire] | None = None


class Desire(BaseModel):
    desire_id: str
    description: str
    priority: float | None = Field(default=None, ge=0, le=1)
    status: Literal["active", "satisfied", "blocked", "abandoned"] = "active"


class Capability(BaseModel):
    name: str
    level: float = Field(ge=0, le=1)


class DecisionPolicy(BaseModel):
    risk_tolerance: float = Field(ge=0, le=1)
    compromise_tendency: float = Field(ge=0, le=1)
    response_speed: float = Field(ge=0, le=1)


class PrivateObservation(BaseModel):
    """One private fact delivered to exactly one ordinary Agent.

    The Game Master decides which facts each Agent knows.  It does not explain
    why the Agent knows them and it never writes the Agent's BDI directly.
    """

    observation_id: str
    description: str
    received_round: int
    processed: bool = False
    processed_round: int | None = None


class ActivationState(BaseModel):
    status: Literal["active", "inactive", "cooldown"] = "inactive"
    score: float = Field(default=0, ge=0, le=1)
    master_score: float = Field(default=0, ge=0, le=1)
    self_score: float = Field(default=0, ge=0, le=1)
    trigger_score: float = Field(default=0, ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    last_active_round: int | None = None
    last_review_round: int | None = None
    last_deliberation_round: int | None = None
    consecutive_active_rounds: int = 0
    inactive_rounds: int = 0
    low_score_streak: int = 0
    cooldown_rounds: int = 0
    last_information_update_round: int | None = None


class AgentProfile(BaseModel):
    """One canonical decision subject. ``agent_id`` is its only identity key."""

    agent_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    agent_type: str = Field(min_length=1)
    decision_subject: str = Field(min_length=1)
    role: str = Field(min_length=1)
    decision_core: DecisionCore | None = None
    current_plan: CurrentPlan | None = None
    behavioral_model: BehavioralModel = Field(default_factory=BehavioralModel)
    title: str = ""
    affiliation: str = ""
    description: str = Field(min_length=1)
    public_state: dict[str, Any] = Field(default_factory=dict)
    relevant_timeline_refs: list[str] = Field(default_factory=list)
    # Frozen public evidence; per-trajectory retrieval context, not private BDI.
    evidence_context: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_context: dict[str, Any] = Field(default_factory=dict)
    historical_actions: list[dict[str, Any]] = Field(default_factory=list)
    private_action_history: list[dict[str, Any]] = Field(default_factory=list)
    private_observations: list[PrivateObservation] = Field(default_factory=list)
    beliefs: Beliefs
    desires: list[Desire] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    decision_policy: DecisionPolicy
    capability_constraint_claims: list[dict[str, Any]] = Field(default_factory=list)
    activation: ActivationState = Field(default_factory=ActivationState)
    last_processed_world_round: int = 0
    last_information_fingerprint: str = ""


class AgentBDIPatch(BaseModel):
    beliefs: Beliefs | None = None
    desires: list[Desire] | None = None


class AgentProfilePatch(BaseModel):
    capabilities: list[Capability] | None = None
    constraints: list[str] | None = None
    reason: str
    timeline_refs: list[str] = Field(default_factory=list)
