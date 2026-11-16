from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, StringConstraints, field_validator

from app.domain.agents import AgentProfilePatch
from app.domain.decision_core import CurrentPlan
from app.domain.common import LLMOptions
from app.domain.world import Relationship, TimelineEvent, WorldSnapshot
from app.domain.process import IssueUpdate
from app.domain.activities import ActivityStage


_WATCHPOINT_ALIASES = ("watchpoints", "watch_points", "next_watchpoint")


class SelectedAction(BaseModel):
    agent_id: str = Field(min_length=1)
    agent_name: str = ""
    action: str = Field(min_length=1)
    is_custom: bool = False
    visibility: Literal["public", "direct", "private", "secret"] = "public"
    target: str = ""


class ActivationDecision(BaseModel):
    agent_id: str
    selected: bool
    score: float = Field(ge=0, le=1)
    master_score: float = Field(ge=0, le=1)
    self_score: float = Field(ge=0, le=1)
    trigger_score: float = Field(ge=0, le=1)
    mandatory: bool = False
    reasons: list[str] = Field(default_factory=list)


class ActionProposal(BaseModel):
    agent_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    round_id: int
    snapshot_version: int
    action_id: str = Field(min_length=1)
    decision: Literal["act", "wait", "monitor", "no_action"]
    action_type: str = Field(min_length=1)
    action: str = Field(min_length=1)
    target: str = ""
    visibility: Literal["public", "direct", "private", "secret"] = "public"
    timeline_refs: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)
    expected_effect: str = Field(default="", max_length=400)
    condition: str = Field(default="", max_length=300)
    commitment: str = Field(default="", max_length=300)
    plan_update: CurrentPlan | None = None
    confidence: float = Field(ge=0, le=1)


class ForecastOutcome(BaseModel):
    description: str
    probability: float = Field(ge=0, le=1)
    affected_agent_ids: list[str] = Field(default_factory=list)


class AgentKnowledgeUpdate(BaseModel):
    """Private information the Game Master assigns to one ordinary Agent."""

    agent_id: str
    information: list[str] = Field(default_factory=list)


class AgentPublicUpdate(BaseModel):
    """Evidence-backed public profile changes produced outside the Game Master."""

    agent_id: str
    status: str = ""
    stance: str = ""
    summary_delta: str = ""
    profile_patch: AgentProfilePatch | None = None


class PowerImpact(BaseModel):
    """One actor's strength change in one dimension, stated by the settlement.

    The Game Master judges the change directly rather than leaving it implied:
    it names the actor, the event strength dimension, and whether that standing
    strengthened or weakened this round.
    """

    agent: str
    dimension: str
    status: str
    new_description: str = ""
    reason: str = ""


class TerminalObservation(BaseModel):
    dimension_id: str
    event_ref: str
    stage: ActivityStage
    quote: str = Field(min_length=8, max_length=500)


class RoundStateObservation(BaseModel):
    dimension_id: str
    category_id: str
    actor_ids: list[str] = Field(default_factory=list, max_length=8)
    stage: ActivityStage | None = None
    quote: str = Field(min_length=8, max_length=500)
    event_ref: str | None = None
    expires_on: date | None = None
    until_superseded: bool = False


class ContractOutlook(BaseModel):
    p_yes: float = Field(ge=0, le=1, allow_inf_nan=False)
    basis: str = Field(min_length=8, max_length=600)
    evidence_quote: str = Field(default="", max_length=600)
    change_driver: Literal["settled_event", "validated_state", "time_elapsed"] | None = None
    evidence_ref: str | None = Field(default=None, max_length=200)
    estimate_kind: Literal["terminal_estimate", "terminal_carried_forward", "verified_simulation"] | None = None
    readout_policy: str | None = None
    driver_ref: str | None = Field(default=None, max_length=200)


StatePathSegment = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class WorldStateChange(BaseModel):
    """One proposed change to the frozen public scenario state."""

    model_config = ConfigDict(extra="forbid")

    path: list[StatePathSegment] = Field(min_length=1, max_length=8)
    operation: Literal["add", "replace", "remove", "expire"]
    before: JsonValue
    after: JsonValue
    reason: str = Field(min_length=5, max_length=350)
    event_ref: str | None = Field(default=None, max_length=200)
    evidence_quote: str | None = Field(default=None, min_length=8, max_length=500)


class ActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: str
    status: Literal["executed", "partial", "blocked", "pending", "waiting", "unknown"]
    summary: str = Field(min_length=1, max_length=300)
    event_refs: list[str] = Field(default_factory=list, max_length=4)
    private_observation_quote: str = Field(default="", max_length=400)


class ActionInteraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_ids: list[str] = Field(min_length=2, max_length=6)
    kind: Literal["supports", "conflicts", "depends_on", "competes_for_resource", "delayed_effect"]
    precondition: str = Field(min_length=1, max_length=250)
    available_at: Literal["round_start", "precommitted", "new_this_round"]
    outcome: Literal["realized", "pending", "unresolved"]
    effect: str = Field(min_length=1, max_length=300)
    event_refs: list[str] = Field(default_factory=list, max_length=4)
    evidence_quote: str = Field(default="", max_length=500)
    private_observation_quote: str = Field(default="", max_length=400)


class PendingDecision(BaseModel):
    """A public reason for actor review, distinct from observer uncertainty."""

    model_config = ConfigDict(extra="forbid")
    actor_id: str
    kind: Literal["response", "decision_window", "opportunity"]
    event_ref: str
    evidence_quote: str = Field(min_length=8, max_length=500)
    choice: str = Field(default="", max_length=240)
    review_by: str = Field(default="", max_length=10)

    @field_validator("review_by")
    @classmethod
    def calendar_deadline(cls, value):
        return date.fromisoformat(value).isoformat() if value else ""


class QuestionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition_id: str
    before: Literal["unknown", "pending", "met", "unmet", "blocked"]
    after: Literal["unknown", "pending", "met", "unmet", "blocked"]
    summary: str = Field(min_length=1, max_length=300)
    event_refs: list[str] = Field(default_factory=list, max_length=4)
    evidence_quote: str = Field(default="", max_length=500)
    pending_actor_ids: list[str] = Field(default_factory=list, max_length=8)
    pending_decisions: list[PendingDecision] = Field(default_factory=list, max_length=8)


class GameMasterSettlement(BaseModel):
    """The Game Master's complete authority boundary for one frozen round."""

    state_summary: str
    next_world_state: dict[str, Any]
    contract_outlook: ContractOutlook | None = None
    outcomes: list[ForecastOutcome] = Field(default_factory=list)
    public_events: list[TimelineEvent] = Field(default_factory=list)
    terminal_observations: list[TerminalObservation] = Field(default_factory=list, max_length=6)
    state_observations: list[RoundStateObservation] = Field(default_factory=list, max_length=9)
    issue_updates: list[IssueUpdate] = Field(default_factory=list, max_length=6)
    continuing_state_notes: list[str] = Field(default_factory=list, max_length=12)
    _process_diagnostics: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    agent_knowledge_updates: list[AgentKnowledgeUpdate] = Field(default_factory=list)
    # Cardinality is checked against the actual chosen actions, including
    # interventions, rather than imposing a smaller independent fixed cap.
    action_results: list[ActionResult] = Field(default_factory=list)
    interactions: list[ActionInteraction] = Field(default_factory=list, max_length=16)
    question_updates: list[QuestionUpdate] = Field(default_factory=list, max_length=8)
    power_impacts: list[PowerImpact] = Field(default_factory=list)
    analysis: str
    next_watchpoints: list[str] = Field(default_factory=list, validation_alias=AliasChoices("next_watchpoints", *_WATCHPOINT_ALIASES))


class SequentialGameMasterSettlement(GameMasterSettlement):
    """Sequential runs settle a typed delta against the existing world."""

    model_config = ConfigDict(extra="forbid")

    next_world_state: dict[str, Any] = Field(default_factory=dict, max_length=0)
    state_changes: list[WorldStateChange] = Field(max_length=12)
    contract_progress: str | None = Field(default=None, max_length=600)
    contract_updates: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=6)


class AgentProfileProjection(BaseModel):
    """Public Agent-profile changes derived from settled public facts.

    ``agent_updates`` has no producer: the layered design fixes profiles at
    initialisation and never rewrites them per round, so this projection is
    always constructed empty and the round transition applies nothing. The
    field is kept so ``ForecastRound`` can keep reporting it -- it is part of
    the round payload the front end receives -- but it is empty by design, not
    a step that silently failed.
    """

    agent_updates: list[AgentPublicUpdate] = Field(default_factory=list)


class RelationshipProjection(BaseModel):
    """Relationship changes derived from settled public facts."""

    relationship_updates: list[Relationship] = Field(default_factory=list)


class EventReference(BaseModel):
    sample_id: int = Field(ge=1)
    timeline_id: str = Field(min_length=1)


class EventMatchGroup(BaseModel):
    """Semantic equivalence decided exclusively by the model."""

    event: TimelineEvent
    members: list[EventReference] = Field(min_length=1)
    reasoning: str


class EventGrouping(BaseModel):
    groups: list[EventMatchGroup] = Field(default_factory=list)


class EventVote(EventMatchGroup):
    occurrence_count: int
    sample_ids: list[int]
    retained: bool


class EventConsensus(BaseModel):
    prediction_repeats: int
    event_threshold_percent: int
    required_occurrences: int
    events: list[EventVote] = Field(default_factory=list)


class ConsensusStateProjection(BaseModel):
    """Project accepted facts only; the model cannot add or remove timeline events."""

    state_summary: str
    next_world_state: dict[str, Any]
    analysis: str
    next_watchpoints: list[str] = Field(default_factory=list, validation_alias=AliasChoices("next_watchpoints", *_WATCHPOINT_ALIASES))


class ForecastRound(BaseModel):
    simulation_interval: dict[str, Any] = Field(default_factory=dict)
    contract_observation: dict[str, Any] | None = None
    contract_outlook: ContractOutlook | None = None
    outlook_issue: str | None = None
    sampled_choices: list[dict[str, Any]] = Field(default_factory=list)
    action_results: list[ActionResult] = Field(default_factory=list)
    intervention_actions: list[ActionProposal] = Field(default_factory=list)
    interactions: list[ActionInteraction] = Field(default_factory=list)
    question_updates: list[QuestionUpdate] = Field(default_factory=list)
    state_changes: list[WorldStateChange] = Field(default_factory=list)
    step: int
    input_snapshot_version: int
    output_snapshot_version: int
    active_agent_ids: list[str] = Field(default_factory=list)
    activation_decisions: list[ActivationDecision] = Field(default_factory=list)
    agent_actions: list[ActionProposal] = Field(default_factory=list)
    decision_sampling: dict[str, int] = Field(default_factory=dict)
    state_summary: str
    issue_updates: list[IssueUpdate] = Field(default_factory=list)
    continuing_state_notes: list[str] = Field(default_factory=list)
    analysis: str = ""
    outcomes: list[ForecastOutcome] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    terminal_observations: list[TerminalObservation] = Field(default_factory=list, max_length=6)
    state_observations: list[RoundStateObservation] = Field(default_factory=list, max_length=9)
    new_relationships: list[Relationship] = Field(default_factory=list)
    agent_updates: list[AgentPublicUpdate] = Field(default_factory=list)
    next_watchpoints: list[str] = Field(default_factory=list, validation_alias=AliasChoices("next_watchpoints", *_WATCHPOINT_ALIASES))
    # The strength verdicts this round's settlement produced.
    power_impacts: list[PowerImpact] = Field(default_factory=list)

    event_consensus: EventConsensus | None = None


class Trajectory(BaseModel):
    candidate_id: str
    rounds: list[ForecastRound]
    final_snapshot: WorldSnapshot


class ScenarioBranch(BaseModel):
    id: str
    title: str
    probability: float
    trigger: str
    description: str
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    outcomes: list[dict[str, Any]] = Field(default_factory=list)
    children: list[dict[str, Any]] = Field(default_factory=list)
    rounds: list[ForecastRound] = Field(default_factory=list)
    final_state: str = ""
    final_world_state: dict[str, Any] = Field(default_factory=dict)
    selection_reason: str = ""


class SimulationRequest(LLMOptions):
    event_id: str | None = None
    event_title: str
    background: str = ""
    selected_actions: list[SelectedAction] = Field(default_factory=list)
    all_agents: list[dict[str, Any]] = Field(default_factory=list)
    agents: list[dict[str, Any]] = Field(default_factory=list)
    recent_timeline: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    steps: int = Field(default=1, ge=1, le=8)
    prediction_repeats: int = Field(default=4, ge=1, le=16)
    event_threshold_percent: int = Field(default=50, ge=1, le=100)


class SimulationResponse(BaseModel):
    mode: Literal["single_step", "multi_step"]
    prediction_repeats: int | None = None
    event_threshold_percent: int | None = None
    branches: list[ScenarioBranch] = Field(default_factory=list)
    analysis: str
    state_summary: str = ""
    final_state: str = ""
    steps: list[ForecastRound] = Field(default_factory=list)
    agent_actions: list[ActionProposal] = Field(default_factory=list)
    outcomes: list[ForecastOutcome] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    new_relationships: list[Relationship] = Field(default_factory=list)
    agent_updates: list[AgentPublicUpdate] = Field(default_factory=list)
    updated_context: WorldSnapshot | None = None
    probability_note: str | None = None
    run_id: str | None = None
