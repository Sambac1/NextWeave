from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Literal

from app.domain import AgentProfile, Beliefs, Capability, DecisionPolicy, Desire, PrivateObservation, WorldSnapshot
from app.domain.identity import identity_key, stable_agent_id
from app.domain.behavior import behavior_from_record
from app.domain.agency import is_decision_subject, normalize_entity_record
from app.domain.decision_core import DecisionCore, grounded_decision_fields


MAX_ORDINARY_AGENTS = 8


def _desires(raw: object) -> list[Desire]:
    if not isinstance(raw, list):
        raise ValueError("Agent desires must be a list of structured objects")
    prepared = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Agent desire entries must be objects")
        prepared.append(Desire.model_validate(item))
    return prepared


class NoEligibleSimulationActors(ValueError):
    """No extracted subject can independently participate."""

    def __init__(self, message: str, *, readiness_counts: dict[str, int] | None = None):
        super().__init__(message)
        self.readiness_counts = readiness_counts or {}
        self.retryable = self.readiness_counts.get('retryable_error', 0) > 0


SimulationReadiness = Literal['ready', 'restricted', 'needs_evidence', 'needs_bdi', 'retryable_error']


def simulation_readiness(raw_actor: dict[str, Any], *, require_grounded: bool = False,
                         on_date: str = '', require_bdi: bool = False) -> SimulationReadiness:
    if not is_decision_subject(raw_actor):
        return 'needs_evidence'
    raw_core = raw_actor.get('decision_core')
    if raw_core is None:
        return 'needs_evidence' if require_grounded or require_bdi else 'ready'
    try:
        core = DecisionCore.model_validate(raw_core)
    except ValueError:
        return 'retryable_error'
    if core.origin == 'minimal_fallback':
        return 'retryable_error'
    if core.seat and identity_key(core.seat.owner_name) != identity_key(
        raw_actor.get('name') or raw_actor.get('source_figure') or raw_actor.get('canonical_name')):
        return 'retryable_error'
    if core.simulation_eligible is False:
        return 'needs_evidence'
    if core.origin == 'legacy_projection':
        return 'needs_evidence' if require_grounded or require_bdi else 'ready'
    supported = grounded_decision_fields(core, on_date or core.as_of)
    if supported and require_bdi:
        from app.simulation.decision_profile import initial_beliefs, initial_plan

        boundary = (on_date or core.as_of)[:10]
        has_goal = any(goal.provenance in {"source_statement", "inferred"}
                       and (goal.timeline_refs or goal.source_refs)
                       and (not boundary or ((not goal.valid_from or goal.valid_from <= boundary)
                                             and (not goal.valid_until or boundary <= goal.valid_until)))
                       for goal in core.goals)
        if not (initial_beliefs(core).situation_views or has_goal or initial_plan(core, boundary)):
            return 'needs_bdi'
    if len(supported) == 2:
        return 'ready'
    return 'restricted' if supported else 'needs_evidence'


def eligible_simulation_inputs(raw_agents: list[dict[str, Any]], *, require_grounded: bool = False,
                               on_date: str = '', require_bdi: bool = False) -> list[dict[str, Any]]:
    """Admit grounded dossiers and retain caller-declared stored inputs outside strict evaluation."""
    decisions = [(actor, simulation_readiness(actor, require_grounded=require_grounded, on_date=on_date, require_bdi=require_bdi))
                 for actor in raw_agents]
    eligible = [actor for actor, readiness in decisions if readiness in {'ready', 'restricted'}]
    if raw_agents and not eligible:
        counts = dict(Counter(readiness for _, readiness in decisions))
        message = ("Profile generation failed. Resume profile generation to identify actors who can make independent decisions."
                   if counts.get('retryable_error') else
                   "No actor has both supported decision authority and a usable initial BDI state. Add relevant Wikipedia evidence."
                   if counts.get('needs_bdi') else
                   "The profiles lack verified decision authority or means of action. No independent decision actors are available.")
        raise NoEligibleSimulationActors(message, readiness_counts=counts)
    return eligible


def profile_from_input(raw: dict[str, Any], event_id: str | None, event_title: str) -> AgentProfile:
    """Validate an API/extraction record at the boundary; internal code uses one schema only."""
    raw = normalize_entity_record(raw)
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError("Agent name is required")
    if not is_decision_subject(raw):
        raise ValueError(f"{name} is an asset, place, or event. Reanalyze the sources to identify the responsible person or organization as the decision actor.")
    agent_id = str(raw.get("agent_id") or "").strip() or stable_agent_id(event_id, event_title, name)
    raw_history = raw.get("historical_actions")
    if raw_history is None:
        raw_history = []
    historical_actions = [item if isinstance(item, dict) else {"description": str(item)} for item in raw_history]
    raw_beliefs = raw.get("beliefs")
    if not isinstance(raw_beliefs, dict):
        raise ValueError(f"Agent '{name}' has no structured beliefs")
    raw_policy = raw.get("decision_policy")
    if not isinstance(raw_policy, dict):
        raise ValueError(f"Agent '{name}' has no structured decision policy")
    return AgentProfile(
        agent_id=agent_id,
        name=name,
        aliases=[str(value) for value in raw.get("aliases") or [] if str(value).strip()],
        agent_type=str(raw.get("agent_type") or "").strip(),
        decision_subject=str(raw.get("decision_subject") or "").strip(),
        role=str(raw.get("role") or "").strip(),
        decision_core=raw.get("decision_core"),
        current_plan=raw.get("current_plan"),
        behavioral_model=behavior_from_record(raw),
        title=str(raw.get("title") or "").strip(),
        affiliation=str(raw.get("affiliation") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        public_state=dict(raw.get("public_state") or {}),
        relevant_timeline_refs=[str(value) for value in raw.get("relevant_timeline_refs") or []],
        historical_actions=historical_actions,
        private_action_history=list(raw.get("private_action_history") or []),
        private_observations=[PrivateObservation.model_validate(item) for item in raw.get("private_observations") or []],
        beliefs=Beliefs.model_validate(raw_beliefs),
        desires=_desires(raw.get("desires") or []),
        capabilities=[Capability.model_validate(item) for item in raw.get("capabilities") or []],
        constraints=[str(value) for value in raw.get("constraints") or []],
        decision_policy=DecisionPolicy.model_validate(raw_policy),
        capability_constraint_claims=list(raw.get("capability_constraint_claims") or []),
        activation=raw.get("activation") or {},
        last_processed_world_round=int(raw.get("last_processed_world_round") or 0),
    )


def _refs(agent: AgentProfile) -> set[str]:
    return {identity_key(agent.agent_id), identity_key(agent.name), identity_key(agent.decision_subject), *(identity_key(alias) for alias in agent.aliases)} - {""}


def _merge_unique(left: list[Any], right: list[Any], key) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for item in [*left, *right]:
        marker = key(item)
        if marker and marker not in seen:
            seen.add(marker)
            result.append(deepcopy(item))
    return result


def merge_profiles(primary: AgentProfile, duplicate: AgentProfile) -> AgentProfile:
    merged = primary.model_copy(deep=True)
    merged.aliases = _merge_unique([primary.name, *primary.aliases], [duplicate.name, *duplicate.aliases], identity_key)
    merged.relevant_timeline_refs = _merge_unique(primary.relevant_timeline_refs, duplicate.relevant_timeline_refs, str)
    merged.historical_actions = _merge_unique(primary.historical_actions, duplicate.historical_actions, lambda item: identity_key(str(item.get("action_id") or item.get("description") or item)))
    merged.private_action_history = _merge_unique(primary.private_action_history, duplicate.private_action_history, lambda item: identity_key(str(item.get("action_id") or item)))
    merged.private_observations = _merge_unique(primary.private_observations, duplicate.private_observations, lambda item: item.observation_id)
    return merged


def _importance(agent: AgentProfile) -> float:
    return min(len(agent.relevant_timeline_refs),20)*0.05 + min(len(agent.historical_actions),10)*0.05 + (0.2 if agent.activation.status=="active" else 0)


def maintain_registry(snapshot: WorldSnapshot) -> WorldSnapshot:
    """Merge duplicate agents, apply each one's alias set, and keep at most eight."""
    updated = snapshot.model_copy(deep=True)
    agents = list(updated.agents)
    changes: list[dict[str, Any]] = []

    canonical: list[AgentProfile] = []
    for candidate in agents:
        duplicate_index = next((i for i, item in enumerate(canonical) if _refs(item) & _refs(candidate)), None)
        if duplicate_index is None:
            canonical.append(candidate)
        else:
            canonical[duplicate_index] = merge_profiles(canonical[duplicate_index], candidate)
            updated.archived_agents.append(candidate)
            changes.append({"operation": "merge", "agent_id": candidate.agent_id, "merge_into_agent_id": canonical[duplicate_index].agent_id, "reason": "Duplicate name, alias, ID, or decision subject."})

    ranked = sorted(canonical, key=lambda item: (-_importance(item), identity_key(item.name)))
    updated.agents = ranked[:MAX_ORDINARY_AGENTS]
    for archived in ranked[MAX_ORDINARY_AGENTS:]:
        updated.archived_agents.append(archived)
        changes.append({"operation": "archive", "agent_id": archived.agent_id, "reason": "Registry capacity exceeded"})
    updated.registry_changes.extend(changes)
    return updated
