from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import date

from app.layers.simulation.strength import apply_power_impacts
from app.domain import (
    ActionProposal,
    GameMasterSettlement,
    PrivateObservation,
    Relationship,
    RelationshipProjection,
    RoundManifest,
    WorldSnapshot,
)
from app.domain.forecast import SequentialGameMasterSettlement
from app.simulation.protocol import RESERVED, program_projection, set_path


_SETTLEMENT_RESERVED = RESERVED | {
    "benchmark_contract", "contract_state", "contract_outlook", "contract_updates",
    "contract_progress", "contract_result", "process_topics", "process_tracking_version",
    "question_conditions", "question_states",
}


def _merge_state_fields(state: dict, updates: dict) -> None:
    """Adapt ordinary full-state responses to preserve omitted nested fields."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            _merge_state_fields(state[key], value)
        else:
            state[key] = deepcopy(value)


def _state_json(value):
    # JSON identity distinguishes a Boolean from the numeric values 0 and 1.
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _paths_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    size = min(len(left), len(right))
    return left[:size] == right[:size]


def project_world_state(snapshot: WorldSnapshot, settlement: GameMasterSettlement,
                        actions: list[ActionProposal], program_owned: dict | None = None) -> dict:
    """Build the next state off a copy and validate each proposed write."""
    state = deepcopy(snapshot.world_state)
    # Staged contract observations are consumed once at the end of the round.
    state.pop("contract_updates", None)
    if not isinstance(settlement, SequentialGameMasterSettlement):
        _merge_state_fields(state, {key: value for key, value in settlement.next_world_state.items()
                                    if key not in RESERVED})
        if "benchmark_contract" in snapshot.world_state:
            state["benchmark_contract"] = deepcopy(snapshot.world_state["benchmark_contract"])
            progress = state.pop("contract_progress", None)
            if isinstance(progress, str) and progress.strip():
                state["contract_progress"] = progress.strip()[:600]
            elif "contract_progress" in snapshot.world_state:
                state["contract_progress"] = snapshot.world_state["contract_progress"]
        return state

    if program_owned is None:
        program_owned = program_projection(snapshot, actions)
    protected = {tuple(path.split(".")) for path in program_owned}
    for path, value in program_owned.items():
        set_path(state, path, deepcopy(value))

    events = {event.timeline_id: event for event in settlement.public_events}
    measurement_paths = {tuple(path.split(".")) for path in snapshot.world_state.get("measurement_paths", [])}
    written = set()
    from app.simulation.joint_settlement import enabled
    for change in settlement.state_changes:
        path = tuple(change.path)
        label = ".".join(path)
        if path[0] in _SETTLEMENT_RESERVED or any(_paths_overlap(path, other) for other in protected):
            raise ValueError(f"Settlement cannot change program-owned state: {label}")
        if any(_paths_overlap(path, other) for other in written):
            raise ValueError(f"Settlement contains overlapping state changes: {label}")
        written.add(path)

        parent = state
        for segment in path[:-1]:
            if (segment not in parent and change.operation == "add" and change.before is None
                    and enabled(snapshot)):
                # Materialize only absent object containers for this exact add.
                # Existing null/scalar values are never reinterpreted as objects.
                parent[segment] = {}
                diagnostic = {"code": "state_object_parent_created", "path": list(path),
                              "round_id": snapshot.round_id + 1}
                if diagnostic not in settlement._process_diagnostics:
                    settlement._process_diagnostics.append(diagnostic)
            parent = parent.get(segment)
            if not isinstance(parent, dict):
                raise ValueError(f"Settlement state path has no object parent: {label}")
        key = path[-1]
        previous = parent.get(key)
        before, after = _state_json(change.before), _state_json(change.after)
        if change.operation == "add":
            if key in parent or change.before is not None:
                raise ValueError(f"Settlement add must name a new state path: {label}")
        elif key not in parent or _state_json(previous) != before:
            raise ValueError(f"Settlement before-value does not match the frozen state: {label}")

        if change.operation == "expire":
            try:
                end = date.fromisoformat(snapshot.world_state["simulation_clock"]["interval_end"])
                expires = date.fromisoformat(previous["expires_on"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Settlement expiration needs a stored expiry and a round boundary: {label}") from None
            if expires > end or change.after is not None:
                raise ValueError(f"Settlement expiration has not taken effect: {label}")
        elif change.event_ref or change.evidence_quote:
            event = events.get(change.event_ref)
            if event is None or not change.evidence_quote or change.evidence_quote not in event.description:
                raise ValueError(f"Settlement state change must cite a retained public event and exact quote: {label}")
        elif not (change.operation == "replace" and change.after is None and path in measurement_paths):
            raise ValueError(f"Settlement state change lacks a public observation: {label}")

        if change.operation in {"expire", "remove"}:
            if change.after is not None:
                raise ValueError(f"Settlement removal must have null as its after-value: {label}")
            del parent[key]
        elif change.operation == "replace" and before == after:
            raise ValueError(f"Settlement replacement must change the state: {label}")
        else:
            parent[key] = deepcopy(change.after)

    if settlement.contract_progress and snapshot.world_state.get("benchmark_contract"):
        state["contract_progress"] = settlement.contract_progress.strip()
    if settlement.contract_updates and snapshot.world_state.get("benchmark_contract", {}).get("spec", {}).get("stages"):
        state["contract_updates"] = deepcopy(settlement.contract_updates)
    from app.simulation.joint_settlement import apply_question_updates, enabled
    if enabled(snapshot):
        state["question_states"] = apply_question_updates(snapshot, settlement)
    return state


def _merge_relationship(existing: Relationship, incoming: Relationship) -> Relationship:
    merged = incoming.model_copy(deep=True)
    evidence = {item.timeline_id: item for item in [*existing.evidence, *incoming.evidence]}
    merged.evidence = list(evidence.values())
    return merged


def _strength_dimensions(agents) -> list[str]:
    """The event's strength dimensions, in the order the profiles list them."""
    dimensions: list[str] = []
    for agent in agents:
        for capability in agent.capabilities:
            if capability.name and capability.name not in dimensions:
                dimensions.append(capability.name)
    return dimensions


def apply_round_update(
    snapshot: WorldSnapshot,
    actions: list[ActionProposal],
    settlement: GameMasterSettlement,
    relationship_projection: RelationshipProjection,
) -> WorldSnapshot:
    """Atomically project one validated round onto a copy of the frozen snapshot.

    Profiles stay fixed across rounds (the layered design never rewrites them),
    so there is no agent-profile projection to apply here: public standing moves
    through ``settlement.power_impacts`` and ``agent_knowledge_updates``, both of
    which this function handles below.
    """
    updated = snapshot.model_copy(deep=True)
    input_version = updated.snapshot_version
    next_round = updated.round_id + 1
    updated.round_id = next_round
    updated.snapshot_version += 1
    updated.state_summary = settlement.state_summary
    updated.world_state = project_world_state(snapshot, settlement, actions)
    from app.simulation.process import apply_issues
    updated.process_issues = apply_issues(snapshot, actions, settlement.issue_updates)
    updated.validation_issues.extend(settlement._process_diagnostics)

    # The strength dimensions are an event-level constant, so they are carried on the
    # snapshot rather than through the world state the settlement replaces.
    if not updated.strength_dimensions:
        updated.strength_dimensions = _strength_dimensions(updated.agents)

    applied_impacts = apply_power_impacts(updated.agents, settlement.power_impacts, next_round)
    # An Agent's reported claims about its own capabilities or constraints are settled
    # once this round's strength verdict actually moved it.
    impacted = {item["agent_id"] for item in applied_impacts}
    if impacted:
        by_id = {agent.agent_id: agent for agent in updated.agents}
        for agent_id in impacted:
            by_id[agent_id].capability_constraint_claims = []

    for item in settlement.public_events:
        event = item.model_copy(deep=True)
        event.origin = "simulated"
        event.round_id = next_round
        updated.timeline.append(event)
    updated.timeline = updated.timeline[-200:]

    relationship_index = {
        (item.source, item.target, item.relation_type): index
        for index, item in enumerate(updated.relationships)
    }
    for item in relationship_projection.relationship_updates:
        relation = item.model_copy(deep=True)
        relation.origin = "simulated"
        key = (relation.source, relation.target, relation.relation_type)
        if key in relationship_index:
            updated.relationships[relationship_index[key]] = _merge_relationship(updated.relationships[relationship_index[key]], relation)
        else:
            relationship_index[key] = len(updated.relationships)
            updated.relationships.append(relation)

    agents = {item.agent_id: item for item in updated.agents}
    for action in actions:
        agent = agents[action.agent_id]
        if agent.decision_core is not None:
            if action.plan_update is not None:
                if not set(action.plan_update.goal_ids) <= {g.desire_id for g in agent.desires}:
                    raise ValueError("Plan update names an unknown goal")
                if not set(action.plan_update.timeline_refs) <= {e.timeline_id for e in snapshot.timeline}:
                    raise ValueError("Plan update cites an unknown event")
                agent.current_plan = action.plan_update.model_copy(update={"updated_round":next_round},deep=True)
        action_record = action.model_dump(exclude={"reasoning", "plan_update"})
        if action.visibility == "public":
            agent.historical_actions.append(action_record)
            agent.historical_actions = agent.historical_actions[-40:]
        else:
            agent.private_action_history.append(action_record)
            agent.private_action_history = agent.private_action_history[-30:]

    for knowledge_update in settlement.agent_knowledge_updates:
        recipient = agents[knowledge_update.agent_id]
        for information in knowledge_update.information:
            recipient.private_observations.append(PrivateObservation(
                observation_id=(
                    "knowledge_"
                    + uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{snapshot.event_id}:{next_round}:{knowledge_update.agent_id}:{information}",
                    ).hex[:16]
                ),
                description=information,
                received_round=next_round,
            ))
        recipient.private_observations = recipient.private_observations[-30:]

    # The claims list had no cap anywhere: an Agent that kept reporting the same
    # complaint grew it without bound, and it is serialised whole into the Agent's own
    # decision prompt every round.
    for agent in updated.agents:
        agent.capability_constraint_claims = agent.capability_constraint_claims[-20:]

    updated.round_manifests.append(RoundManifest(
        round_id=next_round,
        input_snapshot_version=input_version,
        output_snapshot_version=updated.snapshot_version,
        active_agent_ids=[item.agent_id for item in updated.agents if item.activation.status == "active"],
        action_ids=[item.action_id for item in actions],
    ))
    updated.round_manifests = updated.round_manifests[-40:]
    return updated
