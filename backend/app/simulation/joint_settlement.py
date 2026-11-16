"""Contracts for joint action resolution and persistent public question conditions."""
from copy import deepcopy
import re

VERSION = "joint-settlement-1"


def enabled(snapshot):
    return snapshot.world_state.get("simulation_clock", {}).get("interaction_protocol") == VERSION


def condition_definitions(contract):
    if not contract:
        return []
    spec = contract.get("spec", {})
    labels = [("target_event", contract.get("question", "Whether the requested outcome occurs"))]
    if spec.get("predicate") in {"rank", "threshold"}:
        labels = [("participation", f"Participation of {spec.get('subject') or 'the target subject'}"),
                  ("outcome_measure", "Available evidence about the requested count, rank, or threshold")]
    if criteria := contract.get("forecast_criteria"):
        if spec.get("predicate") not in {"rank", "threshold"}:
            labels[0] = ("target_event", criteria["yes"])
        labels.append(("negative_outcome", criteria["no"]))
    labels.extend((key, f"Required observation: {key.replace('_', ' ')}")
                  for key in spec.get("required_observations", []))
    return [{"id": key, "description": label} for key, label in labels][:8]


def question_rows(snapshot):
    stored = snapshot.world_state.get("question_states", {})
    return [{**definition, **stored.get(definition["id"], {"status": "unknown"})}
            for definition in snapshot.world_state.get("question_conditions", [])]


def intervention_actions(snapshot, interventions):
    from app.domain.forecast import ActionProposal
    names = {actor.agent_id: actor.name for actor in snapshot.agents}
    return [ActionProposal(
        action_id=f"intervention_{snapshot.round_id + 1}_{index:02d}", agent_id=item.agent_id,
        agent_name=names.get(item.agent_id, item.agent_name), round_id=snapshot.round_id + 1,
        snapshot_version=snapshot.snapshot_version, decision="act", action_type="user_intervention",
        action=item.action, visibility=item.visibility, target=item.target,
        reasoning="User-specified intervention", confidence=1,
    ) for index, item in enumerate(interventions, 1)]


def recorded_causal_actions(snapshot):
    """Compact GM-only registry, including private causes of delayed effects.

    Histories are bounded by the round transition. This projection omits BDI,
    reasoning and outcomes: a recorded attempt is never proof of success.
    """
    fields = {"action_id", "round_id", "action", "action_type", "target", "visibility", "condition", "commitment"}
    return [{**{key: value for key, value in row.items() if key in fields},
             "agent_id": actor.agent_id, "agent_name": actor.name}
            for actor in snapshot.agents for row in [*actor.historical_actions, *actor.private_action_history]
            if row.get("action_id") and row.get("decision") == "act"]


def round_briefing(snapshot):
    clock = snapshot.world_state.get("simulation_clock", {})
    contract = snapshot.world_state.get("benchmark_contract", {})
    recent = [event for event in snapshot.timeline if event.origin == "simulated" and event.round_id == snapshot.round_id]
    return {"interval_start": clock.get("interval_start_utc", clock.get("interval_start")),
            "interval_end": clock.get("interval_end_utc", clock.get("interval_end")),
            "simulation_horizon": clock.get("horizon_end_utc", clock.get("horizon_end")), "horizon_basis": contract.get("deadline_basis"),
            "question": contract.get("question", snapshot.event_title), "question_conditions": question_rows(snapshot),
            "latest_public_changes": [{"event_ref": event.timeline_id, "text": event.description} for event in recent],
            "open_requests": [issue.model_dump() for issue in snapshot.process_issues if issue.pending_actor_ids],
            "state_location": "world_state contains the current durable public state; timeline contains historical evidence"}


def validate_interaction(interaction, events, knowledge, selected, available):
    refs = set(interaction.action_ids)
    if len(refs) != len(interaction.action_ids) or len(refs) < 2 or not refs <= available or not refs.intersection(selected):
        raise ValueError("Interaction must name distinct recorded actions, including a selected action")
    if not set(interaction.event_refs) <= events.keys():
        raise ValueError("Interaction cites an unavailable public event")
    if interaction.evidence_quote and not any(
        interaction.evidence_quote in events[ref].description for ref in interaction.event_refs
    ):
        raise ValueError("Interaction public quote is absent from its cited event")
    if interaction.outcome == "realized":
        public_effect = len(interaction.evidence_quote.strip()) >= 8 and any(
            interaction.evidence_quote in events[ref].description for ref in interaction.event_refs
        )
        recipients = {value for ref in refs if ref in selected
                      for value in (selected[ref].agent_id, selected[ref].target)}
        private_effect = bool(interaction.private_observation_quote) and any(
            interaction.private_observation_quote in text for actor_id in recipients for text in knowledge.get(actor_id, [])
        )
        if not public_effect and not private_effect:
            raise ValueError("Realized interaction needs a quoted public or delivered private consequence")
        if interaction.available_at == "new_this_round":
            raise ValueError("A newly learned decision must remain pending until its recipient can deliberate")
        if interaction.available_at == "precommitted" and not any(
            getattr(selected.get(ref), "condition", "") for ref in refs
        ):
            raise ValueError("Precommitted interaction needs a selected conditional action")


def validate_question_update(snapshot, update, events):
    prior = snapshot.world_state.get("question_states", {})
    known = {actor.agent_id for actor in snapshot.agents}
    if update.before != prior.get(update.condition_id, {}).get("status", "unknown"):
        raise ValueError("Question condition before-state differs from the frozen state")
    if not set(update.pending_actor_ids) <= known or not set(update.event_refs) <= events.keys():
        raise ValueError("Question update has an unknown actor or event")
    if update.evidence_quote and not any(update.evidence_quote in events[ref].description for ref in update.event_refs):
        raise ValueError("Question update public quote is absent from its cited event")
    if update.after != update.before or update.event_refs:
        if len(update.evidence_quote.strip()) < 8 or not any(
            update.evidence_quote in events[ref].description for ref in update.event_refs
        ):
            raise ValueError("Question condition change needs a quoted public event")
    if update.after == update.before and not update.event_refs:
        if update.pending_actor_ids != prior.get(update.condition_id, {}).get("pending_actor_ids", []):
            raise ValueError("New pending decisions need public evidence")
    seen = set()
    for pending in update.pending_decisions:
        if pending.actor_id in seen or pending.actor_id not in update.pending_actor_ids:
            raise ValueError("Pending decision must name one pending actor exactly once")
        seen.add(pending.actor_id)
        if pending.event_ref not in update.event_refs or pending.evidence_quote not in events[pending.event_ref].description:
            raise ValueError("Pending decision needs a quote from a cited public event")


def validate_joint_settlement(snapshot, settlement, actions, interventions=(), *, annotations=True):
    from app.domain.identity import leading_actor_span
    if not enabled(snapshot):
        return
    selected = {action.action_id: action for action in actions}
    selected.update({action.action_id: action for action in intervention_actions(snapshot, interventions)})
    ids = [row.action_id for row in settlement.action_results]
    if len(ids) != len(set(ids)) or set(ids) != set(selected):
        raise ValueError("action_results must cover every selected action exactly once")
    events = {event.timeline_id: event for event in settlement.public_events}
    knowledge = {row.agent_id: row.information for row in settlement.agent_knowledge_updates}

    for row in settlement.action_results:
        action = selected[row.action_id]
        if not set(row.event_refs) <= events.keys():
            raise ValueError("Action result cites an unavailable public event")
        if getattr(action, "decision", "act") != "act" and row.status != "waiting":
            raise ValueError("A wait or monitor action must have a waiting result")
        if row.status == "waiting" and any(row.action_id in events[ref].cause_action_ids for ref in row.event_refs):
            raise ValueError("Waiting cannot cause a new outward action")
        if row.private_observation_quote:
            recipients = {action.agent_id, action.target}
            if not any(row.private_observation_quote in text for actor_id in recipients for text in knowledge.get(actor_id, [])):
                raise ValueError("Private action result must quote information delivered to its participant")
        if row.status in {"executed", "partial"} and not row.event_refs and not row.private_observation_quote:
            raise ValueError("Executed action needs a public consequence or a delivered private observation")

    historical_actions = {row.get("action_id"): row for actor in snapshot.agents
                          for row in [*actor.historical_actions, *actor.private_action_history] if row.get("action_id")}
    historical = set(historical_actions)
    available = set(selected) | historical
    wait_ids = {key for key, action in selected.items() if getattr(action, "decision", "act") != "act"}
    represented = {ref for row in settlement.action_results for ref in row.event_refs}
    for event in settlement.public_events:
        if not set(event.cause_action_ids) <= available or wait_ids.intersection(event.cause_action_ids):
            raise ValueError("Public effect cites an unknown action or a waiting action")
        if event.timeline_id not in represented and not event.cause_action_ids:
            raise ValueError("A model-generated public event needs a recorded selected or prior cause")
        if not event.executed_decisions:
            raise ValueError("Public event needs explicit execution records for its recorded causal actions")
        for execution in event.executed_decisions:
            source = selected.get(execution.action_id)
            previous = historical_actions.get(execution.action_id, {})
            owner = source.agent_id if source is not None else previous.get("agent_id")
            decision = getattr(source, "decision", "act") if source is not None else previous.get("decision")
            if owner != execution.actor_id or decision != "act":
                raise ValueError("Event executor must match a recorded acting subject, never an unselected responder")
            if execution.quote not in event.description:
                raise ValueError("Execution quote must be copied verbatim from its public event")
            if leading_actor_span(execution.quote, snapshot.agents, owner) is None:
                raise ValueError("Execution quote must identify its registered subject at the start of the quoted action")
            if execution.action_id not in event.cause_action_ids and event.timeline_id not in {
                ref for row in settlement.action_results if row.action_id == execution.action_id for ref in row.event_refs
            }:
                raise ValueError("Execution must belong to the action result or recorded cause of this event")
    for row in settlement.action_results:
        if row.status in {"executed", "partial"} and row.event_refs and not row.private_observation_quote:
            if not any(execution.action_id == row.action_id for ref in row.event_refs
                       for execution in events[ref].executed_decisions):
                raise ValueError("Executed action result needs its own execution in the cited public consequence")
    if not annotations:
        return
    for interaction in settlement.interactions:
        validate_interaction(interaction, events, knowledge, selected, available)

    definitions = {row["id"] for row in snapshot.world_state.get("question_conditions", [])}
    seen = set()
    for update in settlement.question_updates:
        if update.condition_id not in definitions or update.condition_id in seen:
            raise ValueError("Question update names an unknown or duplicate condition")
        seen.add(update.condition_id)
        validate_question_update(snapshot, update, events)
    if seen != definitions:
        raise ValueError("question_updates must cover every tracked condition exactly once")


def apply_question_updates(snapshot, settlement):
    states = deepcopy(snapshot.world_state.get("question_states", {}))
    if enabled(snapshot):
        for update in settlement.question_updates:
            # Without new evidence, an unchanged annotation cannot replace the
            # validated public account or reset its observation time.
            if update.after == update.before and not update.event_refs:
                continue
            prior = states.get(update.condition_id, {})
            pending = {row["actor_id"]: dict(row) for row in prior.get("pending_decisions", [])
                       if row["actor_id"] in update.pending_actor_ids}
            for decision in update.pending_decisions:
                previous = pending.get(decision.actor_id, {})
                opened = previous.get("opened_round", prior.get("updated_round")) if previous.get("event_ref") == decision.event_ref else None
                pending[decision.actor_id] = decision.model_dump() | {"opened_round": opened if isinstance(opened, int) else snapshot.round_id + 1}
            states[update.condition_id] = {"status": update.after, "summary": update.evidence_quote,
                "event_refs": list(update.event_refs), "evidence_quote": update.evidence_quote,
                "pending_actor_ids": list(update.pending_actor_ids),
                "pending_decisions": list(pending.values()),
                "updated_round": snapshot.round_id + 1}
    return states


def public_resolution(frame, definitions=(), public_registry=None):
    labels = {row["id"]: row["description"] for row in definitions}
    public_actions = {row["action_id"] for row in frame.get("agent_actions", [])
                      if row.get("action_id") and row.get("visibility", "public") == "public"}
    current = [*frame.get("agent_actions", []), *frame.get("intervention_actions", [])]
    actor_names = {row["action_id"]: row.get("agent_name", "") for row in current if row.get("action_id")}
    if public_registry is not None:
        public_actions = set(public_registry)
        actor_names.update(public_registry)
    events = {row["timeline_id"]: row["description"] for row in frame.get("timeline", [])}
    return {
        "action_results": [
            {"action_id": row["action_id"], "actor_name": actor_names[row["action_id"]],
             "status": row["status"] if row.get("event_refs") else
             ("waiting" if row["status"] == "waiting" else "unknown"),
             "summary": " ".join(events[ref] for ref in row.get("event_refs", []) if ref in events)
                        or "No public consequence was recorded."}
            for row in frame.get("action_results", []) if row["action_id"] in public_actions
        ],
        "interactions": [
            {"kind": row["kind"], "action_ids": row["action_ids"], "outcome": row["outcome"],
             "effect": row["evidence_quote"]}
            for row in frame.get("interactions", []) if set(row["action_ids"]) <= public_actions
            and len(row.get("evidence_quote", "")) >= 8 and any(
                row["evidence_quote"] in events.get(ref, "") for ref in row.get("event_refs", [])
            )
        ],
        "question_updates": [
            {"condition_id": row["condition_id"], "label": labels.get(row["condition_id"], row["condition_id"].replace("_", " ")),
             "before": row["before"], "after": row["after"],
             "summary": row["evidence_quote"] if len(row.get("evidence_quote", "")) >= 8 and any(
                 row["evidence_quote"] in events.get(ref, "") for ref in row.get("event_refs", [])
             ) else "No new public evidence.", "pending_actor_ids": row["pending_actor_ids"]}
            for row in frame.get("question_updates", [])
        ],
    }


def redact_private_executions(snapshot, settlement, actions):
    public = {action.action_id for action in actions if action.visibility == "public"}
    public.update(row["action_id"] for actor in snapshot.agents for row in actor.historical_actions if row.get("action_id"))
    for event in settlement.public_events:
        event.executed_decisions = [row for row in event.executed_decisions if row.action_id in public]
