"""Bounded annotation repair; sampled decisions and public facts stay immutable."""
from collections import Counter
import re

from app.domain.forecast import PendingDecision, QuestionUpdate
from app.domain.identity import leading_actor_span
from app.knowledge.evidence import date_parts, locate_quote
from app.knowledge.retrieval import _named_in
from app.knowledge.sentences import sentence_records
from app.simulation.joint_settlement import (
    enabled, intervention_actions, validate_interaction, validate_joint_settlement, validate_question_update,
)

_DELIVERED_REQUEST = re.compile(
    r"\b(?:sent|delivered|submitted)\b[^.;]{0,120}\b(?:request|offer|invitation|proposal|application|notice)\b"
    r"|\bissued (?:an?|the) (?:formal )?(?:request|invitation|notice)\b"
    r"|\b(?:asked|invited|called on|requested)\b"
    r"|已送达|已提交|发出邀请|提出请求|正式要求",
    re.I,
)
_CONDITIONAL_OFFER = re.compile(
    r"\b(?:conditional|contingent|provided that|subject to|if|offer(?:ed)?|proposal)\b"
    r"|有条件|如果|只要|取决于|提议",
    re.I,
)


def _recipient_names(actor, agents):
    names = [actor.name, *actor.aliases]
    parts = actor.name.split()
    if actor.agent_type == "person" and len(parts) > 1:
        surname = parts[-1]
        title = re.split(r"\s+(?:of|at|for)\s+|,", actor.title, maxsplit=1, flags=re.I)[0].strip()
        if title and not any(
            other.agent_id != actor.agent_id and
            any(name.split() and name.split()[-1].casefold() == surname.casefold()
                for name in [other.name, *other.aliases])
            for other in agents
        ):
            names.append(f"{title} {surname}")
    return [name for name in names if name]


def _subject_span(snapshot, event, owner):
    """Only reconstruct a leading subject, never a target mentioned in prose."""
    start = leading_actor_span(event.description, snapshot.agents, owner)
    if start is None:
        return None
    text = event.description[start:]
    if len(text) <= 500:
        return text if len(text) >= 8 else None
    # Use a complete sentence/clause rather than cutting a quote mid-word.
    boundaries = [match.end() for match in re.finditer(r'[.!?;](?=\s|$)', text[:501]) if match.end() >= 8]
    return text[:boundaries[0]] if boundaries else None


def normalize_joint_settlement(snapshot, settlement, actions, interventions=()):
    if not enabled(snapshot):
        return
    selected = {action.action_id: action for action in [*actions, *intervention_actions(snapshot, interventions)]}
    historical = {row["action_id"]: row for actor in snapshot.agents
                  for row in [*actor.historical_actions, *actor.private_action_history] if row.get("action_id")}
    events = {event.timeline_id: event for event in settlement.public_events}
    knowledge = {row.agent_id: row.information for row in settlement.agent_knowledge_updates}

    def record(code, **fields):
        settlement._process_diagnostics.append({"code": code, "round_id": snapshot.round_id + 1, **fields})

    for event in settlement.public_events:
        for execution in event.executed_decisions:
            source = selected.get(execution.action_id)
            old = historical.get(execution.action_id, {})
            owner = source.agent_id if source else old.get("agent_id")
            mode = source.decision if source else old.get("decision")
            linked = execution.action_id in event.cause_action_ids or any(
                row.action_id == execution.action_id and event.timeline_id in row.event_refs
                for row in settlement.action_results)
            if owner != execution.actor_id or mode != "act" or not linked:
                continue  # Core validation still rejects unauthorized execution.
            if execution.quote in event.description and leading_actor_span(execution.quote, snapshot.agents, owner) is not None:
                continue
            if quote := _subject_span(snapshot, event, owner):
                execution.quote = quote
                record("execution_quote_span_repaired", event_ref=event.timeline_id,
                       actor_id=owner, action_id=execution.action_id)

    for row in settlement.action_results:
        source = selected.get(row.action_id)
        if source is None:
            continue
        lost_private_proof = False
        if quote := row.private_observation_quote:
            delivered = [text for actor_id in {source.agent_id, source.target}
                         for text in knowledge.get(actor_id, [])]
            if not any(quote in text for text in delivered):
                matches = {text[span[0]:span[1]] for text in delivered if (span := locate_quote(text, quote))
                           and 8 <= span[1] - span[0] <= 400}
                row.private_observation_quote = next(iter(matches)) if len(matches) == 1 else ""
                lost_private_proof = not row.private_observation_quote
                record("private_action_quote_unverified" if lost_private_proof else "private_action_quote_repaired",
                       action_id=row.action_id)
        if (source.decision != "act" or row.status not in {"executed", "partial"}
                or row.event_refs or row.private_observation_quote):
            continue
        refs = [event.timeline_id for event in settlement.public_events
                if row.action_id in event.cause_action_ids and any(
                    execution.action_id == row.action_id and execution.actor_id == source.agent_id
                    for execution in event.executed_decisions)]
        if 0 < len(refs) <= 4:
            row.event_refs = refs
            record("action_result_refs_repaired", action_id=row.action_id)
        elif lost_private_proof:
            row.status = "unknown"
            row.summary = "No verified consequence was recorded for this action."
            record("action_result_marked_unknown", action_id=row.action_id)

    # Optional labels may never hide invalid identity, authority or execution.
    validate_joint_settlement(snapshot, settlement, actions, interventions, annotations=False)
    valid_interactions = []
    for interaction in settlement.interactions:
        try:
            validate_interaction(interaction, events, knowledge, selected, set(selected) | set(historical))
            valid_interactions.append(interaction)
        except ValueError as exc:
            record("interaction_annotation_rejected", reason=str(exc), fallback="core_settlement_retained")
    settlement.interactions = valid_interactions

    prior = snapshot.world_state.get("question_states", {})
    counts = Counter(row.condition_id for row in settlement.question_updates)
    definitions = [row["id"] for row in snapshot.world_state.get("question_conditions", [])]
    proposed = {row.condition_id: row for row in settlement.question_updates}
    for key in proposed.keys() - set(definitions):
        record("question_annotation_rejected", condition_id=key, reason="unknown_condition")
    normalized = []
    for key in definitions:
        before = prior.get(key, {})
        row = proposed.get(key)
        try:
            if row is None or counts[key] != 1:
                raise ValueError("Missing or ambiguous question annotation")
            # Missing evidence is an observer concern, not an actor's duty to
            # answer. New pending actors need a typed, cited decision trigger.
            old_pending = before.get("pending_actor_ids", [])
            accepted_pending = []
            rejected = []
            actors = {actor.agent_id: actor for actor in snapshot.agents}
            counts_by_actor = Counter(pending.actor_id for pending in row.pending_decisions)
            for pending in row.pending_decisions:
                if counts_by_actor[pending.actor_id] != 1:
                    rejected.append("duplicate_pending_decision")
                    continue
                actor = actors.get(pending.actor_id)
                event = events.get(pending.event_ref)
                if not actor or not event or pending.event_ref not in row.event_refs:
                    rejected.append("unregistered_actor_or_unlinked_event")
                    continue
                if pending.evidence_quote not in event.description:
                    rejected.append("quote_not_in_public_event")
                    continue
                names = _recipient_names(actor, snapshot.agents)
                if not any(_named_in(name, pending.evidence_quote) for name in names):
                    body = event.description
                    sentence = next((body[span["start"]:span["end"]] for span in sentence_records(body)
                                     if pending.evidence_quote in body[span["start"]:span["end"]]
                                     and span["end"] - span["start"] <= 500
                                     and any(_named_in(name, body[span["start"]:span["end"]]) for name in names)), None)
                    if not sentence:
                        rejected.append("recipient_missing_from_quoted_clause")
                        continue
                    pending = pending.model_copy(update={"evidence_quote": sentence})
                    record("pending_quote_span_repaired", condition_id=key)
                if pending.kind == "opportunity" and not pending.choice.strip():
                    rejected.append("opportunity_missing_choice")
                    continue
                if pending.review_by and pending.review_by not in date_parts(pending.evidence_quote):
                    pending = pending.model_copy(update={"review_by": ""})
                    record("pending_deadline_unverified", condition_id=key)
                if pending.kind == "decision_window" and pending.choice and not pending.review_by:
                    pending = pending.model_copy(update={"kind": "opportunity"})
                if (pending.kind == "response" and pending.choice
                        and not _DELIVERED_REQUEST.search(pending.evidence_quote)):
                    pending = pending.model_copy(update={"kind": "opportunity"})
                    record("pending_response_downgraded", condition_id=key)
                if pending.kind == "opportunity":
                    if not _CONDITIONAL_OFFER.search(pending.evidence_quote):
                        rejected.append("no_public_evidence_of_opportunity")
                        continue
                elif pending.actor_id not in event.actors and not any(
                    selected[action_id].target == pending.actor_id and selected[action_id].visibility == "public"
                    for action_id in event.cause_action_ids if action_id in selected
                ):
                    rejected.append("recipient_not_linked_to_recorded_action")
                    continue
                accepted_pending.append(pending)
            # An untyped follow-up can be repaired only as an optional review.
            # The event's executor list need not include its named counterpart.
            for actor_id in dict.fromkeys(row.pending_actor_ids):
                if actor_id not in actors or actor_id in old_pending or counts_by_actor[actor_id]:
                    continue
                names = _recipient_names(actors[actor_id], snapshot.agents)
                matches = [event for ref in row.event_refs if (event := events.get(ref))
                           and 8 <= len(event.description) <= 500
                           and _CONDITIONAL_OFFER.search(event.description)
                           and any(_named_in(name, event.description) for name in names)]
                if len(matches) == 1:
                    event = matches[0]
                    accepted_pending.append(PendingDecision(actor_id=actor_id, kind="opportunity",
                        event_ref=event.timeline_id, evidence_quote=event.description,
                        choice="Review the public terms and decide whether to follow up."))
                    record("pending_opportunity_recovered", condition_id=key)
            supported = {pending.actor_id for pending in accepted_pending}
            pending_ids = list(dict.fromkeys(actor_id for actor_id in row.pending_actor_ids
                                            if actor_id in old_pending or actor_id in supported))
            pending_ids.extend(pending.actor_id for pending in accepted_pending if pending.actor_id not in pending_ids)
            if row.after == row.before and not row.event_refs:
                pending_ids = list(old_pending)
                accepted_pending = []
            if rejected or set(row.pending_actor_ids) - set(pending_ids):
                record("pending_annotation_rejected", condition_id=key,
                       reasons=sorted(set(rejected or ["missing_cited_pending_decision"])),
                       fallback="validated_pending_actors_only")
            row = row.model_copy(update={"pending_actor_ids": pending_ids, "pending_decisions": accepted_pending}, deep=True)
            validate_question_update(snapshot, row, events)
        except ValueError as exc:
            record("question_annotation_rejected", condition_id=key, reason=str(exc), fallback="previous_condition_preserved")
            row = QuestionUpdate(condition_id=key, before=before.get("status", "unknown"),
                                 after=before.get("status", "unknown"), summary="No new validated public evidence.",
                                 pending_actor_ids=list(before.get("pending_actor_ids", [])))
        normalized.append(row)
    settlement.question_updates = normalized
    validate_joint_settlement(snapshot, settlement, actions, interventions)


def state_change_repair_context(snapshot, settlement, error):
    """Expose the authoritative leaf and current evidence for a rejected write."""
    label = str(error).rpartition(": ")[2]
    if not hasattr(settlement, "state_changes"):
        return None
    change = next((item for item in settlement.state_changes if ".".join(item.path) == label), None)
    if change is None:
        return None
    parent = snapshot.world_state
    for index, part in enumerate(change.path[:-1]):
        if part not in parent or not isinstance(parent[part], dict):
            parent_status = "absent" if part not in parent else "not_an_object"
            break
        parent = parent[part]
    else:
        index, parent_status = len(change.path) - 1, "object"
    key = change.path[-1]
    result = {"path": change.path, "parent_status": parent_status,
              "checked_path": change.path[:index + 1], "leaf_exists": parent_status == "object" and key in parent}
    if result["leaf_exists"]:
        value = parent[key]
        if len(str(value)) <= 1000:
            result["current_value"] = value
    if event := next((row for row in settlement.public_events if row.timeline_id == change.event_ref), None):
        result["retained_event_text"] = event.description[:1000]
    return result

