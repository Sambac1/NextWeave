"""Cheap, deterministic issue validation and public-state transition tracking."""
import re
from app.domain.process import ProcessIssue

VERSION = 'issue-process-1'


def discard_explicitly_private_events(settlement):
    """Keep explicitly private narration out of the public event channel."""
    retained = []
    for event in settlement.public_events:
        if re.search(r"私下|秘密|保密|未公开|\b(?:privately|nonpublic|behind closed doors|in private|in secret)\b",
                     event.description, re.I):
            settlement._process_diagnostics.append({"code": "private_event_not_published", "event_ref": event.timeline_id})
        else:
            retained.append(event)
    settlement.public_events = retained


def redact_private_causes(snapshot, settlement, actions):
    """A private action may cause a public event; its ID stays out of public views.

    Unknown IDs remain invalid. Public event references, quotations, and states
    are retained and still validated; this never changes action visibility.
    """
    private_ids = {a.action_id for a in actions if a.visibility != 'public'}
    private_ids.update(a.get('action_id') for actor in snapshot.agents for a in actor.private_action_history)
    private_ids.discard(None)
    for item in [*settlement.public_events, *settlement.issue_updates]:
        item.cause_action_ids = [identifier for identifier in item.cause_action_ids if identifier not in private_ids]


def retain_valid_issue_updates(snapshot, settlement, actions):
    """After bounded repair, reject only invalid optional annotations.

    This is called only after core settlement/resource validation has passed.
    Neither before-state nor source refs are guessed or silently repaired.
    """
    from collections import Counter
    counts = Counter(u.issue_id for u in settlement.issue_updates)
    accepted = []
    for update in settlement.issue_updates:
        try:
            if counts[update.issue_id] > 1:
                raise ValueError('Duplicate issue updates are ambiguous')
            validate_issue_updates(snapshot, settlement.model_copy(update={'issue_updates': [update]}), actions)
            accepted.append(update)
        except ValueError as exc:
            settlement._process_diagnostics.append({'code': 'issue_update_rejected', 'round_id': snapshot.round_id + 1,
                                                    'issue_id': update.issue_id, 'reason': str(exc),
                                                    'fallback': 'previous_issue_state_preserved'})
    settlement.issue_updates = accepted

def key(text):
    return re.sub(r'[\W_]+', '', str(text).casefold())


def validate_issue_updates(snapshot, settlement, actions):
    known = {a.agent_id for a in snapshot.agents}
    prior = {i.issue_id: i for i in snapshot.process_issues}
    topics = {i['id']: i['title'] for i in snapshot.world_state.get('process_topics', [])}
    public = {e.timeline_id: e for e in settlement.public_events}
    public_actions = {a.action_id for a in actions if a.visibility == 'public'}
    public_actions.update(a.get('action_id') for actor in snapshot.agents for a in actor.historical_actions)
    seen = set()
    for update in settlement.issue_updates:
        if update.issue_id in seen:
            raise ValueError('Multiple changes to one issue must be consolidated within the round')
        seen.add(update.issue_id)
        if topics and (update.issue_id not in topics or update.title != topics[update.issue_id]):
            raise ValueError('Issue must use its frozen topic ID and title')
        if not set([*update.actor_ids, *update.pending_actor_ids]) <= known:
            raise ValueError('Issue change names an unknown actor')
        if not set(update.event_refs) <= set(public):
            raise ValueError('Issue change must cite this round public events')
        if not any(update.evidence_quote in public[r].description for r in update.event_refs):
            raise ValueError('Issue quote is absent from cited public events')
        if not set(update.cause_action_ids) <= public_actions:
            raise ValueError('Issue change cannot expose unknown or private actions')
        if update.issue_id in prior and update.before != prior[update.issue_id].state:
            raise ValueError('Issue before-state does not match the frozen state')
        if update.issue_id not in prior and update.before:
            raise ValueError('An untracked issue has no established before-state')
        if update.kind != 'unchanged' and key(update.before) == key(update.after):
            raise ValueError('An unchanged state cannot count as a new issue change')
        if update.kind == 'unchanged' and update.before != update.after:
            raise ValueError('An unchanged update cannot silently replace the issue state')


def suppress_continuations(snapshot, settlement, actions, interventions=()):
    """No actor moves + no configured observation is not seven new public events.

    A consequence of a recorded prior action may still occur without a new move.
    Suppressed material remains in the settlement audit, outside the public timeline.
    """
    step = snapshot.round_id + 1
    observations = any(o.get('step') == step for o in snapshot.world_state.get('program_observations', []))
    acted = any(a.decision == 'act' for a in actions) or bool(interventions)
    prior_actions = {a.get('action_id') for actor in snapshot.agents for a in [*actor.historical_actions, *actor.private_action_history]}
    valid = []
    for event in settlement.public_events:
        delayed = bool(event.cause_action_ids) and set(event.cause_action_ids) <= prior_actions
        if event.change_kind == 'continuation' or (not acted and not observations and not delayed):
            settlement.continuing_state_notes.append(event.description)
            if snapshot.world_state.get("simulation_clock", {}).get("sequential_harness"):
                settlement._process_diagnostics.append({"code": "event_reclassified_as_note",
                                                        "event_ref": event.timeline_id})
        else:
            valid.append(event)
    settlement.public_events = valid
    settlement.continuing_state_notes = list(dict.fromkeys(settlement.continuing_state_notes))[:12]


def apply_issues(snapshot, actions, updates):
    issues = {i.issue_id: i.model_copy(deep=True) for i in snapshot.process_issues}
    round_id = snapshot.round_id + 1
    # Deliberation is a review, not proof that a pending request was answered.
    for update in updates:
        prior = issues.get(update.issue_id)
        responses = dict(prior.last_response_round) if prior else {}
        if update.kind == 'response':
            responses.update({a: round_id for a in update.actor_ids})
        issues[update.issue_id] = ProcessIssue(issue_id=update.issue_id, title=update.title,
            state=update.after, actor_ids=update.actor_ids, event_refs=update.event_refs,
            pending_actor_ids=update.pending_actor_ids, updated_round=round_id,
            last_response_round=responses)
    return list(issues.values())


def pending_for(snapshot, actor_id):
    issues = [i.model_dump() for i in snapshot.process_issues if actor_id in i.pending_actor_ids]
    agent = next((agent for agent in snapshot.agents if agent.agent_id == actor_id), None)
    last = agent.activation.last_deliberation_round if agent else None
    on_date = str(snapshot.world_state.get("simulation_clock", {}).get("interval_start") or "")[:10]
    for key, state in snapshot.world_state.get("question_states", {}).items():
        if actor_id not in state.get("pending_actor_ids", []) or state.get("status") in {"met", "unmet"}:
            continue
        decision = next((row for row in state.get("pending_decisions", []) if row["actor_id"] == actor_id), {})
        opened = decision.get("opened_round", state.get("updated_round"))
        if last is not None and isinstance(opened, int) and opened < last:
            continue
        if on_date and decision.get("review_by") and on_date > decision["review_by"]:
            continue
        issues.append({"condition_id": key, **state, "kind": decision.get("kind", "response")})
    return issues


def issue_cards(snapshot):
    return [i.model_dump() for i in snapshot.process_issues]
