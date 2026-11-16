"""Evidence and coverage checks for scenario-defined terminal dimensions."""

import re

from app.domain.ensemble import TerminalJudgment
from app.simulation.activities import observations

VERSION = "terminal-outcomes-2"


def public_measurement_events(record):
    """stored logs may still attach a direct/private action to a public event."""
    rounds = record.get("rounds", [])
    private_ids = {action["action_id"] for frame in rounds
                   for action in [*frame.get("agent_actions", []), *frame.get("intervention_actions", [])]
                   if action.get("action_id") and action.get("visibility", "public") != "public"}
    return {event["timeline_id"]: (frame, event) for frame in rounds for event in frame.get("timeline", [])
            if not private_ids.intersection(event.get("cause_action_ids") or [])}


def terminal_state_notes(record):
    """Read existing public boundary notes without turning them into timeline events."""
    frames = record.get("rounds", [])
    if not frames:
        return {}
    return {f"state_note_{index}": {"description": note, "actors": [], "source": "saved_boundary_note"}
            for index, note in enumerate(frames[-1].get("continuing_state_notes", []), 1)
            if isinstance(note, str) and not re.search(
                r"私下|秘密|保密|未公开|不公开|\b(?:privately|private|secret|confidential|nonpublic)\b", note, re.I,
            )}


def validate_judgments(dimensions, judgments, record):
    rounds = record.get("rounds", [])
    terminal = rounds[-1] if rounds else {}
    events = {ref: event for ref, (frame, event) in public_measurement_events(record).items()
              if frame is terminal}
    events.update(terminal_state_notes(record))
    results = []
    for dimension in dimensions:
        choices = [judgment for judgment in judgments if judgment.dimension_id == dimension.id]
        reason = "The outcome lacks verifiable evidence from the final interval."
        if len(choices) == 1:
            claim = choices[0]
            category = next((option for option in dimension.categories if option.id == claim.category_id), None)
            refs = list(dict.fromkeys(claim.event_refs))
            quote = claim.evidence_quote.strip()
            if claim.category_id == "other":
                reason = claim.reason or reason
            elif category and set(claim.matched_category_ids) == {category.id} and refs and set(refs) <= events.keys():
                cited = [events[ref] for ref in refs]
                anchored = [event for event in cited if len(quote) >= 8 and quote in event.get("description", "")]
                actors = {actor for event in anchored for actor in event.get("actors", [])}
                annotations = [annotation for event in anchored for annotation in observations(event)]
                stage_matches = not category.evidence_stages or any(
                    item["stage"] in category.evidence_stages
                    and (quote in item["quote"] or item["quote"] in quote)
                    for item in annotations
                )
                if anchored and set(category.required_actor_ids) <= actors and stage_matches:
                    results.append(claim.model_copy(update={"event_refs": refs, "evidence_quote": quote}))
                    continue
        results.append(TerminalJudgment(dimension_id=dimension.id, category_id="other", reason=reason))
    return results


def summarize_dimensions(dimensions, completed, planned, interval):
    output = []
    for dimension in dimensions:
        ids_by_category = {category.id: [] for category in dimension.categories}
        unknown = []
        for sample in completed:
            claim = next((item for item in (sample.get("classification") or {}).get("terminal_judgments", [])
                          if item["dimension_id"] == dimension.id), {})
            ids_by_category.get(claim.get("category_id"), unknown).append(sample["sample_id"])
        assessable = len(completed) - len(unknown)
        output.append({
            "id": dimension.id, "label": dimension.label, "assessable_samples": assessable,
            "observation_mode": dimension.observation_mode, "observation_scope": dimension.observation_scope,
            "unknown_sample_ids": unknown, "unclear_count": len(unknown), "unfinished_count": planned-len(completed),
            "categories": [
                category.model_dump() | {
                    "count": len(ids_by_category[category.id]), "sample_ids": ids_by_category[category.id],
                    "frequency": len(ids_by_category[category.id])/assessable if assessable else None,
                    "sampling_interval_95": interval(len(ids_by_category[category.id]), assessable)
                    if assessable and assessable == planned else None,
                } for category in dimension.categories
            ],
        })
    return output
