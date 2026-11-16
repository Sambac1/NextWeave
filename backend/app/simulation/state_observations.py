"""Validate, retain and resolve observations for a frozen outcome contract."""

from datetime import date
import re

from app.domain.ensemble import TerminalJudgment
from app.simulation.activities import observations
from app.simulation.measurements import at_path, condition_value

SETTLEMENT_MODES = {"standing", "boundary"}
EFFECTIVE_STAGES = {"started", "ongoing", "completed", "paused", "cancelled"}
OPEN_ENDED = re.compile(
    r"\b(?:until (?:it is )?(?:withdrawn|revoked|rescinded|replaced|superseded|further notice)|"
    r"no (?:fixed )?(?:expiry|expiration) date|indefinitely)\b|"
    r"(?:直至|直到|除非).{0,18}(?:撤销|废止|取代|替代)|(?:撤销|废止|取代|替代)前|(?:长期有效|无固定期限|直至另行通知)", re.I,
)


def expiry_in_quote(expiry, quote):
    """Carry only dates explicitly visible in the retained evidence clause."""
    y, m, d = expiry.year, expiry.month, expiry.day
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", quote))
    if years and years != {str(y)}:
        return False
    month = expiry.strftime("%B")
    formats = (expiry.isoformat(), f"{y}年{m}月{d}日", f"{m}月{d}日", f"{month} {d}", f"{d} {month}")
    return any(re.search(rf"(?<![0-9]){re.escape(value)}(?![0-9])", quote, re.I) for value in formats)


def configured_paths(protocol):
    paths = {rate.path for rate in protocol.rates}
    paths.update(path for event in protocol.scheduled_observations for path in event.updates)
    paths.update(effect.path for rule in protocol.action_mechanisms for effect in rule.effects)
    return paths


def validate_measurement_plan(dimensions, world, protocol):
    protected = configured_paths(protocol)
    for dimension in dimensions:
        if dimension.observation_mode == "mechanism":
            for category in dimension.categories:
                for condition in category.conditions:
                    if condition.path not in protected:
                        raise ValueError(f"{dimension.label} references the state field {condition.path}, which has no configured mechanism.")
                    at_path(world, condition.path)


def save_round_observations(frame, dimensions, actor_ids, state):
    """Use the engine's boundary and public evidence, never a model-supplied observation date."""
    tracked = {dimension.id: dimension for dimension in dimensions}
    notes = frame.continuing_state_notes
    events = {event.timeline_id: event.model_dump() for event in frame.timeline}
    private = {action.action_id for action in frame.agent_actions if action.visibility != "public"}
    end = frame.simulation_interval.get("interval_end")
    if not end:
        return {}
    observed_on = date.fromisoformat(end)
    found = {}
    for item in frame.state_observations:
        dimension = tracked.get(item.dimension_id)
        if not dimension or dimension.observation_mode not in SETTLEMENT_MODES:
            continue
        category = next((option for option in dimension.categories if option.id == item.category_id), None)
        if not category and item.category_id != "other":
            continue
        if not set(item.actor_ids) <= actor_ids:
            continue
        event = events.get(item.event_ref)
        if item.event_ref:
            if not event or private.intersection(event.get("cause_action_ids") or []):
                continue
            if item.quote not in event["description"] or not set(item.actor_ids) <= set(event["actors"]):
                continue
            source_text = event
        else:
            note = next((note for note in notes if item.quote in note), None)
            if note is None:
                continue
            source_text = {"description": note}
        anchored_stages = {row["stage"] for row in observations({
            "description": item.quote,
            "activity_annotations": source_text.get("activity_annotations", []),
        })}
        if item.stage and (event or dimension.observation_mode == "standing"):
            if item.stage not in anchored_stages:
                continue
        if category:
            if not set(category.required_actor_ids) <= set(item.actor_ids):
                continue
            stage_options = set(category.evidence_stages)
            if dimension.observation_mode == "standing":
                # Orders that are in effect need a settled execution/status clause.
                # A statement about what may be lifted, promised or considered is not that clause.
                stage_options = stage_options & EFFECTIVE_STAGES if stage_options else EFFECTIVE_STAGES
                if not anchored_stages.intersection(stage_options):
                    continue
            elif stage_options and item.stage not in stage_options:
                continue
        if item.expires_on and (item.expires_on <= observed_on or item.until_superseded):
            continue
        expires_on = None
        carry = False
        if dimension.observation_mode == "standing":
            if item.expires_on and expiry_in_quote(item.expires_on, item.quote):
                expires_on = item.expires_on.isoformat()
                carry = True
            elif item.until_superseded and OPEN_ENDED.search(item.quote):
                carry = True
        row = {
            "category_id": item.category_id, "observed_on": end, "step": frame.step,
            "event_ref": item.event_ref, "quote": item.quote, "actor_ids": item.actor_ids,
            "source": "public_event" if event else "settled_public_state",
            "expires_on": expires_on, "carry": carry,
        }
        old = found.get(dimension.id)
        if old is None:
            found[dimension.id] = row
        elif old["category_id"] != row["category_id"] or any(
            old[key] != row[key] for key in ("expires_on", "carry")
        ):
            found[dimension.id] = {"category_id": "other", "observed_on": end, "step": frame.step,
                                   "source": "conflicting_observations"}
    for dimension in dimensions:
        if dimension.observation_mode != "boundary" or not dimension.issue_id or dimension.id in found:
            continue
        for update in frame.issue_updates:
            if update.issue_id != dimension.issue_id or update.kind == "unchanged":
                continue
            matches = [category for category in dimension.categories
                       if update.state_code in category.issue_state_codes
                       and set(category.required_actor_ids) <= set(update.actor_ids)]
            valid_refs = [ref for ref in update.event_refs if ref in events
                          and not private.intersection(events[ref].get("cause_action_ids") or [])
                          and update.evidence_quote in events[ref]["description"]]
            if len(matches) == 1 and valid_refs:
                found[dimension.id] = {
                    "category_id": matches[0].id, "observed_on": end, "step": frame.step,
                    "event_ref": valid_refs[0], "quote": update.evidence_quote,
                    "actor_ids": update.actor_ids, "source": "public_issue_update", "carry": False,
                }
    for dimension in dimensions:
        if dimension.observation_mode != "mechanism":
            continue
        matches = [category for category in dimension.categories if condition_value(state, category.conditions) is True]
        if len(matches) == 1:
            category = matches[0]
            found[dimension.id] = {
                "category_id": category.id, "observed_on": end, "step": frame.step,
                "source": "configured_mechanism",
                "quote": "，".join(f"{path} = {at_path(state, path)}" for path in
                                 dict.fromkeys(condition.path for condition in category.conditions)),
            }
    return found


def latest_states(frames):
    latest = {}
    for frame in frames:
        latest.update(frame.get("measurement_states", {}))
    return latest


def resolve_terminal_states(record, dimensions):
    frames = record.get("rounds", [])
    final = frames[-1] if frames else {}
    end = final.get("simulation_interval", {}).get("interval_end")
    latest = latest_states(frames)
    judgments = []
    for dimension in dimensions:
        if dimension.observation_mode == "legacy":
            continue
        row = latest.get(dimension.id, {})
        carried = (end and row.get("carry") and
                   (not row.get("expires_on") or row["expires_on"] > end))
        current = end and row.get("observed_on") == end and row.get("step") == final.get("step")
        valid = (dimension.observation_mode != "unavailable" and (current or carried) and
                 row.get("category_id") in {category.id for category in dimension.categories})
        if valid:
            source = {
                "public_event": "public event",
                "settled_public_state": "public state from round settlement",
                "configured_mechanism": "configured scenario mechanism",
                "public_issue_update": "issue state supported by a public event",
            }[row["source"]]
            judgments.append(TerminalJudgment(
                dimension_id=dimension.id, category_id=row["category_id"],
                matched_category_ids=[row["category_id"]],
                event_refs=[row["event_ref"]] if row.get("event_ref") else [],
                evidence_quote=row.get("quote", "")[:1200],
                reason=f"Based on a {source} observed on {row['observed_on']}" + (". This state remains valid through the final interval." if not current else "."),
            ))
        else:
            reason = ("The simulation has no observation or mechanism for this variable." if dimension.observation_mode == "unavailable"
                      else "No valid observation meets the scope and time requirements.")
            judgments.append(TerminalJudgment(dimension_id=dimension.id, category_id="other", reason=reason))
    return judgments


def measurement_preflight(records, dimensions):
    required = {dimension.id: dimension.label for dimension in dimensions
                if dimension.observation_mode in SETTLEMENT_MODES | {"mechanism"} and dimension.required_for_sampling}
    usable = [record for record in records if record.get("status") in {"sampled", "completed"}]
    observed = {claim.dimension_id for record in usable for claim in resolve_terminal_states(record, dimensions)
                if claim.category_id != "other"}
    missing = [label for key, label in required.items() if key not in observed]
    return {"passed": bool(usable) and not missing, "pilot_samples": len(usable),
            "attempted_samples": len(records), "missing_dimensions": missing}
