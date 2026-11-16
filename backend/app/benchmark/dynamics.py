"""Cutoff-safe contract drivers shared by the benchmark simulation arms."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.knowledge.wikipedia import utc_timestamp
from app.benchmark.contract_mechanisms import compile_spec
from app.benchmark.seat_projection import build_projection, draw_count
from app.knowledge.sentences import sentence_records
from app.knowledge.public_evidence import packet_manifest, public_sources

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE = rf"(?:{_MONTH} \d{{1,2}},? \d{{4}}|\d{{1,2}} {_MONTH} \d{{4}})"
_ET_DEADLINE = re.compile(
    rf"\b(by|through|until)\s+({_DATE}),?\s+(\d{{1,2}}:\d{{2}}\s*[AP]M)\s+(?:ET|Eastern Time)\b", re.I,
)
_YES_RULE = re.compile(r"""\b(?:resolve[sd]?|settle[sd]?)\s+(?:to\s+)?["“']?yes\b""", re.I)
_WINDOW_START = re.compile(
    rf"\b(on or after|starting(?: on)?|beginning(?: on)?|between|from)\s+({_DATE}|\d{{4}}-\d{{2}}-\d{{2}})"
    r"(?:,?\s+(\d{1,2}:\d{2}\s*[AP]M)\s+(ET|Eastern Time|UTC))?", re.I,
)
_MARKET_WINDOW = re.compile(r"\b(?:market['’]s\s+time\s*frame|since (?:the )?market (?:opened|was created))\b", re.I)
_ELECTION_DATE = re.compile(rf"\belections?\s+(?:is|are) scheduled[^.\n]{{0,160}}?\bon\s+({_DATE})", re.I)
_SEAT_OUTCOME = re.compile(r"\b(?:seats?|seat count)\b", re.I)
_SCHEDULED_EVENT = r"(?:elections?|runoffs?|votes?|meetings?|hearings?|summits?|visits?|launch(?:es)?|releases?)"
_SCHEDULED_DATE = rf"(?:{_DATE}|\d{{4}}-\d{{2}}-\d{{2}})"
_ROUND_DATES = re.compile(
    rf"\b{_SCHEDULED_EVENT}\s+(?:(?:is|are|was|were)\s+)?(?:scheduled|planned|set)"
    rf"[^.;\n]{{0,90}}?\b(?:on|for)\s+({_SCHEDULED_DATE})\b|"
    rf"\b({_SCHEDULED_DATE})\s+(?:(?:monetary policy|presidential|parliamentary|board|court)\s+)?{_SCHEDULED_EVENT}\b",
    re.I,
)


def _calendar_date(value: str) -> datetime:
    return datetime.strptime(value.replace(",", ""), "%B %d %Y" if value[0].isalpha() else "%d %B %Y")


def outcome_window(rules: str, market: dict | None = None) -> dict:
    """Read only start boundaries supported by the rules and the frozen market record."""
    statements = [row["quote"] for row in sentence_records(rules)]
    if not statements:
        statements = [rules]
    starts = set()
    for statement in statements:
        if not _YES_RULE.search(statement):
            continue
        for match in _WINDOW_START.finditer(statement):
            modifier, raw, raw_time, zone = match.groups()
            connector = {"between": r"\s+and\b", "from": r"\s+(?:to|through|until)\b"}.get(modifier.casefold())
            if connector and not re.match(connector, statement[match.end():], re.I):
                continue
            try:
                day = date.fromisoformat(raw) if "-" in raw else _calendar_date(raw).date()
                if raw_time:
                    local_time = datetime.strptime(raw_time.upper().replace(" ", ""), "%I:%M%p").time()
                    tz = timezone.utc if zone.upper() == "UTC" else ZoneInfo("America/New_York")
                    starts.add(datetime.combine(day, local_time, tz).astimezone(timezone.utc).isoformat())
                else:
                    starts.add(day.isoformat())
            except ValueError:
                continue
    if len(starts) == 1:
        return {"outcome_window_start": starts.pop(), "outcome_window_basis": "explicit_contract_start"}
    if not starts and market and _MARKET_WINDOW.search(rules):
        try:
            start = utc_timestamp(market.get("startDate"), "market startDate")
        except ValueError:
            return {}
        return {"outcome_window_start": start.isoformat(), "outcome_window_basis": "rule_uses_market_time_frame"}
    return {}


def scheduled_vote(case: dict) -> str | None:
    match = _ELECTION_DATE.search(case["rules"])
    if match and _SEAT_OUTCOME.search(case["question"]):
        try:
            return _calendar_date(match[1]).date().isoformat()
        except ValueError:
            pass
    return None


def scheduled_round_dates(rules: str) -> list[date]:
    """Read explicit, year-qualified dates attached to scheduled events in the frozen rules."""
    dates = set()
    for match in _ROUND_DATES.finditer(rules):
        value = match[1] or match[2]
        try:
            dates.add(date.fromisoformat(value) if "-" in value else _calendar_date(value).date())
        except ValueError:
            continue
    return sorted(dates)


def simulation_case(case: dict) -> dict:
    """Separate an observable outcome horizon from a provider's nominal close."""
    if case.get("deadline_basis") != "market_end_date_requires_rule_review":
        return case
    first_paragraph = case["rules"].split("\n\n", 1)[0]
    sentences = sentence_records(first_paragraph)
    statements = [row["quote"] for row in sentences]
    statements.append(first_paragraph[sentences[-1]["end"] if sentences else 0:])
    explicit = [match for statement in statements
                if _YES_RULE.search(statement)
                for match in _ET_DEADLINE.finditer(statement)]
    deadline, basis = None, None
    if len(explicit) == 1:
        try:
            local_date = _calendar_date(explicit[0][2])
            local_time = datetime.strptime(explicit[0][3].upper().replace(" ", ""), "%I:%M%p").time()
            deadline = datetime.combine(local_date.date(), local_time, ZoneInfo("America/New_York"))
            basis = "explicit_contract_deadline"
        except ValueError:
            pass
    elif not explicit and (vote := scheduled_vote(case)):
        nominal = utc_timestamp(case["outcome_deadline"], "outcome_deadline")
        vote_start = datetime.fromisoformat(vote).replace(tzinfo=timezone.utc)
        if nominal <= vote_start + timedelta(days=1):
            deadline = vote_start + timedelta(days=7)
            basis = "scheduled_election_count_window_assumption"
    if deadline is None or deadline <= utc_timestamp(case["as_of"], "as_of"):
        return case
    return case | {"outcome_deadline": deadline.astimezone(timezone.utc).isoformat(),
                   "deadline_basis": basis, "nominal_market_end": case["outcome_deadline"]}


def contract_state(case: dict, sources: list[dict], *, include_projection: bool = True) -> dict:
    """Retain the same selected public sources as QA, outside the event timeline."""
    cutoff = utc_timestamp(case["as_of"], "as_of")
    evidence = public_sources(sources, text_key="excerpt")
    for row in evidence:
        row["source"] = row.get("title", "")
    state = {
        "question": case["question"],
        "rules": case["rules"],
        "as_of": case["as_of"],
        "outcome_deadline": case["outcome_deadline"],
        "deadline_basis": case.get("deadline_basis", "unspecified"),
        "historical_evidence": evidence,
        "evidence_packet": packet_manifest(evidence, text_key="excerpt"),
    }
    state.update({key: case[key] for key in ("outcome_window_start", "outcome_window_basis") if key in case})
    vote = scheduled_vote(case)
    spec = compile_spec(case["question"], case["rules"], scheduled_election=bool(vote))
    state["spec"] = spec.model_dump(exclude_none=True)
    if vote:
        state["scheduled_milestone"] = {"date": vote, "event": "Scheduled election",
                                       "counting_from": (datetime.fromisoformat(vote) + timedelta(days=1)).date().isoformat(),
                                       "measure": "Simulated vote and seat ranking under the contract rules"}
    if spec.route == "environment" and include_projection:
        if projection := build_projection(case["question"], sources, cutoff.date()):
            projection["votes_tiebreak"] = "ties will be broken by the total number of valid votes" in case["rules"].lower()
            state["projection"] = projection
        else:
            state["spec"]["reason"] = "Missing a supported pre-cutoff polling table, party abbreviation, or seat allocation inputs."
    if spec.route == "environment" and not include_projection:
        state["spec"]["route"] = "hybrid"
    return state


def milestone_phase(contract: dict, clock: dict) -> str | None:
    milestone = contract.get("scheduled_milestone")
    if not milestone:
        return None
    counting_from = milestone["counting_from"]
    if clock["interval_end"] < counting_from:
        return "pre_count"
    if clock["interval_start"] < counting_from:
        return "count_window_opens"
    return "count_window_open"


def scheduled_window_event(contract, clock, step):
    """Create a clock fact, never a measurement or an agent action."""
    from app.domain import TimelineEvent
    from app.simulation.clock import event_in_window

    date_label = contract.get("scheduled_milestone", {}).get("counting_from")
    if not date_label:
        return None
    timestamp = f"{date_label}T00:00:00+00:00" if clock.get("interval_end_utc") else date_label
    if event_in_window(timestamp, clock) is not True:
        return None
    return TimelineEvent(timeline_id=f"sim_{step}_scheduled_window", timestamp=timestamp,
                         description="The configured count window opened.",
                         event_type="scheduled_window", change_kind="external_observation", confidence=1, importance=.5,
                         origin="simulated", round_id=step)


def discard_unsupported_counts(snapshot, settlement):
    contract = snapshot.world_state.get("benchmark_contract", {})
    if (not contract.get("scheduled_milestone") or contract.get("projection")
            or contract.get("spec", {}).get("predicate") not in {"rank", "threshold"}):
        return
    from app.simulation.activities import observations
    retained = []
    for event in settlement.public_events:
        text = event.description
        measured = re.search(r"\b(?:seats?|votes?|seat rankings?|vote totals?|seat counts?|official results)\b", text, re.I)
        reported = re.search(r"\b(?:counted|tallied|allocated|won|placed|finished|official count|certified|"
                             r"received|secured|reported|declared|announced)\b", text, re.I)
        observational = event.change_kind == "external_observation" or any(
            row.get("stage") in {"completed", "measured", "in_effect"} for row in observations(event.model_dump())
        )
        if event.event_type == "scheduled_window" or (measured and (observational or reported)):
            settlement._process_diagnostics.append({"code": "unsupported_count_observation", "event_ref": event.timeline_id})
        else:
            retained.append(event)
    settlement.public_events = retained


def settle_projected_round(snapshot, clock, *, seed, sample):
    """Advance a scheduled count without creating actors or making a model call."""
    from app.domain import ForecastRound, GameMasterSettlement, RelationshipProjection
    from app.simulation.transitions import apply_round_update

    contract = snapshot.world_state["benchmark_contract"]
    projection = contract["projection"]
    spec = contract["spec"]
    prior = snapshot.world_state.get("contract_result")
    observation = prior
    summary = "Scheduled count has not opened."
    events = []
    if prior:
        summary = "The previously simulated seat count remains in force."
    elif clock["contract_milestone_phase"] != "pre_count":
        count = draw_count(projection, seed=seed, sample=sample)
        if count is not None:
            target = projection["target"]
            held = count["seats"][target]
            if spec["predicate"] == "rank":
                rank = spec["target_rank"]
                ranking = count["ranking"]
                answer = (ranking[rank - 1] == target and held > 0) if rank <= len(ranking) else False
                if 0 < rank <= len(ranking) and not projection["votes_tiebreak"]:
                    seat_at_rank = count["seats"][ranking[rank - 1]]
                    if list(count["seats"].values()).count(seat_at_rank) > 1:
                        answer = None
            else:
                threshold = spec.get("threshold", projection["total_seats"] // 2 + 1)
                answer = held >= threshold
            observation = count | {"kind": "programmatic_count", "answer": (
                "unknown" if answer is None else "yes" if answer else "no"
            ), "model": projection["model"]}
            summary = "The simulated national count allocates " + ", ".join(
                f"{party}: {count['seats'][party]}" for party in count["ranking"]
            ) + " seats."
            events.append({"timeline_id": f"sim_{snapshot.round_id + 1}_count", "timestamp": clock["interval_end"],
                           "description": summary, "actors": [], "confidence": 1, "importance": 1})
        else:
            summary = "The sampled count could not allocate seats."
    world = snapshot.world_state | {"contract_result": observation}
    settlement = GameMasterSettlement(state_summary=summary, next_world_state=world, public_events=events,
                                      analysis=projection["assumption"])
    after = apply_round_update(snapshot, [], settlement, RelationshipProjection())
    return after, ForecastRound(step=after.round_id, input_snapshot_version=snapshot.snapshot_version,
                                output_snapshot_version=after.snapshot_version, state_summary=summary,
                                timeline=settlement.public_events, contract_observation=observation)
