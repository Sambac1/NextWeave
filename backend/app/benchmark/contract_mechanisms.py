"""Structured contract predicates and evidence-anchored round updates."""
from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from app.domain.identity import matching_subjects
from app.knowledge.sentences import sentence_records
from app.simulation.clock import event_in_window, on_or_after


class ContractSpec(BaseModel):
    route: Literal["environment", "actor", "hybrid", "review"]
    predicate: Literal["rank", "threshold", "occurrence", "stage", "duration", "semantic"]
    subject: str = ""
    action: str = ""
    target: str = ""
    required_observations: list[Literal["in_person", "authorized", "public_confirmation", "first_in_scope"]] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list, max_length=8)
    terminal_stage: str = ""
    target_rank: int | None = None
    threshold: int | None = None
    duration_days: int | None = None
    reason: str = ""


class CompiledPredicate(BaseModel):
    """An optional one-call compiler for contracts outside the fast paths."""
    route: Literal["actor", "hybrid", "review"]
    predicate: Literal["occurrence", "stage", "duration", "semantic"]
    subject: str = Field(default="", max_length=160)
    stages: list[str] = Field(default_factory=list, max_length=8)
    terminal_stage: str = Field(default="", max_length=40)
    duration_days: int | None = Field(default=None, ge=1, le=366)
    rule_quote: str = Field(default="", max_length=420)

    @model_validator(mode="after")
    def validate_predicate(self):
        if self.route == "review":
            if self.predicate != "semantic":
                raise ValueError("Review requires a semantic predicate")
        elif (self.predicate == "semantic" or not self.subject.strip() or self.terminal_stage not in self.stages
              or len(self.stages) != len(set(self.stages))
              or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name) for name in self.stages)
              or (self.duration_days is not None) != (self.predicate == "duration")
              or (self.predicate == "duration" and "ended" not in self.stages)):
            raise ValueError("Invalid structured contract predicate")
        return self

    def bind(self, rules: str) -> ContractSpec | None:
        quote = self.rule_quote.strip()
        if self.route == "review" or len(quote) < 12 or quote not in rules:
            return None
        if self.duration_days is not None and not re.search(rf"\b{self.duration_days}\s+(?:consecutive\s+)?days?\b", quote, re.I):
            return None
        return ContractSpec(**self.model_dump(exclude={"rule_quote"}))


class ContractUpdate(BaseModel):
    stage: str
    event_ref: str
    quote: str = Field(min_length=8, max_length=600)
    # Continuous observation is supplied explicitly, never inferred from silence.
    covered_from: date | None = None
    covered_through: date | None = None


_RANKS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
_RANK = re.compile(r"\b(first|second|third|fourth|fifth|sixth|\d+(?:st|nd|rd|th))\s+(?:most|greatest)\s+seats\b", re.I)
_OCCURRENCE = {
    "visit": r"visited|arrived in|arrived at|paid a visit to",
    "attend": r"attended|took part in|participated in|was present at",
    "travel to": r"traveled to|travelled to|arrived in|arrived at",
    "strike": r"struck|launched a strike on|hit",
    "attack": r"attacked|launched an attack on",
    "withdraw from": r"withdrew from|pulled out of",
}
_STAGES = {
    "signed": r"signed|executed the agreement",
    "approved": r"approved|adopted|passed",
    "in_force": r"(?:entered|came|went) into (?:force|effect)|took effect|(?:was|were) in (?:force|effect)|remained in force|held",
    "ended": r"ended|expired|was lifted|were lifted|ceased|terminated|collapsed",
    "proposed": r"proposed|suggested|offered|drafted",
    "planned": r"planned|announced|scheduled|invited",
    "cancelled": r"cancelled|canceled|called off",
}
_UNCERTAIN = re.compile(
    r"\b(?:not|never|no|false|denied|whether|rumou?rs?|if|unless|would|could|may|might|will|"
    r"expected|possible|hypothetical|plans?|intends?|scheduled|unconfirmed|unclear|unverified|"
    r"alleged|purported|disputed|didn['’]t)\b", re.I,
)
_STOP = {"the", "a", "an", "of", "in", "to", "for", "with", "on", "at", "and", "next", "its", "x"}
_OBSERVATIONS = {
    "in_person": r"\bin[- ]person\b",
    "authorized": r"\b(?:authorized|authorised|formally designated)\b",
    "public_confirmation": r"\b(?:publicly (?:acknowledged|confirmed)|officially confirmed|confirmed by (?:the )?(?:government|ministry)|consensus of credible (?:reports|media))\b",
    "first_in_scope": r"\b(?:first|next)\b",
}


def _tokens(value):
    value = re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda match: match[0].replace(".", ""), value)
    value = re.sub(r"\b(?:United States|USA)\b", "US", value, flags=re.I)
    return {token for token in re.findall(r"[^\W_]+", value.casefold()) if token not in _STOP}


def _has_target(spec, text):
    words = _tokens(text)
    if spec.target == "ceasefire":
        return bool(words & {"ceasefire", "truce"}) or "cease-fire" in text.casefold()
    return bool(spec.target) and _tokens(spec.target) <= words


def _subject_pattern(names):
    parts = [r"[\s.\-]+".join(map(re.escape, re.findall(r"[^\W_]+", name))) for name in names if name]
    return r"(?<!\w)(?:" + "|".join(part for part in parts if part) + r")(?!\w)" if any(parts) else None


def supports_stage(spec, stage, event, *, subject_aliases=(), quote=""):
    """Certify only clauses whose bound subject, action, and scope can be checked."""
    subject = _subject_pattern((spec.subject, *subject_aliases))
    phrase = _OCCURRENCE.get(spec.action) if stage == "occurred" else _STAGES.get(stage)
    if not phrase or not spec.target or (spec.predicate == "occurrence" and not subject):
        return False
    if stage == spec.terminal_stage and (
        getattr(event, "modality", "occurred") != "occurred"
        or getattr(event, "time_role", "occurrence") != "occurrence"
    ):
        return False
    # Search the containing clause, rather than letting a fragment of a denial
    # or conditional sentence independently certify an occurrence.
    sentences = sentence_records(event.description)
    remainder = event.description[sentences[-1]["end"]:] if sentences else event.description
    clauses = (clause for line in [*(row["quote"] for row in sentences), remainder]
               for clause in re.split(r"[;\n]", line))
    for clause in clauses:
        if quote and quote not in clause and clause.strip() not in quote:
            continue
        if not _has_target(spec, clause) or not re.search(rf"\b(?:{phrase})\b", clause, re.I):
            continue
        if stage not in {"planned", "proposed", "cancelled"} and _UNCERTAIN.search(clause):
            continue
        if stage == "occurred":
            # For occurrence, the subject must perform the verb. A target
            # mentioned elsewhere in the sentence cannot supply the object.
            match = re.search(rf"{subject}\s+(?:personally\s+|officially\s+)?(?:{phrase})\b([^,;!?]*)", clause, re.I)
            if not match or not _has_target(spec, match[1]):
                continue
            if spec.action in {"visit", "travel to"} and not re.match(
                rf"\s+(?:(?:the|a|an)\s+)?{re.escape(spec.target.removeprefix('the '))}(?!\w)", match[1], re.I
            ):
                continue
            if any(not re.search(_OBSERVATIONS[key], clause, re.I) for key in spec.required_observations):
                continue
        else:
            if subject and not re.search(subject, clause, re.I):
                continue
            # Don't let 'sanctions were announced while the law took effect'
            # bind the action for the law to sanctions.
            if spec.predicate in {"stage", "duration"} and stage in {"in_force", "ended"}:
                noun = r"cease[- ]?fire|truce" if spec.target == "ceasefire" else re.escape(spec.target)
                if not re.search(rf"\b(?:{noun})\s+(?:(?:officially|formally|continuously)\s+)?(?:{phrase})\b", clause, re.I):
                    continue
        return True
    return False


def subject_aliases(spec, actors):
    matches = matching_subjects(spec.subject, actors, lambda actor: [actor.name, *actor.aliases],
                                lambda actor: actor.agent_type == "person")
    return [matches[0].name, *matches[0].aliases] if len(matches) == 1 else []


def missing_sequential_observation(contract):
    # The separate polling baseline does not implement an observation for
    # the sequential actor simulation.
    if contract.get("spec", {}).get("predicate") in {"rank", "threshold"}:
        return {"kind": "observation", "label": "Supported vote and seat count mechanism", "ready": False}
    return None


def decision_coverage(contract, actors):
    """Report which explicit roles and observations the frozen plan can cover."""
    spec = ContractSpec.model_validate(contract.get("spec") or {"route": "review", "predicate": "semantic"})
    entries = []
    if spec.subject:
        entries.append({"kind": "decision_subject", "label": spec.subject,
                        "ready": bool(subject_aliases(spec, actors))})
    if gap := missing_sequential_observation(contract):
        entries.append(gap)
    elif spec.predicate == "semantic":
        entries.append({"kind": "observation", "label": "Machine-verifiable outcome condition", "ready": False})
    elif spec.predicate == "occurrence":
        entries.append({"kind": "observation", "label": f"Public observation of {spec.action or 'the requested action'}",
                        "ready": bool(spec.action and spec.target)})
        if "authorized" in spec.required_observations:
            entries.append({"kind": "observation", "label": "Participant authorization must be observed in the trajectory",
                            "ready": any(
                                row.source_refs and re.search(r"\b(?:authorize|designate|appoint|delegate)\w*\b", row.description, re.I)
                                for actor in actors for row in (actor.decision_core.mandate if actor.decision_core else [])
                            )})
    return entries


def relevant_to_contract(spec, question, event, aliases=()):
    if event.event_type == "scheduled_window":
        return True
    if any(supports_stage(spec, stage, event, subject_aliases=aliases) for stage in spec.stages):
        return True
    text = event.description
    subject = _subject_pattern((spec.subject, *aliases))
    if subject and not re.search(subject, text, re.I):
        return False
    if _has_target(spec, text):
        return True
    # A nonterminal estimate may cite a narrower development than an exact
    # resolution. Require scope words beyond the actor's name or boilerplate.
    focus = _tokens(spec.target or question) - _tokens(spec.subject)
    focus -= {"will", "does", "did", "was", "before", "after", "most", "first", "fourth", "win", "have", "by"}
    focus = {word for word in focus if len(word) > 2 and not word.isdigit()}
    overlap = focus & _tokens(text)
    return len(overlap) >= min(2, len(focus)) and bool(overlap)


def compile_spec(question: str, rules: str, *, scheduled_election: bool) -> ContractSpec:
    simple = not re.search(r"\b(?:not|unless|either|both|or)\b", question, re.I)
    if simple and scheduled_election and re.search(r"\b(?:rank|ranking|most|greatest)\b", rules, re.I) and (rank := _RANK.search(question)):
        ordinal = rank[1].lower()
        target = _RANKS[ordinal] if ordinal in _RANKS else int(ordinal[:-2])
        subject = re.match(r"(?i)^will\s+(.+?)\s+(?:win|get|receive|secure|hold|have)\s+", question)
        return ContractSpec(route="environment", predicate="rank", target_rank=target, subject=subject[1] if subject else "")
    if simple and scheduled_election and re.search(r"\b(?:seats?|majority)\b", rules, re.I) and (majority := re.search(r"\b(?:at least (\d+) seats|(?:an? )?(?:absolute )?majority of (?:the )?seats)\b", question, re.I)):
        return ContractSpec(route="environment", predicate="threshold",
                            threshold=int(majority[1]) if majority[1] else None)
    if re.search(r"\b(?:ceasefire|truce)\b", question, re.I):
        stages = ["proposed", "signed", "in_force", "ended"]
        if duration := re.search(r"\b(?:last|hold|remain in force)(?:s|ed)?\s+(?:for\s+)?(?:at least\s+)?(\d+)\s+(?:consecutive\s+)?days?\b", question, re.I):
            return ContractSpec(route="hybrid", predicate="duration", stages=stages,
                                terminal_stage="in_force", duration_days=int(duration[1]), target="ceasefire")
        if re.search(r"\b(?:signed?|signing|take effect|enter into force|go into effect|come into effect)\b", question, re.I):
            terminal = "in_force" if re.search(r"effect|force", question, re.I) else "signed"
            return ContractSpec(route="hybrid", predicate="stage", stages=stages, terminal_stage=terminal, target="ceasefire")
    if simple and (action := re.match(
        r"(?i)^will\s+(.+?)\s+(visit|attend|travel to|strike|attack|withdraw from)\s+(.+?)(?=\s+(?:by|before|on or before|no later than)\b|[?]|$)", question
    )):
        target = action[3].strip()
        required = []
        if action[2].lower() == "attend":
            if re.search(r"\b(?:must be|only) in[- ]person\b", rules, re.I):
                required.append("in_person")
            if re.search(r"\b(?:are|be|acting with the knowledge and) authori[sz]ed\b", rules, re.I):
                required.append("authorized")
            if re.search(r"\b(?:must be|and be) publicly acknowledged\b", rules, re.I):
                required.append("public_confirmation")
            if re.search(r"\bnext\b", target, re.I):
                required.append("first_in_scope")
        return ContractSpec(route="actor", predicate="occurrence", subject=action[1],
                            action=action[2].lower(), target=target, required_observations=required,
                            stages=["planned", "cancelled", "occurred"], terminal_stage="occurred")
    if (measure := re.match(r"(?i)^will\s+(.+?)\s+(?:take effect|enter into force|be implemented)\b", question)) and re.search(
        r"\b(?:sanctions?|law|bill|regulation)\b", measure[1], re.I
    ):
        target = re.sub(r"(?i)^(?:the|a|an)\s+", "", measure[1]).strip()
        return ContractSpec(route="hybrid", predicate="stage",
                            stages=["proposed", "approved", "in_force", "ended"], terminal_stage="in_force", target=target)
    return ContractSpec(route="review", predicate="semantic",
                        reason="No supported structured predicate. Existing semantic replay is reported separately.")


def apply_updates(spec: ContractSpec, prior: dict, updates: list[ContractUpdate], events: list, clock: dict,
                  *, subject_aliases=(), outcome_window_start=None) -> dict:
    """Commit supported public updates, in event order, to one trajectory."""
    state = dict(prior)
    coverage = [list(span) for span in prior.get("coverage", [])]
    indexed = {event.timeline_id: (i, event) for i, event in enumerate(events)}
    ordered = (update for update in updates if update.event_ref in indexed)
    for update in sorted(ordered, key=lambda row: (indexed[row.event_ref][1].timestamp, indexed[row.event_ref][0])):
        entry = indexed.get(update.event_ref)
        if entry is None or update.stage not in spec.stages:
            continue
        event = entry[1]
        if (len(update.quote.strip()) < 8 or update.quote.strip() not in event.description
                or event_in_window(event.timestamp, clock) is not True
                or not supports_stage(spec, update.stage, event, subject_aliases=subject_aliases, quote=update.quote.strip())):
            continue
        if update.stage == spec.terminal_stage and on_or_after(event.timestamp, outcome_window_start) is not True:
            continue
        occurred = date.fromisoformat(event.timestamp[:10])
        left, right = (date.fromisoformat(clock[k]) for k in ("interval_start", "interval_end"))
        previous = state.get("stage")
        if previous in {"signed", "in_force"} and update.stage == "proposed":
            continue
        if previous == "in_force" and update.stage == "signed":
            continue
        if spec.predicate == "duration" and update.stage == spec.terminal_stage and (update.covered_from or update.covered_through):
            if (not update.covered_from or not update.covered_through
                    or not left <= update.covered_from < update.covered_through <= occurred
                    or on_or_after(update.covered_from.isoformat(), outcome_window_start) is not True
                    or not all(day.isoformat() in update.quote for day in (update.covered_from, update.covered_through))
                    or not re.search(r"\b(?:continuously|without interruption|throughout|remained in force)\b", update.quote, re.I)):
                continue
        state.update(stage=update.stage, event_ref=update.event_ref)
        if spec.predicate != "duration" and update.stage == spec.terminal_stage:
            state.setdefault("matched_ref", update.event_ref)
        if update.stage == "ended":
            coverage = []
        if spec.predicate == "duration" and update.stage == spec.terminal_stage and update.covered_from and update.covered_through:
            # Dates describe day boundaries. Only an explicitly observed span within this round counts.
            if not left <= update.covered_from < update.covered_through <= occurred:
                continue
            start, end = update.covered_from, update.covered_through
            if coverage and date.fromisoformat(coverage[-1][1]) >= start:
                start = date.fromisoformat(coverage.pop()[0])
            coverage.append([start.isoformat(), end.isoformat()])
            if end - start >= timedelta(days=spec.duration_days or 0):
                state.setdefault("matched_ref", update.event_ref)
    if spec.predicate == "duration":
        state["coverage"] = coverage
    return state
