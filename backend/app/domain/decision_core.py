"""Small, evidence-linked decision dossier; no personality or strength scores."""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Callable, Literal
from datetime import date
from pydantic import BaseModel, ConfigDict, Field, ValidationError

VERSION = "decision-core-1"


class SourceSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    quote: str = Field(min_length=8, max_length=1600)
    url: str = ""
    title: str = ""
    section: str = ""
    snapshot_sha256: str = ""
    revision_id: str = ""
    revision_at: str = ""
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)


class DecisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default="", max_length=80)
    description: str = Field(min_length=1, max_length=500)
    conditions: str = Field(default="", max_length=300)
    timeline_refs: list[str] = Field(default_factory=list, max_length=12)
    provenance: Literal["source_statement", "inferred", "assumption", "legacy_unreviewed"] = "inferred"
    source_refs: list[SourceSupport] = Field(default_factory=list, max_length=4)
    holder: str = Field(default="", max_length=120)
    valid_from: str = ""
    valid_until: str = ""
    review_condition: str = Field(default="", max_length=300)
    claim_scope: Literal["structural", "conditional", "historical_observation", "unreviewed"] = "unreviewed"


class DecisionGoal(DecisionItem):
    priority: Literal["high", "medium", "low", "unknown"] = "unknown"
    tradeoff: str = Field(default="", max_length=300)
    success_condition: str = Field(default="", max_length=300)


class DecisionLimit(DecisionItem):
    kind: Literal["physical", "procedural", "normative", "unknown"] = "unknown"


class DecisionResource(DecisionItem):
    value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str = Field(default="", max_length=80)


class InitialJudgment(DecisionItem):
    unknowns: str = Field(default="", max_length=300)
    perspective: Literal["actor", "observer"] = "observer"


class DecisionSeatBinding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    policy_version: str
    seat_id: str
    owner_name: str
    entity_type: Literal["person", "country", "government", "organization", "company", "group"] | None = None
    represents: str = ''
    role: str
    required: bool
    decision_ids: list[str] = Field(min_length=1, max_length=4)
    source_refs: list[SourceSupport] = Field(min_length=1, max_length=3)


class DecisionCore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["decision-core-1"] = VERSION
    origin: Literal["generated", "legacy_projection", "minimal_fallback"] = "generated"
    as_of: str = ""
    simulation_eligible: bool | None = None
    decision_scope: str = Field(default="", max_length=300)
    seat: DecisionSeatBinding | None = None
    evidence_gaps: list[str] = Field(default_factory=list, max_length=5)
    review_notes: list[str] = Field(default_factory=list, max_length=20)
    mandate: list[DecisionItem] = Field(default_factory=list, max_length=3)
    goals: list[DecisionGoal] = Field(default_factory=list, max_length=20)
    means: list[DecisionItem] = Field(default_factory=list, max_length=5)
    resources: list[DecisionResource] = Field(default_factory=list, max_length=4)
    limits: list[DecisionLimit] = Field(default_factory=list, max_length=20)
    decision_rules: list[DecisionItem] = Field(default_factory=list, max_length=2)
    initial_judgments: list[InitialJudgment] = Field(default_factory=list, max_length=4)
    initial_intentions: list[DecisionItem] = Field(default_factory=list, max_length=1)


class CurrentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=3000, description="A concise current action plan; aim for at most 500 characters. Preserve essential conditions and dependencies.")
    goal_ids: list[str] = Field(default_factory=list, max_length=4)
    status: Literal["active", "completed", "blocked", "abandoned"] = "active"
    review_when: str = Field(default="", max_length=300)
    change_reason: str = Field(default="", max_length=300)
    timeline_refs: list[str] = Field(default_factory=list, max_length=12)
    updated_round: int = Field(default=0, ge=0)


def _key(text):
    return re.sub(r"[\W_]+", "", str(text).casefold())


ITEM_SCHEMAS = {
    "mandate": (DecisionItem, 3),
    "goals": (DecisionGoal, 4),
    "means": (DecisionItem, 5),
    "resources": (DecisionResource, 4),
    "limits": (DecisionLimit, 5),
    "decision_rules": (DecisionItem, 2),
    "initial_judgments": (InitialJudgment, 4),
    "initial_intentions": (DecisionItem, 1),
}


def _validate_item_evidence(item, allowed, sources):
    if set(item.timeline_refs) - allowed:
        raise ValueError("Decision dossier cites unavailable evidence")
    for ref in item.source_refs:
        source = sources.get(ref.evidence_id)
        if not source or ref.quote not in source['text']:
            raise ValueError("Decision dossier cites an unavailable source span")
        for field in ('url', 'title', 'section', 'snapshot_sha256', 'revision_id', 'revision_at'):
            setattr(ref, field, source.get(field, ''))
        if source.get('span_start') is not None:
            start = source['span_start'] + source['text'].index(ref.quote)
            ref.span_start = start
            ref.span_end = start + len(ref.quote)
        else:
            ref.span_start = ref.span_end = None
    for value in (item.valid_from, item.valid_until):
        if value:
            date.fromisoformat(value)
    if item.valid_from and item.valid_until and item.valid_from > item.valid_until:
        raise ValueError("Invalid dossier validity interval")


def validate_core(
    raw: dict, allowed_refs=(), *, as_of="", source_catalog=(),
    item_errors: dict[tuple[str, int], str] | None = None,
    resolve_references: Callable[[dict], dict] | None = None,
) -> DecisionCore:
    if item_errors is None:
        core = DecisionCore.model_validate(raw)
        if len(core.goals) > 4 or len(core.limits) > 5:
            raise ValueError("Generated decision dossier exceeds compact field limits")
    else:
        core = DecisionCore.model_validate({key: value for key, value in raw.items() if key not in ITEM_SCHEMAS})
    core.origin = "generated"
    allowed, seen, used = set(allowed_refs), set(), set()
    sources = {s['evidence_id']: s for s in source_catalog}
    for kind, (model, limit) in ITEM_SCHEMAS.items():
        accepted = []
        rows = getattr(core, kind) if item_errors is None else raw.get(kind, [])
        if not isinstance(rows, list):
            item_errors[kind, -1] = f"{kind} must be a list"
            continue
        for index, row in enumerate(rows):
            try:
                if item_errors is None:
                    item = row
                else:
                    if index >= limit:
                        raise ValueError(f"{kind} allows at most {limit} entries")
                    value = resolve_references(row) if resolve_references and isinstance(row, dict) else row
                    item = model.model_validate(value)
                key = _key(item.description)
                if key in seen:
                    continue
                _validate_item_evidence(item, allowed, sources)
                if kind == "initial_judgments" and (item.perspective == "observer"
                                                    or not (item.timeline_refs or item.source_refs)):
                    gap = item.unknowns or item.description
                    if gap not in core.evidence_gaps and len(core.evidence_gaps) < 5:
                        core.evidence_gaps.append(gap)
                    continue
                if kind == "initial_intentions" and (
                    item.provenance != "source_statement" or not (item.timeline_refs or item.source_refs)
                    or item.claim_scope != "conditional"
                ):
                    continue
                boundary = (as_of or core.as_of)[:10]
                if kind == 'initial_intentions' and boundary and (
                    item.valid_from and item.valid_from > boundary or item.valid_until and item.valid_until < boundary
                ):
                    core.review_notes.append('An intention outside its stated validity window was excluded from the opening decision state.')
                    continue
                if not (item.timeline_refs or item.source_refs) and item.provenance in {"source_statement", "inferred"}:
                    item.provenance = "assumption"
                if kind == "decision_rules" and (not item.conditions or len(set(item.timeline_refs)) < 2):
                    continue
                if kind == "resources" and item.value is not None and (not item.unit or not (item.timeline_refs or item.source_refs)):
                    item.value = None
                if kind in {'mandate', 'means', 'resources', 'limits'}:
                    reason = unsuitable_claim(kind, item)
                    if reason:
                        core.review_notes.append(f"{kind}: {reason}")
                        continue
                identifier = item.id or f"{kind}_{len(accepted)+1}"
                if identifier in used:
                    raise ValueError("Duplicate decision dossier item ID")
                item.id = identifier
            except ValueError as exc:
                if item_errors is None:
                    raise
                if isinstance(exc, ValidationError):
                    message = "; ".join(
                        f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                        for error in exc.errors(include_input=False, include_url=False)
                    )
                else:
                    message = str(exc)
                item_errors[kind, index] = message[:400]
                continue
            seen.add(key)
            used.add(item.id)
            accepted.append(item)
        setattr(core, kind, accepted)
    core.as_of = as_of or core.as_of
    core.review_notes = list(dict.fromkeys(core.review_notes))[:20]
    return core


def unsuitable_claim(kind, item):
    """Conservative rejection of obvious incident/ownership category mistakes."""
    text = item.description.casefold()
    if item.claim_scope == 'historical_observation':
        return 'Historical activity alone did not establish durable authority, means, resources, or constraints.'
    incident = re.search(r'\b(?:was|were)(?:\s+\w+){0,5}\s+(?:hit|struck|attacked)|\b(?:launched|suffered) \d+|曾遭|曾被|遭到.*袭击|发射了?\d+|launched.*missile attacks.*during', text)
    effect = re.search(r'\b(?:lost|destroyed|disabled|depleted|reduced|unable|damaged capacity)\b|损失|摧毁|受损|无法|消耗|削弱|能力下降', text)
    if kind == 'limits' and incident and not effect:
        return 'The source does not establish how the incident affects current capabilities or options.'
    if kind == 'resources' and (re.search(r'\b(?:losses inflicted|damage inflicted)\b|造成.*损失', text) or (re.search(r'export value|出口价值', text) and re.search(r'halted|stopped|lost|disrupted|中断|停止|损失', text))):
        return 'External effects and economic results do not establish resources the actor controls.'
    return ''


def applicable_core(core, on_date=''):
    """Read-only validity view; no old dossier or frozen experiment is rewritten."""
    on_date = (on_date or "")[:10]
    value = core.model_copy(deep=True)
    for kind in ('mandate', 'means', 'resources', 'limits', 'decision_rules'):
        accepted = []
        for item in getattr(value, kind):
            reason = unsuitable_claim(kind, item)
            if not reason and on_date and ((item.valid_from and on_date < item.valid_from)
                                          or (item.valid_until and on_date > item.valid_until)):
                reason = 'This evidence does not apply at the simulation date.'
            if not reason and kind == 'limits' and re.search(r'\b\d+[- ]day\b|\d+\s*天', item.description, re.I) and not item.valid_until:
                reason = 'This temporary constraint needs a verifiable expiration date.'
            if reason:
                value.review_notes.append(f'{kind}: {reason}')
            else:
                accepted.append(item)
        setattr(value, kind, accepted)
    value.review_notes = list(dict.fromkeys(value.review_notes))[:20]
    return value


def grounded_decision_fields(core: DecisionCore, on_date: str = '') -> set[str]:
    """Find the two kinds of decision evidence usable at the relevant boundary."""
    applicable = applicable_core(core, on_date[:10] if len(on_date) >= 10 else '')
    return {
        kind for kind in ('mandate', 'means')
        if any(item.provenance in {'source_statement', 'inferred'} and (item.timeline_refs or item.source_refs)
               for item in getattr(applicable, kind))
    }


def legacy_core(record: dict) -> DecisionCore:
    """Conservative read-only projection. Never infer authority or hard limits."""
    state = record.get("public_state") or {}
    if record.get("decision_core"):
        return DecisionCore.model_validate(record["decision_core"])
    core = DecisionCore(origin="legacy_projection", as_of=str(state.get("knowledge_boundary") or ""))
    for i, goal in enumerate(record.get("desires") or []):
        if goal.get("description") and len(core.goals) < 20:
            core.goals.append(DecisionGoal(id=goal.get("desire_id") or f"goal_{i+1}",
                description=str(goal["description"])[:500], provenance="legacy_unreviewed"))
    for i, limit in enumerate(record.get("constraints") or []):
        if str(limit).strip() and len(core.limits) < 20:
            core.limits.append(DecisionLimit(id=f"legacy_limit_{i+1}",description=str(limit)[:500],
                                            provenance="legacy_unreviewed"))
    # Preserve useful material without relabeling one-off activity as a general
    # capability, personality or verified resource amount.
    return core


def public_core(core: DecisionCore) -> dict:
    value = core.model_dump()
    value.pop("initial_judgments")  # Subjective state never enters public OKF.
    return value


def project_record(record: dict) -> dict:
    value = deepcopy(record)
    core = legacy_core(value)
    value["decision_core"] = applicable_core(core, core.as_of).model_dump()
    return value
