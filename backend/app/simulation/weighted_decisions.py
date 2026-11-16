"""Generate, audit and draw one decision from each active actor's candidate pool."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.decision_core import CurrentPlan
from app.domain.forecast import ActionProposal
from app.llm.client import LLMError
from app.llm.concurrency import gather_complete
from app.simulation.action_audit import _program_violations
from app.simulation.context import actor_context, cited_public_context, compact_json, public_context
from app.simulation.deliberation import _visible_interventions
from app.simulation.main_forecast import ExecutionCondition, gated_action

VERSION = "weighted-decisions-1"
CANDIDATE_FORMAT_REVISION = 6
MAX_CANDIDATES = 3

_OUTWARD_LABELS = {"attack", "strike", "deploy", "send", "sign", "execute", "announce", "issue", "submit", "launch"}
_OUTWARD_INSTRUCTION = re.compile(
    r"(?:^|[.;]\s*|\b(?:and|then|by|while)\s+)(?:(?:publicly|formally)\s+)?"
    r"(?:announce|send|issue|submit|file|sign|impose|launch|notify|dispatch|approve|reject|signal|propose|request|publish|vote)\b"
    r"(?!\s+(?:no|nothing|neither)\b)"
    r"|(?:^|并|同时|随后|立即)(?:正式|公开)?(?:宣布|签署|发送|提交|发布|实施|提议)",
    re.I,
)
_EXPLICIT_WAIT = re.compile(r"^(?:take no (?:external|outward) action|wait|monitor|observe)\b", re.I)
_DEFERRED_CLAUSE = re.compile(r"[,;]\s*(?:and\s+)?then\s+", re.I)
_LATER_BOUNDARY = re.compile(r"\b(?:after|upon|when)\b|\bat (?:[\w'’\-]+ ){0,5}(?:end|conclusion)\b|\bconcludes?\b", re.I)

STAT_FIELDS = (
    "decisions", "multi_candidate_decisions", "single_candidate_decisions",
    "concentrated_decisions", "rejected_candidates", "duplicate_candidates",
    "supplemented_decisions", "not_selected_for_deliberation",
)


class CandidateAction(BaseModel):
    decision: Literal["act", "wait", "monitor", "no_action"]
    action_type: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=1200)
    target: str = ""
    visibility: Literal["public", "direct", "private", "secret"] = "public"
    commitment: str = Field(default="", max_length=300)
    condition: str = Field(default="", max_length=300)
    timeline_refs: list[str] = Field(default_factory=list, max_length=12)
    reasoning: str = Field(min_length=1, max_length=600)
    expected_effect: str = Field(default="", max_length=400)
    plan_update: CurrentPlan | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    weight: float = Field(ge=0, le=100, strict=True)


class CandidatePool(BaseModel):
    candidates: list[CandidateAction] = Field(max_length=MAX_CANDIDATES)


class MainCandidateAction(CandidateAction):
    execution: ExecutionCondition


class MainCandidatePool(CandidatePool):
    candidates: list[MainCandidateAction] = Field(max_length=MAX_CANDIDATES)


class CandidateReview(BaseModel):
    candidate_id: str
    verdict: Literal["allow", "hard_violation", "duplicate"]
    evidence_ref: str = Field(default="", max_length=300)
    evidence_quote: str = Field(default="", max_length=500)
    duplicate_of: str = ""
    reason: str = Field(default="", max_length=500)


class CandidateAuditBatch(BaseModel):
    round_id: int
    snapshot_version: int
    reviews: list[CandidateReview]


class MainCandidateReview(CandidateReview):
    missing_prerequisite: str = Field(default="", max_length=300)
    prerequisite_met: bool = False


class MainCandidateAuditBatch(CandidateAuditBatch):
    reviews: list[MainCandidateReview]


class CandidateWeight(BaseModel):
    candidate_id: str
    weight: float = Field(ge=0, le=100, strict=True)
    reason: str = Field(default="", max_length=500)


class CandidateWeights(BaseModel):
    weights: list[CandidateWeight] = Field(min_length=1, max_length=3)


class CandidateAuditMismatch(ValueError):
    """The auditor returned a stale or incomplete candidate batch."""


@dataclass(frozen=True)
class Candidate:
    id: str
    draft: CandidateAction
    proposal: ActionProposal
    repairs: tuple[str, ...] = ()

    def summary(self):
        return {
            "candidate_id": self.id,
            **self.proposal.model_dump(include={"agent_id", "decision", "action_type", "action", "target", "visibility", "timeline_refs"}),
            "commitment": self.draft.commitment,
            "condition": self.draft.condition,
            "plan_goal_ids": self.proposal.plan_update.goal_ids if self.proposal.plan_update else [],
            **({"execution": self.draft.execution.model_dump(), "reasoning": self.draft.reasoning,
                "expected_effect": self.draft.expected_effect} if isinstance(self.draft, MainCandidateAction) else {}),
        }


def _text(value: str) -> str:
    return " ".join(value.split()).rstrip("。.!?！？")


def _same_boundaries(first: Candidate, second: Candidate) -> bool:
    a, b = first.proposal, second.proposal
    return (
        (a.decision, a.action_type, a.target, a.visibility) == (b.decision, b.action_type, b.target, b.visibility)
        and _text(first.draft.commitment) == _text(second.draft.commitment)
        and _text(first.draft.condition) == _text(second.draft.condition)
        and a.plan_update == b.plan_update
        and getattr(first.draft, "execution", None) == getattr(second.draft, "execution", None)
    )


def _exact_duplicate(first: Candidate, second: Candidate) -> bool:
    return _same_boundaries(first, second) and _text(first.proposal.action) == _text(second.proposal.action)


def _target_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalize_target(raw, visibility, agents):
    target = raw.strip()
    if not target or any(target == agent.agent_id for agent in agents):
        return target, "", "trim_target" if target != raw else ""
    name = _target_name(target)
    matches = {
        agent.agent_id for agent in agents
        if name in {_target_name(label) for label in (agent.name, *agent.aliases)}
    }
    if len(matches) == 1:
        return next(iter(matches)), "", "resolve_unique_target"
    # A public action may name a group or audience in prose without assigning
    # it a registry seat. Do not guess recipients or clear an ambiguous alias,
    # a fabricated identifier, or any non-public destination.
    if (not matches and visibility == "public" and len(target) <= 300
            and re.fullmatch(r"[^\W\d_][\w .’'()/&-]*", target)
            and "_" not in target and len(target.split()) > 1):
        return "", target, "preserve_public_target_in_text"
    return target, "", ""


def _round_action(draft):
    action = draft.action.strip()
    if draft.decision == "act" or draft.commitment or not draft.condition or not _EXPLICIT_WAIT.match(action):
        return action, ""
    parts = _DEFERRED_CLAUSE.split(action, maxsplit=1)
    if len(parts) != 2 or _OUTWARD_INSTRUCTION.search(parts[0]) or not _LATER_BOUNDARY.search(parts[1]):
        return action, ""
    return parts[0], parts[1]


def _can_normalize_non_action(draft, current_action):
    if draft.decision == "act" or draft.action_type == "no_action":
        return False
    words = set(re.split(r"[\W_]+", _target_name(draft.action_type)))
    outward_label = bool(words & _OUTWARD_LABELS and not words & {"no", "not", "without"})
    # The typed decision owns the wait/act distinction. The free-text label
    # is normalized unless it or the action explicitly describes a new act.
    return not outward_label and not _OUTWARD_INSTRUCTION.search(current_action)


def _build_candidates(pool: CandidatePool, snapshot, agent, phase: str) -> list[Candidate]:
    result = []
    for index, draft in enumerate(pool.candidates, 1):
        fields = draft.model_dump(exclude={"weight", "execution"})
        target, public_audience, target_repair = _normalize_target(draft.target, draft.visibility, snapshot.agents)
        fields["target"] = target
        repairs = [target_repair] if target_repair else []
        current_action, deferred = _round_action(draft)
        if deferred:
            fields["action"] = current_action
            if draft.plan_update is None:
                fields["plan_update"] = CurrentPlan(
                    description=deferred[:500], status="active", review_when=draft.condition,
                    change_reason="Future action was separated from the current-round deferral.",
                    updated_round=snapshot.round_id + 1,
                )
            repairs.append("separate_future_plan")
        if _can_normalize_non_action(draft, current_action):
            fields["action_type"] = "no_action"
            repairs.append("normalize_non_action_type")
        # Keep the substance that distinguished this option available to
        # settlement even if the model put it only in a separate field.
        for label, detail in (("Public audience", public_audience), ("Commitment", draft.commitment), ("Trigger or review condition", draft.condition)):
            if detail and detail not in fields["action"]:
                fields["action"] += f"\n{label} · {detail}"
        proposal = ActionProposal(
            agent_id=agent.agent_id, agent_name=agent.name,
            round_id=snapshot.round_id + 1, snapshot_version=snapshot.snapshot_version,
            action_id=f"action_{snapshot.round_id + 1}_{agent.agent_id}",
            **fields,
        )
        result.append(Candidate(f"{agent.agent_id}:{phase}:{index}", draft, proposal, tuple(repairs)))
    return result


def _decision_context(snapshot, agent, interventions):
    return {
        "public": public_context(snapshot, "decision", agent),
        "actor_private": actor_context(agent, observations=True),
        "visible_interventions": _visible_interventions(interventions, agent),
    }


def _audit_anchors(public: dict, snapshot) -> dict[str, tuple[str | None, str]]:
    anchors = {}
    for event in public["timeline"]:
        # Past events can establish a fact, but cannot establish a continuing
        # constraint or a new actor's intentions by themselves.
        if event.get("description"):
            anchors[f"event:{event['timeline_id']}"] = (None, event["description"])
    for actor in snapshot.agents:
        if not actor.decision_core:
            continue
        for kind in ("limits", "mandate", "decision_rules"):
            for item in getattr(actor.decision_core, kind):
                if (
                    item.id and item.provenance == "source_statement"
                    and item.claim_scope in {"structural", "conditional"}
                    and (item.timeline_refs or item.source_refs)
                ):
                    anchors[f"{kind}:{actor.agent_id}:{item.id}"] = (
                        actor.agent_id, " ".join(filter(None, [item.description, item.conditions]))
                    )
    return anchors


async def _audit(snapshot, candidates, accepted, client, call_prefix, cache):
    public = cited_public_context(snapshot, "audit", [c.proposal for c in candidates])
    anchors = _audit_anchors(public, snapshot)
    owners = {c.proposal.agent_id for c in candidates}
    anchors = {ref: pair for ref, pair in anchors.items() if pair[0] is None or pair[0] in owners}
    # One normal audit call. Split oversized batches by actor/candidate rather
    # than failing solely because each actor now has three options.
    system = (
        "Audit possible actions, not outcome predictions. Allow a surprising or risky action, waiting, "
        "and a proposal requiring another party's consent. A hard_violation requires a specific contradiction "
        "with a supplied fact or binding limit; cite its anchor ID and quote exact text from that anchor. "
        "A historical event alone does not make a temporary restriction permanent. "
        "Use duplicate only for two options from the same actor with the same decision, target, visibility, "
        "substantive commitment and condition. Point to an earlier candidate or an already accepted candidate. "
        "Treat proposal, agreement, and execution as different. If uncertain, allow. Never rewrite an action or its weight. "
        "Review every new candidate exactly once and echo round_id and snapshot_version. "
        "Keep evidence quotes and reasons brief."
    )

    main = isinstance(candidates[0].draft, MainCandidateAction)
    audit_schema = MainCandidateAuditBatch if main else CandidateAuditBatch
    if main:
        system += (
            " In this forecast, audit execution separately from feasibility. If the exact current act "
            "depends on another decision, a trigger, approval or a resource not yet available, name that gate "
            "as missing_prerequisite even if the candidate is incorrectly marked direct. A standing mandate or "
            "a historical example does not establish a current trigger. Set prerequisite_met only when the cited "
            "round-start observation actually establishes the specified condition. A visible private observation "
            "must belong to this actor; the engine checks the reference. An attempt can advance only the stated "
            "preparatory act, even when its expected effect is the final outcome."
        )

    else:
        system = system.replace("Keep evidence quotes and reasons brief.",
                                "For allow, return only candidate_id and verdict; otherwise keep the evidence quote and reason brief.")

    async def review_batch(group, suffix):
        group_owners = {c.proposal.agent_id for c in group}
        scoped = {ref: pair[1] for ref, pair in anchors.items() if pair[0] is None or pair[0] in group_owners}
        payload = {
            "round_id": snapshot.round_id + 1, "snapshot_version": snapshot.snapshot_version,
            "clock": snapshot.world_state.get("simulation_clock", {}),
            "anchors": scoped,
            "already_accepted": [c.summary() for c in accepted if c.proposal.agent_id in group_owners],
            "candidates": [c.summary() for c in group],
        }
        context_limit = getattr(client, "context_max_chars", None)
        prompt = compact_json(payload)
        if context_limit and len(system) + len(prompt) + 350 > context_limit and len(group) > 1:
            pivot = len(group) // 2
            left = await review_batch(group[:pivot], f"{suffix}_1")
            right = await review_batch(group[pivot:], f"{suffix}_2")
            return left | right
        if context_limit and len(system) + len(prompt) + 350 > context_limit:
            cited = {f"event:{ref}" for c in group for ref in c.proposal.timeline_refs}
            payload["anchors"] = {ref: text for ref, text in scoped.items() if not ref.startswith("event:") or ref in cited}
            prompt = compact_json(payload)

        async def call():
            result = await client.json_call(
                call_name=f"{call_prefix}_candidate_audit{suffix}", system=system, prompt=prompt,
                schema=audit_schema, temperature=0.1, max_tokens=min(6500, 300 + (340 if main else 240) * len(group)),
            )
            ids = {review.candidate_id for review in result.reviews}
            if (result.round_id != snapshot.round_id + 1 or result.snapshot_version != snapshot.snapshot_version
                    or ids != {c.id for c in group} or len(ids) != len(result.reviews)):
                raise CandidateAuditMismatch("Candidate audit did not cover the exact candidate batch or used a stale snapshot")
            return result

        result = await cache.model("weighted-audit", payload, call, audit_schema)
        return {review.candidate_id: (review, set(payload["anchors"])) for review in result.reviews}

    reviews = await review_batch(candidates, "")
    return reviews, anchors


def _record_invalid(candidate, code, detail="", review=None):
    row = {"candidate_id": candidate.id, "code": code, "detail": detail}
    if review is not None:
        row["model_review"] = review.model_dump()
    return row


async def _filter_candidates(snapshot, agents, candidates, accepted, client, prefix, cache):
    checks = {}
    viable = []
    available = list(accepted)
    for candidate in candidates:
        agent_id = candidate.proposal.agent_id
        violations = _program_violations(candidate.proposal, agents[agent_id], snapshot)
        if candidate.proposal.decision != "act" and _OUTWARD_INSTRUCTION.search(_round_action(candidate.draft)[0]):
            violations.append("NON_ACTION_CONTAINS_OUTWARD_ACT")
        if violations:
            checks[candidate.id] = _record_invalid(candidate, "program_rejected", ", ".join(violations))
            continue
        duplicate = next((c for c in available if c.proposal.agent_id == agent_id and _exact_duplicate(c, candidate)), None)
        if duplicate:
            checks[candidate.id] = _record_invalid(candidate, "duplicate", duplicate.id)
            continue
        available.append(candidate)
        viable.append(candidate)
    if not viable:
        return [], checks

    reviews, anchors = await _audit(snapshot, viable, accepted, client, prefix, cache)
    retained = list(accepted)
    for candidate in viable:
        review, shown_anchors = reviews[candidate.id]
        if review.verdict == "hard_violation":
            anchor = anchors.get(review.evidence_ref) if review.evidence_ref in shown_anchors else None
            if (anchor and anchor[0] in (None, candidate.proposal.agent_id)
                    and review.evidence_quote.strip() and review.evidence_quote.strip() in anchor[1]
                    and review.reason.strip()):
                checks[candidate.id] = _record_invalid(candidate, "model_rejected", review.evidence_ref, review)
                continue
        if review.verdict == "duplicate":
            original = next((c for c in retained if c.id == review.duplicate_of), None)
            if (original and original.proposal.agent_id == candidate.proposal.agent_id
                    and _same_boundaries(original, candidate) and review.reason.strip()):
                checks[candidate.id] = _record_invalid(candidate, "duplicate", original.id, review)
                continue
        retained.append(candidate)
        checks[candidate.id] = {"candidate_id": candidate.id, "code": "accepted", "model_review": review.model_dump()}
    return retained[len(accepted):], checks


def _draw(seed: int, sample_id: int, round_id: int, agent_id: str, probabilities: list[float]):
    key = compact_json([VERSION, seed, sample_id, round_id, agent_id]).encode("utf-8")
    point = (int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11) / (1 << 53)
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if point < cumulative:
            return index, point
    return max(i for i, p in enumerate(probabilities) if p > 0), point


async def produce_weighted_actions(snapshot, active_agents, selected_actions, client, call_prefix, *, seed, sample_id, cache, selection="weighted", compact_benchmark=False, export_pools=False, record_draws=False, main_forecast=None):
    agents = {agent.agent_id: agent for agent in active_agents}
    if not agents:
        return [], {"not_selected_for_deliberation": len(snapshot.agents)}
    contexts = {agent_id: _decision_context(snapshot, agent, selected_actions) for agent_id, agent in agents.items()}
    if compact_benchmark:
        for context in contexts.values():
            for actor in context["public"].get("actors", []):
                for item in actor.get("decision_boundary", []):
                    item.pop("source_refs", None)
            for field in ("mandate", "means", "resources", "limits", "decision_rules"):
                for item in context["actor_private"].get(field, []):
                    item.pop("source_refs", None)
    targets = {actor.agent_id: actor.name for actor in snapshot.agents}
    pool_schema = MainCandidatePool if main_forecast else CandidatePool
    execution_instructions = (
        " For this forecast set execution.mode on every option. Direct means this exact step can be executed now "
        "with no unresolved prerequisite. Attempt means only the stated preparatory step (seek consent, commission "
        "an assessment, request a decision) happens now; leave the resulting decision or final effect to later. "
        "Conditional means the action requires a specific trigger, approval or resource; provide execution.prerequisite "
        "and only cite a settled public event or your own delivered private observation with a verbatim quote if it "
        "already establishes the condition. Uncertain or missing prerequisites stay conditional. Never infer a trigger "
        "from historical authority. Put waiting review conditions in condition, and for acts put execution prerequisites "
        "only in the structured execution fields. Apply current-round timing when assigning weights."
    ) if main_forecast else ""

    async def propose(agent_id, phase, existing=(), feedback=()):
        payload = {"context": contexts[agent_id]}
        if phase == "supplement":
            payload["retained_options"] = [c.summary() for c in existing]
            payload["feedback"] = list(feedback)
            payload["maximum_new_options"] = MAX_CANDIDATES - len(existing)

        async def call():
            return await client.json_call(
                call_name=f"{call_prefix}_agent_{agent_id}_weighted_{phase}",
                system=(
                    "You are the named bounded scenario actor. Use only the public evidence and your own private state. "
                    "Give one to three feasible actions, using the smallest set that covers the materially different choices actually open now. "
                    "One is sufficient when authority, resources or timing leave only one feasible course. "
                    "Add a second or third only if choosing it could materially change the public outcome, recipient, commitment or next decision. "
                    "Do not split one course into differently worded statements or add passive waiting just to create another option. "
                    "Separate a proposal from unilateral implementation. Vary the commitment, timing, target, or trigger only when "
                    "it changes what you would actually do. Keeping an existing position without a new outward act is no_action. "
                    "For wait, monitor or no_action, always set action_type=no_action and give a review condition. "
                    "A passive review needs no recipient, evidence citation or new plan. In that case use private "
                    "visibility, target='', timeline_refs=[] and no plan_update. Keep the action sentence to the "
                    "current deferral, and put the later trigger to reconsider in condition. "
                    "In action, describe only what happens in this round. Put any act planned for a later round exclusively in plan_update. "
                    "A fresh announcement, response or formal rejection is act, even if the position stays the same. "
                    "For target, copy an exact ID from allowed_target_actors, or use an empty string. "
                    "For a public audience, group or intermediary without a registered ID, put that wording in action, not target. "
                    "A direct action requires one exact registered recipient ID. Never turn a non-action into a statement or other external move. "
                    "Copy evidence IDs from allowed_timeline_refs and use supplied goal IDs only. Check authority, resources, dates and current plan. "
                    "For a restricted decision dossier, choose only moves within the supported scope. A request for authorization or a conditional proposal does not execute the underlying action. Public source passages are background evidence; use a named subject and never assume another actor's power. "
                    "Put any concrete commitment or precondition in the action text itself, so settlement can read it. "
                    "Also return expected_effect as a brief, conditional expectation. The environment will decide "
                    "whether it actually occurs. Use the shared round_briefing to check timing and open requests. "
                    "The observer's forecast question is not your personal objective. A missing outcome measurement "
                    "is not an action constraint; choose using your own goals and actual decision opportunities. "
                    "Give each action a nonnegative relative selection weight (0 to 100) with at least one positive value. "
                    "Explain the relative fit to your beliefs, goals, plan or constraints briefly. Weight is a choice tendency, not "
                    "the action's chance of success, and is distinct from confidence. "
                    "Treat historical restrictions, background reports and dossier evidence gaps as context. "
                    "Use only currently applicable limits or explicit scenario rules as hard restrictions. "
                    "Aim for at most 300 characters per action, 150 per rationale, and 500 per optional plan_update description. "
                    "When supplementing, return only new choices, up to maximum_new_options; do not repeat retained choices. "
                    "Return an empty list if no additional distinct feasible choice exists."
                ) + execution_instructions,
                prompt=compact_json(prompt_payload), schema=pool_schema, temperature=0.55,
                max_tokens=(3100 if main_forecast else 2300) if compact_benchmark else 4100,
            )

        prompt_payload = {
            **payload,
            "allowed_target_actors": targets,
            "allowed_timeline_refs": [item["timeline_id"] for item in contexts[agent_id]["public"]["timeline"]],
        }
        old_key = {"actor_id": agent_id, "phase": phase, "payload": payload}
        pool = None
        if phase == "initial" and not main_forecast:
            # Existing paid drafts can be repaired and re-audited. Audited
            # actions already checkpointed for settlement are reused upstream.
            pool = cache.read("weighted-pool", old_key, CandidatePool.model_validate)
        if pool is None:
            pool = await cache.model(
                "weighted-pool", {**old_key, "revision": CANDIDATE_FORMAT_REVISION, "payload": prompt_payload},
                call, pool_schema,
            )
        drafts = _build_candidates(pool, snapshot, agents[agent_id], phase)
        return drafts[:MAX_CANDIDATES - len(existing)]

    initial_by_actor = await gather_complete(*(propose(agent_id, "initial") for agent_id in agents))
    initial = [candidate for pool in initial_by_actor for candidate in pool]
    accepted, checks = await _filter_candidates(snapshot, agents, initial, [], client, call_prefix, cache)
    final = {agent_id: [c for c in accepted if c.proposal.agent_id == agent_id] for agent_id in agents}
    # Respect an intentional singleton. Repair an empty pool or a set of
    # alternatives that collapsed when checked against the actual state.
    supplement_ids = [
        agent_id for agent_id, drafts in zip(agents, initial_by_actor)
        if not final[agent_id] or (len(final[agent_id]) == 1 and len(drafts) > 1)
    ]

    async def supplement(agent_id):
        feedback = [
            {"action": c.proposal.action, "code": checks[c.id]["code"],
             "detail": checks[c.id].get("detail", ""), "previous_target": c.draft.target,
             "decision": c.draft.decision, "previous_action_type": c.draft.action_type}
            for c in initial if c.proposal.agent_id == agent_id and checks[c.id]["code"] != "accepted"
        ]
        try:
            return await propose(agent_id, "supplement", final[agent_id], feedback), ""
        except LLMError:
            if not final[agent_id]:
                raise
            # An optional call should not discard the action already approved.
            return [], "supplement_model_error"

    supplements = dict(zip(supplement_ids, await gather_complete(*(supplement(agent_id) for agent_id in supplement_ids))))
    added = [candidate for pool, _ in supplements.values() for candidate in pool]
    try:
        added_valid, added_checks = await _filter_candidates(
            snapshot, agents, added, accepted, client, f"{call_prefix}_supplement", cache,
        )
    except (LLMError, CandidateAuditMismatch):
        if any(not final[agent_id] for agent_id in supplement_ids):
            raise
        # Only the already audited options remain eligible after an optional
        # supplemental audit fails. Budget and cancellation still propagate.
        added_valid = []
        added_checks = {candidate.id: _record_invalid(candidate, "supplement_audit_unavailable") for candidate in added}
    checks.update(added_checks)
    for candidate in added_valid:
        final[candidate.proposal.agent_id].append(candidate)

    if compact_benchmark and any(not options for options in final.values()):
        from app.simulation.activation import _seed_candidates

        seeds = _seed_candidates(snapshot, selected_actions)
        optional_triggers = {"initial_deliberation", "periodic_deliberation", "previous_active",
                             "public_decision_opportunity", "periodic_review", "event_affected"}
        for agent_id, options in final.items():
            if options or set(seeds.get(agent_id, {}).get("sources", ())) - optional_triggers:
                continue
            fallback_schema = MainCandidateAction if main_forecast else CandidateAction
            fallback, = _build_candidates(pool_schema(candidates=[fallback_schema(
                decision="no_action", action_type="no_action", action="Take no outward action in this round.",
                visibility="private", condition="Review when a feasible choice or new decision trigger becomes available.",
                reasoning="No candidate passed the current feasibility checks and no active response is required.",
                weight=1, **({"execution": {"mode": "direct"}} if main_forecast else {}),
            )]), snapshot, agents[agent_id], "safe_deferral")
            if not _program_violations(fallback.proposal, agents[agent_id], snapshot):
                added.append(fallback)
                options.append(fallback)
                checks[fallback.id] = {"candidate_id": fallback.id, "code": "accepted", "basis": "safe_deferral"}

    async def weigh(agent_id):
        options = final[agent_id]
        if not options:
            raise ValueError(f"Actor {agent_id} still has no eligible action after supplementation")
        initial_ids = [c.id for c in initial if c.proposal.agent_id == agent_id]
        selected_ids = [c.id for c in options]
        positive = any(c.draft.weight > 0 for c in options)
        weights = [CandidateWeight(candidate_id=c.id, weight=c.draft.weight, reason=c.draft.reasoning) for c in options]
        if len(options) == 1:
            probabilities = [1.0]
        else:
            if initial_ids != selected_ids or not positive:
                # Compare the actual pool together. A removed or duplicated
                # option should not retain or transfer probability mass.
                payload = {"context": contexts[agent_id], "options": [c.summary() for c in options]}

                async def score():
                    result = await client.json_call(
                        call_name=f"{call_prefix}_agent_{agent_id}_final_candidate_weights",
                        system=(
                            "Compare all supplied valid options together from this actor's own BDI, current plan and constraints. "
                            "Give one relative selection weight from 0 to 100 per exact candidate_id, with at least one positive weight. "
                            "Give a short basis. Do not rewrite choices or estimate their chance of success."
                        ),
                        prompt=compact_json(payload), schema=CandidateWeights, temperature=0.1, max_tokens=1300,
                    )
                    ids = {w.candidate_id for w in result.weights}
                    if ids != set(selected_ids) or len(result.weights) != len(ids) or not any(w.weight > 0 for w in result.weights):
                        raise ValueError(f"Actor {agent_id} has missing or zero-total weights for the final candidate pool")
                    return result

                scores = await cache.model("weighted-final-scores", {"actor_id": agent_id, "payload": payload}, score, CandidateWeights)
                by_id = {w.candidate_id: w for w in scores.weights}
                weights = [by_id[c.id] for c in options]
            total = math.fsum(weight.weight for weight in weights)
            probabilities = [weight.weight / total for weight in weights]

        return options, weights, probabilities

    pools = dict(zip(agents, await gather_complete(*(weigh(agent_id) for agent_id in agents))))

    def choose(agent_id):
        options, weights, probabilities = pools[agent_id]
        stratified = (main_forecast.root_draw(pools, actor_id=agent_id, sample_id=sample_id,
                                              probabilities=probabilities, seed=seed)
                      if main_forecast and not snapshot.round_id else None)
        if stratified is not None:
            index, point = stratified
        elif selection == "greedy":
            index, point = max(range(len(probabilities)), key=probabilities.__getitem__), None
        else:
            index, point = _draw(seed, sample_id, snapshot.round_id + 1, agent_id, probabilities)
        chosen = options[index]
        action = chosen.proposal
        execution_status = None
        if main_forecast:
            raw_review = checks[chosen.id].get("model_review")
            review = MainCandidateReview.model_validate(raw_review) if raw_review else None
            action, execution_status = gated_action(chosen, snapshot, agents[agent_id], review)
        audit = {
            "actor_id": agent_id,
            "candidates": [
                {"id": c.id, "draft": c.draft.model_dump(), "review": checks[c.id], "repairs": list(c.repairs)}
                for c in initial + added if c.proposal.agent_id == agent_id
            ],
            "final_pool": [
                {"candidate_id": w.candidate_id, "weight": w.weight, "reason": w.reason, "probability": p}
                for w, p in zip(weights, probabilities)
            ],
            "draw": point, "selected_candidate_id": chosen.id, "candidate_format_revision": CANDIDATE_FORMAT_REVISION,
        }
        if execution_status:
            audit["execution_status"] = execution_status
        if agent_id in supplements:
            audit["supplement_error"] = supplements[agent_id][1]
        if export_pools:
            audit["execution_pool"] = [
                {"id": candidate.id, "action": candidate.proposal.model_dump(mode="json"),
                 "probability": float(i == index) if selection == "greedy" else probabilities[i]}
                for i, candidate in enumerate(options)
            ]
        return action, audit

    chosen = [choose(agent_id) for agent_id in agents]
    audits = [audit for _, audit in chosen]
    counts = Counter(check["code"] for check in checks.values())
    stats = {
        "decisions": len(chosen),
        "multi_candidate_decisions": sum(len(audit["final_pool"]) > 1 for audit in audits),
        "single_candidate_decisions": sum(len(audit["final_pool"]) == 1 for audit in audits),
        "concentrated_decisions": sum(
            len(audit["final_pool"]) > 1 and max(row["probability"] for row in audit["final_pool"]) >= 0.8
            for audit in audits
        ),
        "rejected_candidates": counts["program_rejected"] + counts["model_rejected"],
        "duplicate_candidates": counts["duplicate"],
        "supplemented_decisions": len(supplement_ids),
        "not_selected_for_deliberation": max(0, len(snapshot.agents) - len(agents)),
    }

    async def checkpoint():
        return {"audit": audits, "actions": [action.model_dump() for action, _ in chosen]}

    saved = await cache.run(
        "weighted-draw", {"seed": seed, "sample_id": sample_id, "round_id": snapshot.round_id + 1,
                          "audit": audits}, checkpoint, lambda value: value,
    )
    if export_pools:
        stats["execution_pools"] = [audit["execution_pool"] for audit in saved["audit"]]
    if record_draws:
        stats["sampled_choices"] = [
            {"actor_id": audit["actor_id"], "actor_name": action["agent_name"],
             "candidate_id": audit["selected_candidate_id"],
             **({"execution_status": audit["execution_status"]} if "execution_status" in audit else {}),
             "action": action["action"] if action["visibility"] == "public" else None,
             "p_selected": next(row["probability"] for row in audit["final_pool"]
                                if row["candidate_id"] == audit["selected_candidate_id"]),
             "public_options": [
                 {"candidate_id": row["candidate_id"], "weight": row["weight"], "probability": row["probability"],
                  "action": candidate["draft"]["action"], "selected": row["candidate_id"] == audit["selected_candidate_id"]}
                 for row in audit["final_pool"] for candidate in audit["candidates"]
                 if candidate["id"] == row["candidate_id"] and candidate["draft"]["visibility"] == "public"
             ],
             "public_rejections": [
                 {"candidate_id": candidate["id"], "action": candidate["draft"]["action"],
                  "code": candidate["review"]["code"]}
                 for candidate in audit["candidates"] if candidate["draft"]["visibility"] == "public"
                 and candidate["review"]["code"] != "accepted"
             ]}
            for audit, action in zip(saved["audit"], saved["actions"])
        ]
    return [ActionProposal.model_validate(action) for action in saved["actions"]], stats
