"""Activation/projection of compact dossiers; stored source artifacts stay intact."""
import re

from app.domain.agents import Beliefs, SituationView
from app.domain.behavior import BehavioralModel
from app.domain.decision_core import legacy_core, applicable_core

_CURRENT_RESTRICTION = re.compile(
    r"\b(?:maintains?|enforces?|currently (?:has|imposes?)|(?:is|are) still subject to)\b[^.;]{0,90}"
    r"\b(?:ban|restriction|prohibition)\b|\b(?:ban|restriction|prohibition)\b[^.;]{0,60}"
    r"\b(?:remains? (?:active|in force|in effect|in place)|(?:is|are) (?:still |currently )?in (?:force|effect))\b"
    r"|(?:仍然?有效|仍在生效|目前实施|继续实施)[^。；]{0,25}(?:禁令|限制)"
    r"|(?:禁令|限制)[^。；]{0,25}(?:仍然?有效|仍在生效|仍然?适用)",
    re.I,
)
_UNCERTAIN_RESTRICTION = re.compile(
    r"\b(?:no evidence|unknown|unclear|uncertain|unverified|unconfirmed|not established)\b"
    r"|没有证据|尚未核实|尚不明确|未知|无法确认",
    re.I,
)


def current_constraint_beliefs(beliefs, core, on_date=""):
    """A belief can cite an admitted limit, but cannot establish a binding one."""
    accepted = applicable_core(core, on_date[:10]) if core else None
    limits = {item.id: item for item in accepted.limits
              if item.provenance == "source_statement" and (item.source_refs or item.timeline_refs)
              and (item.claim_scope == "structural" or (item.valid_from and item.valid_until))} if accepted else {}
    changes, rejected = {}, []
    for field in ("situation_views", "actor_models"):
        rows = []
        for item in getattr(beliefs, field):
            if item.constraint_ref:
                limit = limits.get(item.constraint_ref)
                if limit:
                    assessment = " ".join(filter(None, [limit.description, limit.conditions]))
                    rows.append(item.model_copy(update={"assessment": assessment, "timeline_refs": limit.timeline_refs}))
                else:
                    rejected.append(item.constraint_ref)
            else:
                unsupported = any(
                    _CURRENT_RESTRICTION.search(sentence) and not _UNCERTAIN_RESTRICTION.search(sentence)
                    for sentence in re.split(r"[.!?;。！？；]+", item.assessment)
                )
                if unsupported:
                    rejected.append("missing_current_constraint_reference")
                else:
                    rows.append(item)
        changes[field] = rows
    return beliefs.model_copy(update=changes), rejected


def initial_beliefs(core):
    return Beliefs(situation_views=[
        SituationView(topic="current assessment", assessment=item.description, timeline_refs=item.timeline_refs)
        for item in core.initial_judgments
        if item.perspective == "actor" and (item.timeline_refs or item.source_refs)
    ])


def initial_plan(core, on_date=""):
    from app.domain.decision_core import CurrentPlan

    boundary = (on_date or core.as_of)[:10]
    for item in core.initial_intentions:
        if (item.provenance == "source_statement" and item.claim_scope == "conditional"
                and (item.timeline_refs or item.source_refs)
                and (not boundary or ((not item.valid_from or item.valid_from <= boundary)
                                      and (not item.valid_until or boundary <= item.valid_until)))):
            return CurrentPlan(description=item.description, timeline_refs=item.timeline_refs,
                               review_when=item.review_condition or item.conditions)
    return None


def dossier_evidence_gaps(core):
    observer_gaps = [item.unknowns or item.description for item in core.initial_judgments if item.perspective == "observer"]
    return list(dict.fromkeys([*core.evidence_gaps, *observer_gaps]))[:5]


def activate_decision_core(agent):
    if agent.decision_core is None:
        agent.decision_core = legacy_core(agent.model_dump())
    agent.decision_core = applicable_core(agent.decision_core, agent.decision_core.as_of)
    if agent.decision_core.seat and not agent.last_information_fingerprint:
        agent.beliefs = initial_beliefs(agent.decision_core)
    if agent.decision_core.origin == "legacy_projection":
        # These were identity/stance seed placeholders, not substantive beliefs.
        agent.beliefs = Beliefs(
            situation_views=[v for v in agent.beliefs.situation_views
                             if v.topic != "stance" and v.assessment.strip() != agent.description.strip()],
            actor_models=[v for v in agent.beliefs.actor_models
                          if v.agent_name != "the other actors in this event"])
        for goal in agent.desires:
            goal.priority = None
    # Old numeric ladders and sensitive-point "constraints" remain in the raw
    # archive, not in new decision/audit contexts. Real scenario rules still apply.
    agent.capabilities = []
    agent.constraints = []
    agent.behavioral_model = BehavioralModel()
    agent.public_state = {k:v for k,v in agent.public_state.items()
                          if k in {"profile_kind", "knowledge_boundary", "subordinates"}}
    return agent


def core_context(agent):
    core = agent.decision_core
    goals = {g.id:g for g in core.goals}
    beliefs = current_constraint_beliefs(agent.beliefs, core)[0].model_dump(exclude_none=True, exclude={"desires"})
    for kind in ("situation_views", "actor_models"):
        for view in beliefs.get(kind, []):
            view.pop("confidence", None)
    return {
        "agent_id":agent.agent_id,"name":agent.name,"agent_type":agent.agent_type,
        "decision_subject":agent.decision_subject,
        "decision_scope":core.decision_scope,
        "evidence_gaps":dossier_evidence_gaps(core),
        "decision_seat": core.seat.model_dump() if core.seat else None,
        "evidence_gap_scope":{"as_of":core.as_of, "status":"unverified_against_current_evidence",
                              "policy":"These gaps describe the dossier input, not the current world's limits. Recheck the shared public sources and retrieved evidence. Missing information is not a prohibition, a lack of authority, or proof of absence."},
        "dossier_version":core.version,"dossier_origin":core.origin,"as_of":core.as_of,
        "mandate":[v.model_dump(exclude_defaults=True) for v in core.mandate],
        "goals":[{"id":g.desire_id,"description":g.description,"status":g.status,
                  **({"priority":goals[g.desire_id].priority,"tradeoff":goals[g.desire_id].tradeoff,
                      "success_condition":goals[g.desire_id].success_condition,
                      "timeline_refs":goals[g.desire_id].timeline_refs,
                      "provenance":goals[g.desire_id].provenance} if g.desire_id in goals else {"priority":"unknown"})}
                 for g in agent.desires],
        "means":[v.model_dump(exclude_defaults=True) for v in core.means],
        "resources":[v.model_dump(exclude={"value"},exclude_defaults=True) for v in core.resources],
        "resource_state_location":f"world_state.actor_resources.{agent.agent_id}",
        "limits":[v.model_dump(exclude_defaults=True) for v in core.limits],
        "decision_rules":[v.model_dump(exclude_defaults=True) for v in core.decision_rules],
        "beliefs":beliefs,"current_plan":agent.current_plan.model_dump(exclude_none=True) if agent.current_plan else None,
        "interpretation":"Unreviewed legacy limits are clues, not hard prohibitions. Unknown resources/authority stay unknown. Only configured program rules are mechanically enforced. Reassess old information-absence judgments against the current public packet; do not let them override present source data. Preserve uncertainty without inventing powers or quantities.",
    }
