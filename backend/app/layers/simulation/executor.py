"""Ground a round of actions in the members who actually carry them out."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.domain import ActionProposal, WorldSnapshot
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Literal markers, compared verbatim by the inaction-cost rule.
MECHANICAL_ACTIONS = {"no_action", "wait"}

EXECUTOR_ANNOTATION_PROMPT = """You are grounding a round of a scenario simulation in
concrete behaviour. Each agent is a PRINCIPAL actor (an institution, collective,
organization or individual) and it acts as a whole — but its public action is
carried out or voiced BY one of its own members (a head of state, a minister, a
department...). Your job has two parts for each action:

1. pick the member of that principal who would most plausibly perform or
   announce it (or none, when the principal truly acts as a whole);
2. REWRITE the action as that member would actually carry it out — informed by
   the member's capability (what it can do) and style (how it does it).

Event: {event_title}

Principals and their members (each member with its capability and style):
{members_text}

This round's public actions (each by one principal):
{actions_text}

Rules for choosing the member:
- Match the action to the member's documented capability and authority: a
  shipment needs an operations role, a software release needs a release owner,
  a purchase needs a budget holder, a statement may use a spokesperson.
- If several members fit, use the most specific documented responsibility.
  Do not invent authority from seniority, a title or an inferred affiliation.
- If no member plausibly executes it (the principal acts collectively, or the
  action is generic), leave the executor empty.
- Only use names from that principal's member list. Never invent names.
- A principal with no members always gets an empty executor.

Rules for rewriting:
- Keep the SAME substance: same decision, same target, same direction. You are
  re-voicing the act through a specific person, not changing what happened.
- Make it concrete: name the organ or instrument the member would use (a
  ministry, a decree, a press briefing, a directive) and the member's own
  manner as given by its style.
- Keep the original action's scale and severity. A cautious member still
  carries the order out — it may phrase it as procedural, but it does not
  refuse or dilute it.
- One sentence. Write in the same language as the original action text.
- When the executor is empty, return the ORIGINAL action text unchanged.

Return ONLY valid JSON:
{{"executors": [{{"agent": "principal name exactly as in the actions", "action": "the original action text", "executor": "member name or empty string", "concrete_action": "the rewritten action, or the original text when executor is empty"}}]}}

One entry per action, in the same order. Do not include markdown fences, prose outside JSON, comments, or trailing commas."""


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _name_key(value: object) -> str:
    return _clean(value).lower()


def _members_of(snapshot: WorldSnapshot) -> dict[str, list[dict[str, Any]]]:
    """Each principal's members, read from its profile.

    Members never reach ``snapshot.agents`` -- only principals are agents -- so
    they travel on the principal's ``public_state``.
    """
    members: dict[str, list[dict[str, Any]]] = {}
    for agent in snapshot.agents:
        raw = agent.public_state.get("subordinates")
        if isinstance(raw, list) and raw:
            members[agent.name] = [item for item in raw if isinstance(item, dict)]
    return members


def _members_text(members_by_principal: dict[str, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for principal, members in members_by_principal.items():
        rendered: list[str] = []
        for member in members:
            name = _clean(member.get("name"))
            if not name:
                continue
            detail = name
            role = _clean(member.get("role_in_entity"))
            if role:
                detail += f" ({role})"
            capability = _clean(member.get("capability"))
            if capability:
                detail += f" [can: {capability[:120]}]"
            style = _clean(member.get("style"))
            if style:
                detail += f" [style: {style[:120]}]"
            rendered.append(detail)
        lines.append(f"- {principal}: " + (", ".join(rendered) if rendered else "(no members)"))
    return "\n".join(lines)


def _parse_executors(raw: str) -> dict[str, dict[str, str]]:
    """Read the model's per-action answers. Returns a map keyed by agent name."""
    text = _clean(raw)
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            return {}
    items = parsed.get("executors") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return {}
    result: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        agent = _clean(item.get("agent"))
        if not agent:
            continue
        result[_name_key(agent)] = {
            "executor": _clean(item.get("executor")),
            "concrete_action": _clean(item.get("concrete_action")),
        }
    return result


async def annotate_executors(
    snapshot: WorldSnapshot,
    proposals: list[ActionProposal],
    client: LLMClient,
    call_prefix: str,
) -> list[ActionProposal]:
    """Re-voice each proposal through the member that carries it out.

    Returns new proposals; the inputs are left untouched. Anything that cannot
    be grounded -- no members, a failed call, an unknown executor name -- keeps
    the original wording, so this step can only ever make an action more
    specific, never remove or replace it.
    """
    members_by_principal = _members_of(snapshot)
    external = [p for p in proposals if p.decision == 'act']
    if not external or not members_by_principal:
        return proposals

    actions_text = "\n".join(
        f"- {proposal.agent_name}: {proposal.action[:200]}" for proposal in external
    )
    prompt = EXECUTOR_ANNOTATION_PROMPT.format(
        event_title=snapshot.event_title,
        members_text=_members_text(members_by_principal),
        actions_text=actions_text,
    )
    try:
        raw = await client.text_call(
            call_name=f"{call_prefix}_executor_annotation",
            prompt=prompt,
            max_tokens=3000,
            temperature=0.2,
        )
    except Exception:
        # Annotation is an enhancement: a failure leaves the abstract action,
        # which the settlement handles exactly as well.
        logger.warning("executor annotation failed; keeping original actions", exc_info=True)
        return proposals

    parsed = _parse_executors(raw)
    if not parsed:
        return proposals

    known_members = {
        principal: {_name_key(member.get("name")) for member in members}
        for principal, members in members_by_principal.items()
    }
    annotated: list[ActionProposal] = []
    for proposal in proposals:
        if proposal.decision != 'act':
            annotated.append(proposal)
            continue
        answer = parsed.get(_name_key(proposal.agent_name))
        if not answer:
            annotated.append(proposal)
            continue
        executor = answer["executor"]
        # An executor must be one of this principal's own members.
        if executor and _name_key(executor) not in known_members.get(proposal.agent_name, set()):
            executor = ""
        if proposal.action.strip().lower() in MECHANICAL_ACTIONS:
            # Keep the marker verbatim; the inaction-cost rule compares it
            # literally and prose would silently disable that rule.
            annotated.append(proposal)
            continue
        concrete = answer["concrete_action"] if executor else ""
        if not concrete:
            annotated.append(proposal)
            continue
        updated = proposal.model_copy(deep=True)
        updated.action = concrete
        annotated.append(updated)
    return annotated
