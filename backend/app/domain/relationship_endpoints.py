"""Resolving relationship endpoints onto canonical agent ids."""

from __future__ import annotations

import logging
from typing import Any

from app.domain.identity import identity_key

logger = logging.getLogger(__name__)


def member_index(agents: list[Any]) -> dict[str, str]:
    """Map every member name onto the id of the principal it belongs to.

    Members live on their principal's ``public_state["subordinates"]``. A
    relationship naming a member is really about the principal that acts
    through them, so that is the endpoint it resolves to.
    """
    index: dict[str, str] = {}
    for agent in agents:
        agent_id = str(_field(agent, "agent_id") or "")
        if not agent_id:
            continue
        state = _field(agent, "public_state") or {}
        if not isinstance(state, dict):
            continue
        for member in state.get("subordinates") or []:
            if not isinstance(member, dict):
                continue
            for key in ("name", "agent_id"):
                key_text = identity_key(member.get(key))
                if key_text:
                    index.setdefault(key_text, agent_id)
    return index


def describe_agents(agents: list[Any], *, members: bool) -> list[dict[str, Any]]:
    """The agent cards the relationship prompt shows.

    ``members=True`` appends each principal's member names, because the settled
    events name members and a model that cannot see them has no way to write a
    resolvable endpoint -- it invents a name instead.
    """
    cards: list[dict[str, Any]] = []
    for agent in agents:
        card = {
            "agent_id": _field(agent, "agent_id"),
            "name": _field(agent, "name"),
            "aliases": list(_field(agent, "aliases") or []),
            "role": _field(agent, "role"),
            "title": _field(agent, "title"),
            "affiliation": _field(agent, "affiliation"),
        }
        if members:
            state = _field(agent, "public_state") or {}
            names = [
                str(m.get("name"))
                for m in (state.get("subordinates") or [] if isinstance(state, dict) else [])
                if isinstance(m, dict) and m.get("name")
            ]
            card["members"] = names
        cards.append(card)
    return cards


def resolve_endpoint(raw: str, lookup: dict[str, str]) -> str:
    """The canonical agent id for a written endpoint, or "" if there is none."""
    return lookup.get(identity_key(raw), "")


def agent_lookup(agents: list[Any]) -> dict[str, str]:
    """Map every name / alias / id an endpoint might be written with onto the id."""
    lookup: dict[str, str] = {}
    for agent in agents:
        agent_id = str(_field(agent, "agent_id") or "")
        if not agent_id:
            continue
        for candidate in (agent_id, _field(agent, "name"), *(_field(agent, "aliases") or [])):
            key = identity_key(candidate)
            if key:
                lookup.setdefault(key, agent_id)
    return lookup


def clean_relationships(
    relationships: list[Any],
    *,
    agents: list[Any],
    known_timeline: set[str],
    call_name: str,
) -> list[Any]:
    """Drop the endpoints and evidence this round cannot support.

    Returns the relationships worth keeping, with each endpoint rewritten to a
    canonical id. Dropped entries are logged with their reason: silently losing
    a graph edge is worse than the edge being absent, because the reader cannot
    tell the difference between "no such relationship" and "we threw it away".
    """
    lookup = agent_lookup(agents)
    members = member_index(agents)
    kept: list[Any] = []
    for relationship in relationships:
        source = resolve_endpoint(str(_field(relationship, "source") or ""), lookup)
        target = resolve_endpoint(str(_field(relationship, "target") or ""), lookup)
        if not source:
            source = resolve_endpoint(str(_field(relationship, "source") or ""), members)
        if not target:
            target = resolve_endpoint(str(_field(relationship, "target") or ""), members)
        if not source or not target or source == target:
            logger.warning(
                "%s dropped a relationship with an unresolvable endpoint: %r -> %r",
                call_name, _field(relationship, "source"), _field(relationship, "target"),
            )
            continue
        evidence = _field(relationship, "evidence") or []
        kept_evidence = [
            item for item in evidence
            if str(_field(item, "timeline_id") or "") in known_timeline
        ]
        if not kept_evidence:
            logger.warning(
                "%s dropped a relationship with no known timeline evidence: %r -> %r",
                call_name, _field(relationship, "source"), _field(relationship, "target"),
            )
            continue
        _set(relationship, "source", source)
        _set(relationship, "target", target)
        _set(relationship, "evidence", kept_evidence)
        kept.append(relationship)
    return kept


def _field(item: Any, name: str) -> Any:
    """Read a field off a pydantic model or a plain dict."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _set(item: Any, name: str, value: Any) -> None:
    if isinstance(item, dict):
        item[name] = value
    else:
        setattr(item, name, value)
