"""Strength settlement: evolve an agent's capability levels after a settlement."""

from __future__ import annotations

import re
from typing import Any

from app.domain.agents import AgentProfile, Capability

# The strength ladder, weakest to strongest, matching the words the profile stage
# writes.
POWER_LEVELS = ("very low", "low", "moderate", "high", "very high")

_LEVEL_TO_NUMBER = {
    "very low": 0.1,
    "low": 0.25,
    "moderate": 0.5,
    "high": 0.75,
    "very high": 0.9,
}

# The profile stage also writes these for the same rungs.
_LEVEL_SYNONYMS = {
    "weak": "low",
    "medium": "moderate",
    "strong": "high",
    "dominant": "very high",
}


def canonical_level(level: object) -> str:
    """Map any profile wording onto one of ``POWER_LEVELS``."""
    text = str(level or "").strip().lower()
    text = _LEVEL_SYNONYMS.get(text, text)
    return text if text in POWER_LEVELS else "moderate"


def level_to_number(level: str) -> float:
    """Map a strength level onto the forecast's 0-1 capability scale."""
    return _LEVEL_TO_NUMBER[canonical_level(level)]


def _name_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _dimension_key(value: object) -> str:
    """Normalize a dimension name for comparison.

    Dimension names come from the profile stage and may be written in any
    language the event uses, so this keeps word characters generally rather
    than only ASCII -- stripping non-ASCII would collapse every CJK dimension
    name to the empty string and silently match nothing.
    """
    return re.sub(r"[^\w]+", "", str(value or "").lower(), flags=re.UNICODE)


def _find_agent(agents: list[AgentProfile], raw_name: str) -> AgentProfile | None:
    """Match the impact's agent name onto a profile, the way the loop does."""
    key = _name_key(raw_name)
    if not key:
        return None
    known = {_name_key(agent.name): agent for agent in agents}
    known.update({_name_key(alias): agent for agent in agents for alias in agent.aliases})
    known.update({_name_key(agent.agent_id): agent for agent in agents})
    if key in known:
        return known[key]
    for known_key, agent in known.items():
        if len(key) >= 4 and len(known_key) >= 4 and (key in known_key or known_key in key):
            return agent
    return None


_MIN_PARTIAL_MATCH = 2


def _find_capability(agent: AgentProfile, raw_dimension: str) -> Capability | None:
    """Match the impact's dimension onto one of the agent's capabilities.

    Exact match first (after folding case and punctuation). Only then a
    partial match, in either direction, and only when the shorter of the two
    names is at least two characters -- the settlement may name a dimension a
    little longer or shorter than the profile wrote it, but one character
    matches almost anything.
    """
    key = _dimension_key(raw_dimension)
    if not key:
        return None
    for capability in agent.capabilities:
        if _dimension_key(capability.name) == key:
            return capability
    if len(key) < _MIN_PARTIAL_MATCH:
        return None
    for capability in agent.capabilities:
        known = _dimension_key(capability.name)
        if len(known) < _MIN_PARTIAL_MATCH:
            continue
        if key in known or known in key:
            return capability
    return None


def _current_level(agent: AgentProfile, capability: Capability) -> str:
    """The capability's level as a word, from the numeric value it is stored as."""
    number = float(capability.level)
    closest = min(POWER_LEVELS, key=lambda level: abs(_LEVEL_TO_NUMBER[level] - number))
    return closest


def apply_power_impacts(
    agents: list[AgentProfile],
    impacts: list[dict[str, Any]],
    round_number: int,
    *,
    applied_log: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Apply one settlement's strength verdicts to the agents' capabilities.

    Returns the applied changes, one entry per dimension that actually moved,
    each shaped as ``{agent_id, dimension, from, to, reason, round}`` so the
    round snapshot can show why an actor's strength shifted.

    ``applied_log`` optionally carries per-agent history across rounds; entries
    are appended to it when supplied.
    """
    changes: list[dict[str, Any]] = []
    # Dimensions already settled this round.
    settled: set[tuple[str, str]] = set()
    for impact in impacts or []:
        # The settlement hands these over as pydantic models; callers that
        # built one by hand pass plain dicts. Accept both, or the whole step
        # silently does nothing on the production path.
        if hasattr(impact, "model_dump"):
            impact = impact.model_dump()
        if not isinstance(impact, dict):
            continue
        agent = _find_agent(agents, str(impact.get("agent") or ""))
        if agent is None:
            continue
        capability = _find_capability(agent, str(impact.get("dimension") or ""))
        if capability is None:
            # The dimension is not part of this event's strength set. Adding it
            # would make this agent's levels incomparable with everyone else's,
            # so the impact is dropped rather than seeded.
            continue

        history = applied_log.setdefault(agent.agent_id, []) if applied_log is not None else []
        # One move per dimension per round: a second impact on the same dimension this
        # round is ignored, however it is phrased.
        key = (agent.agent_id, _dimension_key(capability.name))
        if key in settled or any(
            item.get("round") == round_number
            and _dimension_key(item.get("dimension")) == _dimension_key(capability.name)
            for item in history
        ):
            continue
        settled.add(key)

        status = str(impact.get("status") or "").strip().lower()
        step = 1 if status.startswith("strengthen") else -1 if status.startswith("weaken") else 0
        current = _current_level(agent, capability)
        index = POWER_LEVELS.index(current)
        new_index = max(0, min(len(POWER_LEVELS) - 1, index + step))
        reason = str(impact.get("reason") or impact.get("new_description") or "").strip()[:200]

        entry = {
            "round": round_number,
            "dimension": capability.name,
            "from": current,
            "to": POWER_LEVELS[new_index],
            "reason": reason,
        }
        history.append(entry)
        if step == 0:
            # A maintained dimension is recorded so it cannot move again this
            # round, but nothing changed on the agent.
            continue
        capability.level = level_to_number(POWER_LEVELS[new_index])
        changes.append({"agent_id": agent.agent_id, **entry})
    return changes
