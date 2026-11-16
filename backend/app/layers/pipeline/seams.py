"""Seams between the framework and the layered-figure stage."""

from __future__ import annotations
from app.domain.behavior import grounded_behavior
from app.domain.agency import resolve_entity_type

import logging
from typing import Any

from app.domain.agents import (
    ActorModel,
    AgentProfile,
    Beliefs,
    Capability,
    DecisionPolicy,
    Desire,
    SituationView,
)
from app.domain.identity import identity_key, stable_agent_id
from app.layers.support.llm_normalizers import _clean_text, _normalize_stance
from app.layers.support.semantic_view import normalize_semantic_view

logger = logging.getLogger(__name__)


def _actor_key(value: object) -> str:
    """Comparison key for actor names (case, spacing and punctuation folded)."""
    return identity_key(_clean_text(value))


_KEPT_ENTRY_FIELDS = (
    "timestamp",
    "date_precision",
    "date_original",
    "time_role",
    "modality",
    "date_evidence",
    "date_range",
    "event_type",
    "confidence",
    "importance_score",
    "evidence_text",
)


def refresh_stored_timeline(
    entries: list[dict[str, Any]],
    timeline: list[Any],
    events: Any,
    event_id: str,
) -> None:
    """Rewrite ``timeline.json`` from the finished stage entries."""
    stored: list[dict[str, Any]] = []
    for item, entry in _pair_entries(entries, timeline):
        event = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        row = dict(event)
        if entry is not None:
            for field in _KEPT_ENTRY_FIELDS:
                if field in entry:
                    row[field] = entry[field]
            # ``actors`` stays as apply_stage_extraction wrote it (forecast
            # agent ids); the retrieval layer gets the scored names instead.
            row["actors"] = list(entry.get("actors") or [])
            actors = _retrieval_actors(entry)
            if actors:
                row["retrieval_actors"] = actors
        stored.append(row)
    try:
        events.save_artifact(event_id, "timeline", stored)
    except Exception:  # noqa: BLE001
        logger.warning("Could not refresh the stored timeline artifact", exc_info=True)


def _pair_entries(
    entries: list[dict[str, Any]],
    timeline: list[Any],
) -> list[tuple[Any, dict[str, Any] | None]]:
    """Pair each framework event with the stage entry built from it."""
    if len(entries) == len(timeline):
        return list(zip(timeline, entries))
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        by_id.setdefault(_clean_text(entry.get("timeline_id")), entry)
    return [
        (item, by_id.get(_clean_text(getattr(item, "timeline_id", ""))))
        for item in timeline
    ]


# ── in: framework timeline → stage entries ─────────────────────────────────


def to_stage_timeline(timeline: list[Any]) -> list[dict[str, Any]]:
    """Convert framework ``TimelineEvent``s into the stage body's dicts."""
    entries: list[dict[str, Any]] = []
    for item in timeline:
        event = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        timeline_id = _clean_text(event.get("timeline_id"))
        entry = {
            "timeline_id": timeline_id,
            "event_id": timeline_id,
            "timestamp": _clean_text(event.get("timestamp")),
            "description": _clean_text(event.get("description")),
            "date_precision": event.get("date_precision") or "approximate",
            "date_original": event.get("date_original") or "",
            "time_role":event.get("time_role") or "occurrence",
            "modality":event.get("modality") or "occurred",
            "date_evidence":event.get("date_evidence") or {},
            "date_range": event.get("date_range") or {},
            "event_type": event.get("event_type") or "other",
            "confidence": event.get("confidence", 0.7),
            "importance_score": event.get("importance", event.get("importance_score", 0.5)),
            "evidence_text": event.get("evidence_text") or "",
            "source_sections": list(event.get("source_sections") or []),
            "actors": list(event.get("actors") or []),
            "figures": [],
            "actions": [],
        }
        entries.append(entry)
    return entries


def apply_stage_extraction(
    entries: list[dict[str, Any]],
    timeline: list[Any],
    *,
    event_id: str = "",
    event_title: str = "",
    profiles: list[dict[str, Any]] | None = None,
) -> None:
    """Copy the stage's extracted figures/actions back onto the framework events."""
    # The stage's own ids (``yoon_suk_yeol``) are slugs, while the forecast addresses
    # agents by ``stable_agent_id``.
    forecast_id_by_stage_id: dict[str, str] = {}
    forecast_id_by_name: dict[str, str] = {}
    for profile in profiles or []:
        if not isinstance(profile, dict):
            continue
        if str(profile.get("agent_level") or "principal").lower() == "member":
            continue
        name = _clean_text(profile.get("source_figure"))
        if not name:
            continue
        forecast_id = stable_agent_id(event_id, event_title, name)
        forecast_id_by_name[_actor_key(name)] = forecast_id
        stage_id = _clean_text(profile.get("agent_id"))
        if stage_id:
            forecast_id_by_stage_id[_actor_key(stage_id)] = forecast_id

    for item, entry in _pair_entries(entries, timeline):
        if entry is None:
            continue
        resolved: list[str] = []
        unresolved: list[str] = []
        for figure in entry.get("figures") or []:
            if not isinstance(figure, dict):
                continue
            name = _clean_text(figure.get("name"))
            if not name:
                continue
            forecast_id = (
                forecast_id_by_name.get(_actor_key(name))
                or forecast_id_by_stage_id.get(_actor_key(figure.get("id")))
            )
            if forecast_id:
                if forecast_id not in resolved:
                    resolved.append(forecast_id)
            elif name not in unresolved:
                unresolved.append(name)
        item.actors = resolved + unresolved
        item.actions = [
            dict(action) for action in entry.get("actions") or [] if isinstance(action, dict)
        ]


# Scoring metadata ``_score_figures`` produces per actor.
_RETRIEVAL_ENTRY_FIELDS = (
    "entity_type",
    "role",
    "role_in_event",
    "title",
    "affiliation",
    "core_actor_score",
    "importance_score",
    "parent_actor",
    "parent_actor_type",
    "aggregation_category",
    "evidence_count",
    "latest_action_ts",
)


def _retrieval_actor(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """One actor as the retrieval layer wants it: the name plus its scores."""
    return {"name": name, **{f: entry[f] for f in _RETRIEVAL_ENTRY_FIELDS if f in entry}}


def _retrieval_actors(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The stage's ``figures`` for one entry, keyed back to their canonical names.

    ``_build_figure_context`` rewrites every entry's ``figures`` to canonical
    records and merges the per-entry scoring metadata onto each one, so this is
    where the canonical name and the scores meet.
    """
    actors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for figure in entry.get("figures") or []:
        if not isinstance(figure, dict):
            continue
        name = _clean_text(figure.get("canonical_name") or figure.get("name"))
        if not name or name in seen:
            continue
        seen.add(name)
        actors.append(_retrieval_actor(name, figure))
    return actors


def export_stage_retrieval_context(
    entries: list[dict[str, Any]],
    timeline: list[Any],
) -> None:
    """Move the stage's per-entry scoring onto the framework events."""
    for item, entry in _pair_entries(entries, timeline):
        if entry is None:
            continue
        actors = _retrieval_actors(entry)
        if actors:
            item.retrieval_actors = actors
        scores = {f: entry[f] for f in ("importance_score", "event_type") if f in entry}
        if scores:
            item.retrieval_scores = scores


# ── out: stage profiles → framework AgentProfile ───────────────────────────


def _semantic_of(profile: dict[str, Any]) -> dict[str, Any]:
    semantic = profile.get("semantic")
    return semantic if isinstance(semantic, dict) else {}


def _identity_of(profile: dict[str, Any]) -> dict[str, Any]:
    identity = _semantic_of(profile).get("identity")
    return identity if isinstance(identity, dict) else {}


def _views_of(profile: dict[str, Any]) -> dict[str, Any]:
    views = _semantic_of(profile).get("views")
    return views if isinstance(views, dict) else {}


def _role_of(profile: dict[str, Any]) -> str:
    """The agent's role in the event, across the three profile forms."""
    source = profile.get("source_profile")
    role = _clean_text((source or {}).get("role_in_event") if isinstance(source, dict) else "")
    if role:
        return role
    identity = _identity_of(profile)
    return _clean_text(identity.get("event_role")) or _clean_text(identity.get("position"))


def _stance_of(profile: dict[str, Any], role: str) -> str:
    """The agent's stance as one of the five labels the UI colours by."""
    labelled = _normalize_stance(profile.get("role_label"))
    if labelled:
        return labelled
    stored = _normalize_stance(profile.get("stance"))
    if stored:
        return stored
    from app.layers.pipeline.helpers import _infer_stance_from_context

    episodic = profile.get("episodic")
    episodic = episodic if isinstance(episodic, dict) else {}
    name = _clean_text(profile.get("source_figure") or profile.get("canonical_name"))
    return _infer_stance_from_context(
        name,
        role,
        episodic.get("actions_initiated") or [],
        episodic.get("actions_received") or [],
    )


def _title_of(profile: dict[str, Any]) -> str:
    identity = _identity_of(profile)
    return _clean_text(identity.get("position")) or _clean_text(profile.get("title"))


def _affiliation_of(profile: dict[str, Any]) -> str:
    identity = _identity_of(profile)
    for key in ("party", "affiliation", "country"):
        value = _clean_text(identity.get(key))
        if value:
            return value
    return _clean_text(profile.get("affiliation"))


def _description_of(profile: dict[str, Any]) -> str:
    """A non-empty one-line summary; ``AgentProfile.description`` is required."""
    from app.layers.pipeline.profile_service import _summary_from_semantic

    summary = _clean_text(_summary_from_semantic(_semantic_of(profile)))
    if summary:
        return summary
    identity = _identity_of(profile)
    name = _clean_text(profile.get("source_figure")) or "This actor"
    role = _role_of(profile) or _clean_text(identity.get("nature"))
    return f"{name}{(' — ' + role) if role else ''}".strip(" —")


def _beliefs_of(profile: dict[str, Any], name: str) -> Beliefs:
    """The agent's opening beliefs — the B in BDI, for the first round."""
    from app.layers.pipeline.profile_service import _summary_from_semantic

    views: list[SituationView] = []
    summary = _clean_text(_summary_from_semantic(_semantic_of(profile)))
    if summary:
        views.append(
            SituationView(
                topic=_role_of(profile) or name,
                assessment=summary,
                confidence=0.5,
                timeline_refs=_timeline_refs(profile)[:6],
            )
        )
    stance = _clean_text(_views_of(profile).get("stance"))
    if stance:
        views.append(
            SituationView(
                topic="stance",
                assessment=stance,
                confidence=0.5,
                timeline_refs=_timeline_refs(profile)[:6],
            )
        )
    return Beliefs(situation_views=views, actor_models=_actor_models_of(profile))


def _actor_models_of(profile: dict[str, Any]) -> list[ActorModel]:
    """What this actor's profile says about the peers it shares the event with.

    Only the aggregate view is available at initialisation: the profile stage
    writes one stance per actor, not a judgement of each counterpart, so this
    is deliberately a single assessment of the field rather than invented
    per-actor models. Round two replaces the block wholesale once the agent has
    actually observed something.
    """
    stance = _clean_text(_views_of(profile).get("stance"))
    if not stance:
        return []
    return [
        ActorModel(
            agent_name="the other actors in this event",
            assessment=stance,
            confidence=0.4,
            timeline_refs=_timeline_refs(profile)[:6],
        )
    ]


def _historical_actions(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """The agent's past actions, shaped the way the forecast reads them."""
    episodic = profile.get("episodic")
    if not isinstance(episodic, dict):
        return []
    actions: list[dict[str, Any]] = []
    for action in episodic.get("actions_initiated") or []:
        if not isinstance(action, dict):
            continue
        actions.append(
            {
                "action": _clean_text(action.get("action")),
                "target": _clean_text(action.get("target")),
                "timestamp": _clean_text(action.get("timestamp")),
                "description": _clean_text(action.get("description")),
            }
        )
    return actions


def _timeline_refs(profile: dict[str, Any]) -> list[str]:
    """Timeline ids this agent appears in, for the forecast's evidence links."""
    refs: list[str] = []
    seen: set[str] = set()
    episodic = profile.get("episodic")
    if not isinstance(episodic, dict):
        return refs
    for bucket in ("history", "actions_initiated", "actions_received"):
        for item in episodic.get(bucket) or []:
            if not isinstance(item, dict):
                continue
            ref = _clean_text(item.get("timeline_id"))
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def _capabilities_of(profile: dict[str, Any]) -> list[Capability]:
    """The strength block, as the forecast's capability list.

    This is the join point for the power dimensions: the stage generates one
    shared dimension set per event, and each profile records its level per
    dimension. Levels arrive as words and become the 0-1 scale the forecast
    uses, so a later settlement can step them.
    """
    strength = _semantic_of(profile).get("strength")
    if not isinstance(strength, dict):
        return []
    capabilities: list[Capability] = []
    for name, entry in strength.items():
        if name == "as_of" or not isinstance(entry, dict):
            continue
        level = _level_to_float(entry.get("level"))
        capabilities.append(Capability(name=_clean_text(name), level=level))
    return capabilities


_LEVEL_WORDS = {
    "very low": 0.1,
    "low": 0.25,
    "weak": 0.25,
    "moderate": 0.5,
    "medium": 0.5,
    "high": 0.75,
    "strong": 0.75,
    "very high": 0.9,
    "dominant": 0.9,
}


def _level_to_float(value: object) -> float:
    """Map a strength level onto the forecast's 0-1 capability scale."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    text = _clean_text(value).lower()
    if text in _LEVEL_WORDS:
        return _LEVEL_WORDS[text]
    for word, level in _LEVEL_WORDS.items():
        if word in text:
            return level
    return 0.5


def _public_state(profile: dict[str, Any]) -> dict[str, Any]:
    """Fields the forecast does not model but the UI and later stages read.

    The three-form profile body is carried whole so nothing is lost: the
    forecast serialises the agent into its prompts, so the narrative profile
    stays visible to the model without the forecast having to know its shape.
    """
    state: dict[str, Any] = {
        "profile_kind": _clean_text(profile.get("profile_kind")),
        "knowledge_boundary": _clean_text(profile.get("knowledge_boundary")),
        "stance": _clean_text(_views_of(profile).get("stance")),
        "summary": _clean_text(profile.get("summary")),
        "semantic": _semantic_of(profile),
    }
    source = profile.get("source_profile")
    if isinstance(source, dict):
        state["source_profile"] = source
    return {key: value for key, value in state.items() if value not in ("", {}, None)}


def _line_from(item: Any) -> str:
    """One goal as a single line, whichever shape the profile wrote it in.

    The templates ask for ``grand_goals`` and all three answer with objects --
    ``{"goal": "...", "description": "..."}`` -- but an older capture may hold
    bare strings. Formatting a dict with ``str()`` produces the Python repr,
    braces and quotes included, which is what the forecast prompt would then
    read as the agent's desire.
    """
    if isinstance(item, dict):
        goal = _clean_text(item.get("goal") or item.get("name") or item.get("title"))
        detail = _clean_text(item.get("description") or item.get("detail"))
        if goal and detail and detail != goal:
            return f"{goal} — {detail}"
        return goal or detail
    return _clean_text(item)


def _desires_of(profile: dict[str, Any]) -> list[Desire]:
    """The profile's goals as the D in BDI.

    All three templates carry ``grand_goals``, and the unified view exposes it
    as ``goals``. Priority comes from list order — the templates ask for the
    goals in order — so the first goal outranks the last. Round two may replace
    the block; this is the opening set.
    """
    view = normalize_semantic_view(_semantic_of(profile))
    goals = view.get("goals")
    if not isinstance(goals, list):
        return []
    descriptions = [text for text in (_line_from(item) for item in goals) if text]
    total = len(descriptions)
    return [
        Desire(
            desire_id=f"goal_{index}",
            description=description,
            # Ordered list: the first is the strongest, the last the weakest.
            # Never zero — a goal with no priority reads as one the agent has
            # already given up on.
            priority=round(1.0 - (index - 1) / (total + 1), 3),
        )
        for index, description in enumerate(descriptions, 1)
    ]


def _constraints_of(profile: dict[str, Any]) -> list[str]:
    """Standing limits the profile states, as plain lines.

    The templates put these under ``concerns``: ``sensitive_points`` for the
    individual form, and the same key plus ``shared_grievances`` for the two
    entity forms. They are what the actor will not trade away, which is what
    the decision prompt means by a constraint.
    """
    view = normalize_semantic_view(_semantic_of(profile))
    concerns = view.get("concerns")
    if not isinstance(concerns, dict):
        return []
    lines: list[str] = []
    for key in ("sensitive_points", "shared_grievances"):
        values = concerns.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            # Same shape hazard as the goals: a bare line is what the templates
            # ask for, but an object would stringify into a Python repr and be
            # read as the constraint itself.
            text = _line_from(value)
            if text and text not in lines:
                lines.append(text)
    return lines


def _policy_from_style(semantic: dict[str, Any]) -> DecisionPolicy:
    """Compatibility priors; the evidence-linked behavioral model is primary."""
    view = normalize_semantic_view(semantic)
    preferences = view.get("preferences") or {}
    if not isinstance(preferences, dict):
        preferences = {}
    conflict = preferences.get("conflict_response")
    mode = ""
    if isinstance(conflict, dict):
        mode = _clean_text(conflict.get("mode")).casefold()

    risk_tolerance = 0.5
    # Initiative does not identify risk preference; retain a compatibility
    # prior and let the evidence-linked contextual model express uncertainty.

    compromise_tendency = 0.5
    # "escalate" is a substring of "deescalate", so testing it first would read
    # a de-escalating actor as an escalating one. The negative forms are checked
    # ahead of it for that reason; the priority order is the reverse of severity.
    if "deescalate" in mode or "de-escalate" in mode:
        compromise_tendency = 0.7
    elif "ignore" in mode:
        compromise_tendency = 0.6
    elif "reciprocate" in mode:
        compromise_tendency = 0.4
    elif "escalate" in mode:
        compromise_tendency = 0.2

    return DecisionPolicy(
        risk_tolerance=risk_tolerance,
        compromise_tendency=compromise_tendency,
        # Response speed has no counterpart in the templates: pace is about
        # whether the actor moves first, not how quickly it answers.
        response_speed=0.5,
    )


def to_agent_profiles(
    profiles: list[dict[str, Any]],
    *,
    event_id: str,
    event_title: str,
) -> list[AgentProfile]:
    """Convert the stage's layered profiles into forecast agents.

    Members are not agents -- they are carried on their principal's
    ``public_state["subordinates"]`` so the executor-annotation step can read
    them and the UI can render the tree. Only principals are returned.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        agent_id = _clean_text(profile.get("agent_id"))
        if agent_id:
            by_id[agent_id] = profile

    members_by_parent: dict[str, list[dict[str, Any]]] = {}
    for profile in by_id.values():
        if str(profile.get("agent_level") or "").lower() != "member":
            continue
        parent = _clean_text(profile.get("parent_agent_id"))
        if parent:
            members_by_parent.setdefault(parent, []).append(profile)

    agents: list[AgentProfile] = []
    for agent_id, profile in by_id.items():
        if str(profile.get("agent_level") or "").lower() == "member":
            continue
        name = _clean_text(profile.get("source_figure")) or agent_id
        state = _public_state(profile)
        members = members_by_parent.get(agent_id) or []
        if members:
            state["subordinates"] = [_subordinate(member) for member in members]
        # The prose role is the human-readable detail; ``role`` itself carries
        # the five-label stance the UI indexes by. Keeping the sentence in
        # ``role`` made every actor fall through to the neutral fallback.
        role_text = _role_of(profile)
        if role_text:
            state["role_detail"] = role_text
        agent = AgentProfile(
            agent_id=stable_agent_id(event_id, event_title, name),
            name=name,
            aliases=[_clean_text(alias) for alias in profile.get("aliases") or [] if _clean_text(alias)],
            agent_type=resolve_entity_type(profile | {"title": _title_of(profile), "description": _description_of(profile)})[0],
            decision_subject=name,
            role=_stance_of(profile, role_text),
            behavioral_model=grounded_behavior(profile.get("behavioral_model"),_timeline_refs(profile),role_text),
            title=_title_of(profile),
            affiliation=_affiliation_of(profile),
            description=_description_of(profile),
            public_state=state,
            relevant_timeline_refs=_timeline_refs(profile),
            historical_actions=_historical_actions(profile),
            beliefs=_beliefs_of(profile, name),
            desires=_desires_of(profile),
            constraints=_constraints_of(profile),
            capabilities=_capabilities_of(profile),
            decision_policy=_policy_from_style(_semantic_of(profile)),
        )
        if profile.get("decision_core"):
            from app.domain.decision_core import DecisionCore
            core = DecisionCore.model_validate(profile["decision_core"])
            agent.decision_core = core
            agent.title = _clean_text(profile.get("title")) or agent.title
            agent.affiliation = _clean_text(profile.get("affiliation")) or agent.affiliation
            from app.simulation.decision_profile import initial_beliefs, initial_plan
            agent.beliefs = initial_beliefs(core)
            agent.current_plan = initial_plan(core)
            agent.desires = [Desire(desire_id=g.id, description=g.description,
                priority={"high":1.0,"medium":.5,"low":.25}.get(g.priority)) for g in core.goals]
            agent.capabilities = []
            agent.constraints = []
            agent.behavioral_model = grounded_behavior({})
            agent.public_state = {k:v for k,v in state.items() if k in {"profile_kind", "knowledge_boundary", "subordinates"}}
        agents.append(agent)
    return agents


def _subordinate(member: dict[str, Any]) -> dict[str, Any]:
    """A member's executor card, the shape the annotation step reads."""
    return {
        "agent_id": _clean_text(member.get("agent_id")),
        "name": _clean_text(member.get("source_figure")),
        "entity_type": _clean_text(member.get("entity_type")),
        "role_in_entity": _clean_text(member.get("role_in_entity")),
        "title": _clean_text(member.get("title")),
        "affiliation": _clean_text(member.get("affiliation")),
        "capability": _clean_text(member.get("capability")),
        "style": _clean_text(member.get("style")),
    }
