"""Semantic view: one concept-keyed projection of a stored semantic block."""

from __future__ import annotations

from typing import Any


def _semantic_block(profile: dict) -> dict:
    """The profile's distilled-knowledge block (form-agnostic)."""
    return profile.get("semantic") if isinstance(profile.get("semantic"), dict) else {}


_CONCEPT_KEYS = (
    "identity",      # identity                   — who this actor is
    "concerns",      # concerns                   — priorities / red lines
    "views",         # views                      — stance / positions
    "experience",    # experience                 — career / evolution
    "style",         # style                      — personality / behavior prefs
    "strength",      # strength | power_base | capabilities — the scored power block
    "goals",         # grand_goals (shared name)
)


def _dict_of(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _list_of(value: object) -> list:
    return value if isinstance(value, list) else []


def normalize_semantic_view(semantic: dict) -> dict:
    """Normalize a stored semantic block into one concept view."""
    out: dict[str, Any] = {}
    if not isinstance(semantic, dict):
        return out
    identity = semantic.get("identity")
    if isinstance(identity, dict):
        out["identity"] = identity
    concerns = semantic.get("concerns")
    if isinstance(concerns, dict):
        out["concerns"] = concerns
    views = semantic.get("views")
    if isinstance(views, dict):
        out["views"] = views
    experience = semantic.get("experience")
    if isinstance(experience, dict):
        out["experience"] = experience
    # New templates lift the two style sub-blocks to the top level (preferences /
    # communication) for the individual form, and nest them inside style for the
    # institution / collective forms.
    style = semantic.get("style")
    prefs = (
        semantic.get("preferences")
        or _dict_of(semantic.get("style")).get("behavior_preferences")
    )
    comm = (
        semantic.get("communication")
        or _dict_of(semantic.get("style")).get("communication_style")
    )
    if isinstance(style, dict):
        # Keep the sub-blocks merged into the style card so a reader that
        # walks style.means_preference (the BDI prompt names) still finds them.
        merged_style = dict(style)
        if isinstance(prefs, dict):
            merged_style.setdefault("means_preference", prefs)
            merged_style.setdefault("behavior_preferences", prefs)
        if isinstance(comm, dict):
            merged_style.setdefault("language_style", comm)
            merged_style.setdefault("communication_style", comm)
        out["style"] = merged_style
    if isinstance(prefs, dict):
        out["preferences"] = prefs
    if isinstance(comm, dict):
        out["communication"] = comm
    strength = semantic.get("strength") or semantic.get("power_base") or semantic.get("capabilities")
    if isinstance(strength, dict):
        out["strength"] = strength
    # Goals: shared name.
    goals = semantic.get("grand_goals")
    if isinstance(goals, list):
        out["goals"] = goals
    return out
