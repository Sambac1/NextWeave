"""Helper functions for the layered-figure step body.

Extracted from the original ``figures_pipeline_service``; only the functions the
Step 4a-6 body transitively calls are included.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.layers.support.layer_rules import (
    CATEGORY_REPRESENTATIVE_LIMITS,
    COMPACTABLE_CATEGORIES,
    evidence_threshold_for,
    infer_entity_type as _infer_entity_type,
    is_leader,
)
from app.layers.support.llm_normalizers import (
    _clean_text,
    _dedupe_texts,
    _normalize_stance,
    _stance_rank,
)

def _is_placeholder_analysis(text: object) -> bool:
    normalized = _clean_text(text).lower()
    return not normalized or normalized == "(analysis unavailable)"


def _collect_unique_event_ids(
    occurrences: list[dict],
    actions_initiated: list[dict],
    actions_received: list[dict],
) -> list[str]:
    """Extract all unique, non-empty event_ids from history, evidence, and actions.

    Returns a deduplicated list preserving insertion order so downstream consumers
    have a single event-id index for this figure, without scanning nested arrays.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in occurrences:
        eid = str(item.get("event_id", "")).strip()
        if eid and eid not in seen:
            seen.add(eid)
            result.append(eid)
    for item in actions_initiated:
        eid = str(item.get("event_id", "")).strip()
        if eid and eid not in seen:
            seen.add(eid)
            result.append(eid)
    for item in actions_received:
        eid = str(item.get("event_id", "")).strip()
        if eid and eid not in seen:
            seen.add(eid)
            result.append(eid)
    return result


def _latest_occurrence_description(occurrences: list[dict]) -> str:
    for entry in reversed(occurrences):
        if not isinstance(entry, dict):
            continue
        description = _clean_text(entry.get("description"))
        if description:
            return description
    return ""


def _summarize_action_for_actor(action: dict, focal_actor: str = "") -> str:
    source = _clean_text(action.get("source") or action.get("actor"))
    target = _clean_text(action.get("target") or action.get("object"))
    action_text = _clean_text(action.get("action") or action.get("text") or action.get("description"))
    result = _clean_text(action.get("result"))
    focal = _clean_text(focal_actor)

    if focal and source.lower() == focal.lower():
        if action_text and target:
            summary = f"{action_text} {target}"
        else:
            summary = action_text or target
    elif focal and target.lower() == focal.lower():
        if source and action_text:
            summary = f"received {action_text} from {source}"
        else:
            summary = action_text or source
    else:
        if source and target and action_text:
            summary = f"{source} {action_text} {target}"
        elif source and action_text:
            summary = f"{source} {action_text}"
        elif action_text and target:
            summary = f"{action_text} {target}"
        else:
            summary = action_text or source or target

    if result:
        summary = f"{summary}, resulting in {result}" if summary else result
    return _clean_text(summary)


def _build_profile_summary(
    canonical: str,
    entity_type: str,
    role: str,
    title: str,
    affiliation: str,
    occurrences: list[dict],
    actions_initiated: list[dict],
    actions_received: list[dict],
) -> str:
    latest_description = _latest_occurrence_description(occurrences)
    representative_action = ""
    if actions_initiated:
        representative_action = _summarize_action_for_actor(actions_initiated[0], canonical)
    elif actions_received:
        representative_action = _summarize_action_for_actor(actions_received[0], canonical)

    identity_bits = _dedupe_texts([role, title, affiliation], limit=2)
    if identity_bits:
        opening = f"{canonical} is {identity_bits[0]}"
        if len(identity_bits) > 1:
            opening = f"{opening} associated with {identity_bits[1]}"
    elif entity_type:
        opening = f"{canonical} is a {entity_type} actor in this event"
    else:
        opening = f"{canonical} is an active actor in this event"

    detail_bits: list[str] = []
    if representative_action:
        detail_bits.append(f"A representative interaction is {representative_action}")
    if latest_description:
        detail_bits.append(f"Timeline context: {latest_description}")
    if not detail_bits and occurrences:
        detail_bits.append(f"It appears in {len(occurrences)} timeline entries")

    return _clean_text(". ".join([opening] + detail_bits) + ".")


def _build_profile_local_view(
    canonical: str,
    role: str,
    parent_actor: str,
    occurrences: list[dict],
    actions_initiated: list[dict],
    actions_received: list[dict],
) -> str:
    counterparts = _dedupe_texts(
        [
            action.get("target")
            for action in actions_initiated
            if isinstance(action, dict)
        ] + [
            action.get("source")
            for action in actions_received
            if isinstance(action, dict)
        ],
        limit=3,
    )
    latest_description = _latest_occurrence_description(occurrences)

    parts: list[str] = []
    if role:
        parts.append(f"In this event, {canonical} mainly appears as {role}")
    else:
        parts.append(f"In this event, {canonical} is a recurring actor")
    if parent_actor and parent_actor.lower() != canonical.lower():
        parts.append(f"and is aligned with {parent_actor}")

    sentence = _clean_text(" ".join(parts)) + "."
    if counterparts:
        sentence = f"{sentence} Its main counterparts are {', '.join(counterparts)}."
    if latest_description:
        sentence = f"{sentence} Recent timeline context: {latest_description}."
    return _clean_text(sentence)


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _actor_id(name: str) -> str:
    """A stable slug for an actor's name."""
    slug = re.sub(r"[^\w]+", "_", (name or "").lower(), flags=re.UNICODE).strip("_")
    if slug:
        return slug
    # No word characters at all (punctuation-only, emoji): fall back to a
    # content hash so distinct names still get distinct ids.
    if name:
        return f"actor_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:10]}"
    return "actor"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


_COERCIVE_ROOTS = {"09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"}
_COOPERATION_ROOTS = {"03", "04", "05", "06", "07", "08"}
_MEDIATOR_ROLE_KEYWORDS = ("mediator", "mediate", "broker", "facilitator", "facilitate", "arbitrat", "intermediar")


def _root_code_of(action: dict) -> str:
    root = _clean_text(action.get("cameo_root_code"))
    if root:
        return root.zfill(2)[:2] if root.isdigit() else root
    code = _clean_text(action.get("cameo_code"))
    if code and code[:2].isdigit():
        return code[:2]
    return ""


def _infer_stance_from_context(
    canonical: str,
    role_text: str,
    initiated: list[dict],
    received: list[dict],
) -> str:
    role_lower = (role_text or "").lower()
    if any(kw in role_lower for kw in _MEDIATOR_ROLE_KEYWORDS):
        return "mediator"
    initiated_coercive = sum(1 for a in initiated if _root_code_of(a) in _COERCIVE_ROOTS)
    received_coercive = sum(1 for a in received if _root_code_of(a) in _COERCIVE_ROOTS)
    if initiated_coercive or received_coercive:
        # An actor that both applies and absorbs pressure is classified by
        # whichever it does more of. Returning "initiator" on a single initiated
        # action would label every retaliating party an initiator too.
        return "initiator" if initiated_coercive >= received_coercive else "opponent"
    initiated_roots = {_root_code_of(a) for a in initiated}
    received_roots = {_root_code_of(a) for a in received}
    if (initiated_roots | received_roots) & _COOPERATION_ROOTS:
        return "supporter"
    return "neutral"


def _fill_missing_stances(
    figures: list[dict],
    actions_initiated: dict,
    actions_received: dict,
) -> None:
    for fig in figures:
        if _normalize_stance(fig.get("stance")):
            continue
        canonical = _clean_text(fig.get("canonical_name") or fig.get("figure") or fig.get("name"))
        fig["stance"] = _infer_stance_from_context(
            canonical,
            _clean_text(fig.get("role_in_event")),
            actions_initiated.get(canonical, []),
            actions_received.get(canonical, []),
        )


def _parse_dt(timestamp: str) -> datetime | None:
    ts = str(timestamp or "")
    for fmt, length in (("%Y-%m-%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        try:
            return datetime.strptime(ts[:length], fmt)
        except ValueError:
            continue
    return None


def _resolve_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """Resolve alias chains to terminal names without mutating the input mapping."""
    resolved: dict[str, str] = {}
    for source, target in mapping.items():
        seen: set[str] = set()
        current = target
        while current in mapping and current not in seen:
            seen.add(current)
            current = mapping[current]
        resolved[source] = current
    return resolved


def _canonical_from(raw: dict, name_mapping: dict) -> str:
    """Follow an alias chain to its canonical actor name."""
    name = _clean_text(raw.get("canonical_name") or raw.get("figure") or raw.get("name"))
    seen: set[str] = set()
    while name in name_mapping and name not in seen:
        seen.add(name)
        name = _clean_text(name_mapping[name])
    return name


def _pick_dominant_stance(tally: dict[str, int]) -> str:
    """Pick the stance an actor holds across the whole event.

    Counts occurrences instead of keeping the most confrontational label seen.
    ``initiator`` and ``opponent`` share a rank, so a "keep the higher rank"
    rule cannot separate them and ends up resolving by iteration order — the
    same actor then gets a different stance from one run to the next. Falls back
    to rank only when the top count is shared, and returns "" when even that
    ties, leaving the decision to _infer_stance_from_context and its action
    evidence.
    """
    if not tally:
        return ""
    top_count = max(tally.values())
    leaders = [stance for stance, count in tally.items() if count == top_count]
    if len(leaders) == 1:
        return leaders[0]
    leaders.sort(key=_stance_rank, reverse=True)
    if _stance_rank(leaders[0]) > _stance_rank(leaders[1]):
        return leaders[0]
    return ""


def _merge_canonical_figures(figures: list[dict], name_mapping: dict) -> list[dict]:
    merged: dict[str, dict] = {}
    stance_tally: dict[str, dict[str, int]] = {}
    for fig in figures:
        if not isinstance(fig, dict):
            continue
        canonical = _canonical_from(fig, name_mapping)
        if not canonical:
            continue
        record = merged.setdefault(
            canonical,
            {
                "id": _actor_id(canonical),
                "canonical_name": canonical,
                "figure": canonical,
                "entity_type": _infer_entity_type(
                    canonical,
                    fig.get("entity_type", ""),
                    fig.get("title", ""),
                ),
                "aliases": [],
                "title": "",
                "affiliation": "",
                "role_in_event": "",
                "stance": "",
                "confidence": 0.0,
            },
        )
        raw_name = _clean_text(fig.get("name") or fig.get("figure"))
        if raw_name and raw_name != canonical and raw_name not in record["aliases"]:
            record["aliases"].append(raw_name)
        for field in ("title", "affiliation"):
            value = _clean_text(fig.get(field))
            if value and not record[field]:
                record[field] = value
        role = _clean_text(fig.get("role_in_event") or fig.get("role"))
        if role and not record["role_in_event"]:
            record["role_in_event"] = role
        stance = _normalize_stance(fig.get("stance"))
        if stance:
            counts = stance_tally.setdefault(canonical, {})
            counts[stance] = counts.get(stance, 0) + 1
        record["confidence"] = max(record["confidence"], _safe_float(fig.get("confidence"), 0.7))
        record["entity_type"] = _infer_entity_type(
            canonical,
            record.get("entity_type", ""),
            record.get("title", ""),
        )
    for canonical, record in merged.items():
        record["stance"] = _pick_dominant_stance(stance_tally.get(canonical, {}))
    return list(merged.values())


def _build_figure_context(timeline: list[dict], name_mapping: dict) -> tuple[dict, dict, dict]:
    occurrences: dict[str, list[dict]] = {}
    actions_initiated: dict[str, list[dict]] = {}
    actions_received: dict[str, list[dict]] = {}

    for entry in timeline:
        ts = entry.get("timestamp", "")
        desc = entry.get("description", "")
        timeline_id = entry.get("timeline_id", "")
        event_id = entry.get("event_id", "")
        source_sections = entry.get("source_sections", [])

        normalized_figures = []
        for fig in entry.get("figures", []):
            if not isinstance(fig, dict):
                continue
            canonical = _canonical_from(fig, name_mapping)
            if not canonical:
                continue
            normalized = {
                **fig,
                "id": _actor_id(canonical),
                "name": canonical,
                "canonical_name": canonical,
                "entity_type": _infer_entity_type(canonical, fig.get("entity_type", ""), fig.get("title", "")),
            }
            normalized_figures.append(normalized)
            occurrences.setdefault(canonical, []).append(
                {
                    "timeline_id": timeline_id,
                    "event_id": event_id,
                    "timestamp": ts,
                    "description": desc,
                    "source_sections": source_sections,
                    "role": _clean_text(fig.get("role")),
                }
            )
        entry["figures"] = normalized_figures

        normalized_actions = []
        for act in entry.get("actions", []):
            if not isinstance(act, dict):
                continue
            src = _clean_text(name_mapping.get(act.get("source", ""), act.get("source", "")))
            tgt = _clean_text(name_mapping.get(act.get("target", ""), act.get("target", "")))
            if not src:
                continue
            normalized = {
                **act,
                "source": src,
                "target": tgt,
                "timeline_id": timeline_id,
                "event_id": event_id,
                "timestamp": ts,
                "source_sections": source_sections,
            }
            normalized_actions.append(normalized)
            actions_initiated.setdefault(src, []).append(normalized)
            if tgt:
                actions_received.setdefault(tgt, []).append(normalized)
        entry["actions"] = normalized_actions

    return occurrences, actions_initiated, actions_received


def _is_high_impact_action(action: dict) -> bool:
    text = " ".join(
        _clean_text(action.get(key, ""))
        for key in ("action", "action_label", "cameo_code", "cameo_root_code", "result")
    ).lower()
    keywords = (
        "tariff", "sanction", "restrict", "ban", "export control", "duty",
        "truce", "pause", "negot", "talk", "agreement", "lawsuit",
        "injunction", "court", "ruling", "strike down",
    )
    return any(keyword in text for keyword in keywords)


def _event_title_tokens(event_title: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", _clean_text(event_title).lower())
    stopwords = {"and", "the", "of", "in", "on", "war", "conflict", "dispute", "relations", "relation"}
    return [token for token in normalized.split() if len(token) > 2 and token not in stopwords]


_COUNTRY_TITLE_VARIANTS = {
    "russia": ("russia", "russian", "russo"),
    "ukraine": ("ukraine", "ukrainian", "ukrain"),
    "china": ("china", "chinese"),
    "united states": ("united states", "u.s.", "us"),
    "japan": ("japan", "japanese"),
    "germany": ("germany", "german"),
    "france": ("france", "french"),
    "korea": ("korea", "korean"),
    "israel": ("israel", "israeli"),
    "palestine": ("palestine", "palestinian"),
    "iran": ("iran", "iranian"),
    "taiwan": ("taiwan", "taiwanese"),
}


def _mentions_event_title_actor(name: str, event_title: str) -> bool:
    name_l = _clean_text(name).lower()
    if not name_l:
        return False
    title_l = _clean_text(event_title).lower()
    if name_l in title_l:
        return True
    for country, variants in _COUNTRY_TITLE_VARIANTS.items():
        if name_l == country or any(name_l == v for v in variants):
            if any(v in title_l for v in variants):
                return True
    tokens = _event_title_tokens(event_title)
    return bool(tokens) and all(token in title_l for token in re.sub(r"[^a-z0-9]+", " ", name_l).split() if token)


def _is_core_party(
    canonical_name: str,
    entity_type: str,
    role: str,
    affiliation: str,
    title: str,
    actions_initiated: list[dict],
    actions_received: list[dict],
    event_title: str,
) -> bool:
    """Keep actors supported as core parties by the question, timeline, or explicit event role."""
    if _mentions_event_title_actor(canonical_name, event_title):
        return True
    interaction = len(actions_initiated or []) + len(actions_received or [])
    if interaction >= 10:
        return True
    text = " ".join(
        _clean_text(v)
        for v in (canonical_name, role, affiliation, title)
        if v
    ).lower()
    if any(
        term in text
        for term in (
            "initiator", "opponent", "adversary", "government",
            "president", "minister", "prime minister", "chancellor",
            "lead", "leader", "head of state", "defense", "military",
        )
    ):
        return True
    return False


def _infer_parent_actor(name: str, entity_type: str, affiliation: str = "", title: str = "") -> tuple[str, str]:
    text = " ".join((_clean_text(name), _clean_text(affiliation), _clean_text(title))).lower()
    mappings = (
        (("united states", "u.s.", "us", "white house", "u.s. trade representative", "department of commerce", "treasury", "supreme court", "pentagon", "state department", "trump administration", "biden administration", "obama administration"), "United States", "country"),
        (("china", "beijing", "chinese", "state council", "ministry of commerce of china", "people's bank of china", "supreme people's court"), "China", "country"),
        (("european union", "e.u.", "eu", "european commission", "european council", "european parliament"), "European Union", "organization"),
        (("united kingdom", "u.k.", "uk", "british", "downing street", "bank of england"), "United Kingdom", "country"),
        (("japan", "japanese", "tokyo", "ministry of economy, trade and industry"), "Japan", "country"),
    )
    for keywords, parent, parent_type in mappings:
        for keyword in keywords:
            if keyword in ("us", "eu", "uk"):
                if re.search(rf"(?<![a-z0-9]){keyword}(?![a-z0-9])", text):
                    return parent, parent_type
            elif re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text):
                return parent, parent_type
    if entity_type in {"country", "government"}:
        return _clean_text(name), "country"
    return "", ""


def _infer_aggregation_metadata(
    canonical_name: str,
    entity_type: str,
    role: str,
    affiliation: str,
    title: str,
    actions_initiated: list[dict],
    actions_received: list[dict],
    event_title: str = "",
) -> tuple[str, str, str, str]:
    parent_actor, parent_actor_type = _infer_parent_actor(canonical_name, entity_type, affiliation, title)
    if _is_core_party(
        canonical_name,
        entity_type,
        role,
        affiliation,
        title,
        actions_initiated,
        actions_received,
        event_title,
    ):
        return parent_actor, parent_actor_type, "", ""
    text = " ".join(
        [
            _clean_text(canonical_name),
            _clean_text(role),
            _clean_text(affiliation),
            _clean_text(title),
            " ".join(_clean_text(a.get("action", "")) for a in actions_initiated + actions_received),
            " ".join(_clean_text(a.get("action_label", "")) for a in actions_initiated + actions_received),
        ]
    ).lower()

    if parent_actor and _clean_text(parent_actor).lower() != _clean_text(canonical_name).lower() and entity_type in {"organization", "government", "person"}:
        return parent_actor, parent_actor_type, "parent_state_subunit", "Sub-unit of a main state or bloc actor with a similar external role in this event"

    if entity_type == "country":
        if event_title and _mentions_event_title_actor(canonical_name, event_title):
            return parent_actor, parent_actor_type, "", ""
        if any(term in text for term in ("supply chain", "supply-chain", "export", "manufacturing", "factory", "reroute", "diversion", "assembly", "shipment", "shipments", "transshipment", "transshipments", "relocate", "relocation", "rerouting")):
            return parent_actor, parent_actor_type, "supply_chain_third_party_country", "Third-party country mainly acting as a similar supply-chain or rerouting actor"
        if any(term in text for term in ("ally", "partner", "support", "backed", "joined", "followed", "cooperation", "trilateral", "bilateral", "multilateral", "dialogue", "talks", "coordinate", "coordination", "align", "alignment", "fta", "diplomatic")):
            return parent_actor, parent_actor_type, "supporting_third_party_country", "Third-party country mainly acting as a similar supporting or aligned actor"
        if any(term in text for term in ("affected", "hit by", "impacted", "caught in", "retaliation", "retaliate", "retaliatory", "target", "targeted", "sanctioned", "penalized")):
            return parent_actor, parent_actor_type, "affected_third_party_country", "Third-party country mainly acting as a similar affected actor"

    if entity_type in {"organization", "government"}:
        if any(term in text for term in ("court", "tribunal", "lawsuit", "injunction", "commission", "regulator", "ministry", "department", "office")):
            return parent_actor, parent_actor_type, "regulatory_or_legal_body", "Organization or agency mainly acting through legal or regulatory functions"
        if any(term in text for term in ("association", "group", "industry", "business", "chamber", "council")):
            return parent_actor, parent_actor_type, "background_organization", "Organization mainly acting as a supporting or background institutional actor"

    return parent_actor, parent_actor_type, "", ""


def _should_compact_actor(fig: dict) -> bool:
    return fig.get("aggregation_category") in set(COMPACTABLE_CATEGORIES)


def _compact_key_figure_names(
    selected_names: list[str],
    ranked_figures: list[dict],
    max_figures: int,
) -> list[str]:
    by_name = {f.get("canonical_name", ""): f for f in ranked_figures}
    ordered_names: list[str] = []
    seen_names = set()
    seen_buckets: dict[tuple, str] = {}
    category_reps: dict[str, str] = {}
    category_counts: dict[str, int] = {}
    category_limits = dict(CATEGORY_REPRESENTATIVE_LIMITS)
    if selected_names:
        names = selected_names
    else:
        names = [f.get("canonical_name", "") for f in ranked_figures]
    for name in names:
        if not name or name not in by_name or name in seen_names:
            continue
        fig = by_name[name]
        category = fig.get("aggregation_category") or ""
        if _should_compact_actor(fig):
            bucket = (fig.get("parent_actor") or name, category)
            limit = category_limits.get(category)
            # A representative for this bucket already stands → this actor is
            # folded into it. Same for a category whose quota is full: the
            # member is recorded under the category's first representative.
            rep_name = seen_buckets.get(bucket)
            if rep_name is None and limit is not None and category_counts.get(category, 0) >= limit:
                rep_name = category_reps.get(category)
            if rep_name:
                rep_fig = by_name.get(rep_name)
                from app.knowledge.actor_identity import delegated_seat_supported
                proof = {'contained': name, 'container': rep_name,
                         'decision_class': fig.get('decision_class'),
                         'delegation_quote': fig.get('delegation_quote')}
                if rep_fig is not None and delegated_seat_supported(proof, [fig, rep_fig]):
                    rep_fig.setdefault("aggregated_members", []).append(name)
                    continue
            seen_buckets[bucket] = name
            category_reps.setdefault(category, name)
            if limit is not None:
                category_counts[category] = category_counts.get(category, 0) + 1
        seen_names.add(name)
        ordered_names.append(name)
        if len(ordered_names) >= max_figures:
            break
    return ordered_names[:max_figures]


def _principal_pick_rank(principal: str, by_name: dict) -> int:
    """Rank specific decision units before broad country containers."""
    entity_type = _clean_text((by_name.get(principal) or {}).get("entity_type")).lower()
    if entity_type in {"organization", "government", "person"}:
        return 2
    if entity_type == "country":
        return 1
    return 0


def _dominant_role(occurrences: list[dict], fallback: str = "") -> str:
    """The role text that best describes this actor ACROSS the event."""
    counts: dict[str, int] = {}
    latest: dict[str, str] = {}
    for occurrence in occurrences:
        if not isinstance(occurrence, dict):
            continue
        role = _clean_text(occurrence.get("role"))
        if not role:
            continue
        counts[role] = counts.get(role, 0) + 1
        ts = _clean_text(occurrence.get("timestamp"))
        if role not in latest or ts > latest[role]:
            latest[role] = ts
    if not counts:
        return fallback
    return max(counts, key=lambda r: (counts[r], latest.get(r, "")))


def _score_figures(
    canonical_figures: list[dict],
    occurrences: dict[str, list[dict]],
    actions_initiated: dict[str, list[dict]],
    actions_received: dict[str, list[dict]],
    event_title: str = "",
) -> list[dict]:
    figures_by_name = {f["canonical_name"]: dict(f) for f in canonical_figures}
    for name in set(occurrences) | set(actions_initiated) | set(actions_received):
        figures_by_name.setdefault(
            name,
            {
                "id": _actor_id(name),
                "canonical_name": name,
                "figure": name,
                "entity_type": _infer_entity_type(name),
                "aliases": [],
                "title": "",
                "affiliation": "",
                "role_in_event": "",
                "stance": "",
                "confidence": 0.7,
            },
        )

    # ``or [1]`` only covers an empty dict. A dict whose values are all empty
    # lists yields [0, 0, ...], which is truthy, so max() returned 0 and the
    # divisions below raised ZeroDivisionError. Trailing ``or 1`` covers that.
    max_occ = max((len(v) for v in occurrences.values()), default=1) or 1
    max_init = max((len(v) for v in actions_initiated.values()), default=1) or 1
    max_recv = max((len(v) for v in actions_received.values()), default=1) or 1
    all_dates = [
        _parse_dt(item.get("timestamp", ""))
        for events in occurrences.values()
        for item in events
    ]
    all_dates = [d for d in all_dates if d is not None]
    recent_cutoff = (max(all_dates) - timedelta(days=365)) if all_dates else None

    scored = []
    for name, fig in figures_by_name.items():
        occ = occurrences.get(name, [])
        init = actions_initiated.get(name, [])
        recv = actions_received.get(name, [])
        occ_score = len(occ) / max_occ
        init_score = len(init) / max_init
        recv_score = len(recv) / max_recv
        recent_score = 0.0
        if recent_cutoff is not None:
            recent_score = 1.0 if any((_parse_dt(e.get("timestamp", "")) or datetime.min) >= recent_cutoff for e in occ) else 0.0
        authority_score = 1.0 if (
            fig.get("entity_type") in {"person", "country", "government"}
            or any(term in (fig.get("title", "") or "").lower() for term in ("president", "minister", "secretary", "government"))
        ) else 0.4
        importance = (
            0.35 * occ_score
            + 0.30 * init_score
            + 0.10 * recv_score
            + 0.15 * recent_score
            + 0.10 * authority_score
        )
        importance_score = round(max(0.0, min(1.0, importance)), 3)

        fig["role_in_event"] = _dominant_role(occ, fallback=fig.get("role_in_event", ""))

        role_text = " ".join(
            _clean_text(fig.get(key, ""))
            for key in ("role_in_event", "role", "title", "affiliation")
        ).lower()
        # aggregation_reason (4th return value) is human-readable prose meant for
        # debugging: it is deliberately dropped here — no downstream consumer.
        # Kept in the function's signature so a debugger can print it.
        parent_actor, parent_actor_type, aggregation_category, _aggregation_reason = _infer_aggregation_metadata(
            name,
            fig.get("entity_type", ""),
            fig.get("role_in_event", "") or fig.get("role", ""),
            fig.get("affiliation", ""),
            fig.get("title", ""),
            init,
            recv,
            event_title,
        )
        main_party_bonus = 0.15 if _mentions_event_title_actor(name, event_title) else 0.0
        high_impact_bonus = 0.12 if any(_is_high_impact_action(action) for action in init + recv) else 0.0
        pivot_bonus = 0.08 if any(term in role_text for term in ("initiator", "opponent", "supporter", "mediator", "negotiator", "government", "court")) else 0.0
        tail_penalty = 0.12 if len(occ) <= 1 and (len(init) + len(recv)) <= 1 and main_party_bonus == 0.0 and high_impact_bonus == 0.0 else 0.0
        core_score = max(0.0, min(1.0, importance_score + main_party_bonus + high_impact_bonus + pivot_bonus - tail_penalty))

        fig["importance_score"] = importance_score
        fig["core_actor_score"] = round(core_score, 3)
        fig["parent_actor"] = parent_actor
        fig["parent_actor_type"] = parent_actor_type
        fig["aggregation_category"] = aggregation_category
        fig["evidence_count"] = len(occ)
        fig["actions_initiated_count"] = len(init)
        fig["actions_received_count"] = len(recv)
        all_ts = [
            _clean_text(e.get("timestamp"))
            for e in occ + init + recv
            if _clean_text(e.get("timestamp"))
        ]
        fig["latest_action_ts"] = max(all_ts) if all_ts else ""
        scored.append(fig)

    return sorted(
        scored,
        key=lambda f: (f.get("core_actor_score", 0.0), f.get("importance_score", 0.0)),
        reverse=True,
    )


def _filter_by_min_evidence(
    scored_figures: list[dict],
    min_evidence_ratio: float,
    min_evidence_count: int,
    total_entries: int,
    min_keep_figures: int,
) -> tuple[list[dict], list[dict], int]:
    """Hard-filter figures that appear in too small a share of the timeline."""
    if not scored_figures:
        return [], [], 0

    thresholds = [
        evidence_threshold_for(f, min_evidence_ratio, total_entries, min_evidence_count)
        for f in scored_figures
    ]

    relax = 0
    passing = scored_figures
    for relax in range(max(thresholds)):
        passing = [
            f
            for f, threshold in zip(scored_figures, thresholds)
            if is_leader(f) or f.get("source_authority_evidence") or f.get('decision_seat')
            or f.get("evidence_count", 0) >= max(1, threshold - relax)
        ]
        if len(passing) >= min_keep_figures:
            break

    passing_ids = {id(f) for f in passing}
    filtered_out = [f for f in scored_figures if id(f) not in passing_ids]
    return passing, filtered_out, relax


def _compose_figure_profile(
    base: dict,
    detail: dict,
    occurrences: list[dict],
    actions_initiated: list[dict],
    actions_received: list[dict],
) -> dict:
    canonical = base.get("canonical_name") or detail.get("figure") or detail.get("canonical_name") or "Unknown"
    history = occurrences
    raw_local_view = _clean_text(detail.get("local_view"))
    raw_summary = _clean_text(detail.get("summary"))
    role = _clean_text(detail.get("role_in_event")) or base.get("role_in_event", "")
    if _is_placeholder_analysis(raw_summary):
        summary = _build_profile_summary(
            canonical,
            base.get("entity_type", ""),
            role,
            base.get("title", ""),
            base.get("affiliation", ""),
            occurrences,
            actions_initiated,
            actions_received,
        )
    else:
        summary = raw_summary
    if _is_placeholder_analysis(raw_local_view):
        local_view = _build_profile_local_view(
            canonical,
            role,
            base.get("parent_actor", ""),
            occurrences,
            actions_initiated,
            actions_received,
        )
    else:
        local_view = raw_local_view
    profile = {
        **base,
        "id": base.get("id") or _actor_id(canonical),
        "figure": canonical,
        "canonical_name": canonical,
        "aliases": [
            alias for alias in (base.get("aliases") or [])
            if _clean_text(alias) and _clean_text(alias) != canonical
        ],
        "entity_type": base.get("entity_type") or _infer_entity_type(canonical),
        "parent_actor": base.get("parent_actor", ""),
        "parent_actor_type": base.get("parent_actor_type", ""),
        "aggregation_category": base.get("aggregation_category", ""),
        "aggregated_members": [m for m in base.get("aggregated_members", []) if m != canonical],
        "role_in_event": role,
        "stance": _normalize_stance(detail.get("stance")) or _normalize_stance(base.get("stance")) or "neutral",
        "summary": summary,
        "history": history,
        "local_view": local_view,
        "actions_initiated": actions_initiated,
        "actions_received": actions_received,
        "related_event_ids": _collect_unique_event_ids(occurrences, actions_initiated, actions_received),
        "confidence": max(_safe_float(base.get("confidence"), 0.7), _safe_float(detail.get("confidence"), 0.0)),
    }
    return profile


def _unique_texts(value: object) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value in (None, ""):
        items = []
    else:
        items = [value]
    result: list[str] = []
    for item in items:
        text = _clean_text(item)
        if text and text not in result:
            result.append(text)
    return result
