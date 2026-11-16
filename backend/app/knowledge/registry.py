"""Original decision-center normalization for the bounded ordinary-Agent registry."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


REGISTRY_STATUSES = {
    "same_entity", "represented_by", "independent_agent",
    "historical_actor", "context_entity", "uncertain",
}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _name_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _candidate_words(candidate: dict) -> list[str]:
    name = unicodedata.normalize(
        "NFKC", _clean(candidate.get("canonical_name") or candidate.get("name")),
    ).casefold()
    return re.findall(r"[^\W_]+", name, flags=re.UNICODE)


def _administration_stem(candidate: dict) -> str:
    normalized_name = _name_key(candidate.get("canonical_name") or candidate.get("name"))
    for suffix in ("administration", "government", "cabinet", "presidency", "executive"):
        suffix_key = _name_key(suffix)
        if normalized_name.endswith(suffix_key) and len(normalized_name) > len(suffix_key):
            return normalized_name[:-len(suffix_key)]
    words = _candidate_words(candidate)
    suffixes = {"administration", "government", "cabinet", "presidency", "executive"}
    while words and words[-1] in suffixes:
        words.pop()
    return words[-1] if words else ""


def _person_surname(candidate: dict) -> str:
    words = _candidate_words(candidate)
    titles = {"president", "prime", "minister", "chairman", "chancellor", "secretary"}
    words = [word for word in words if word not in titles]
    return words[-1] if words else ""


def _country_key(value: object) -> str:
    key = _name_key(value)
    return {
        "us": "unitedstates",
        "usa": "unitedstates",
        "unitedstatesofamerica": "unitedstates",
        "prc": "china",
        "peoplesrepublicofchina": "china",
        "chinese": "china",
    }.get(key, key)


def _candidate_country(candidate: dict) -> str:
    entity_type = _clean(candidate.get("agent_type") or candidate.get("entity_type")).lower()
    if entity_type == "country":
        return _country_key(candidate.get("canonical_name"))
    affiliation = _country_key(candidate.get("affiliation"))
    if affiliation.startswith(("unitedstates", "usgovernment")):
        return "unitedstates"
    if affiliation.startswith(("china", "chinese")):
        return "china"
    name_key = _name_key(candidate.get("canonical_name"))
    if name_key.startswith(("us", "unitedstates")):
        return "unitedstates"
    if name_key.startswith(("china", "chinese")):
        return "china"
    return ""


def _is_independent_institution(candidate: dict) -> bool:
    name_key = _name_key(candidate.get("canonical_name"))
    return any(term in name_key for term in (
        "court", "supremecourt", "congress", "parliament", "legislature",
        "federalreserve", "centralbank", "worldtradeorganization", "wto",
    ))


def _same_decision_center(left: dict, right: dict) -> bool:
    left_key = _name_key(left.get("canonical_name"))
    right_key = _name_key(right.get("canonical_name"))
    if left_key == right_key:
        return True
    left_type = _clean(left.get("agent_type") or left.get("entity_type")).lower()
    right_type = _clean(right.get("agent_type") or right.get("entity_type")).lower()
    types = {left_type, right_type}
    left_aliases = {_name_key(item) for item in left.get("aliases", []) if _name_key(item)}
    right_aliases = {_name_key(item) for item in right.get("aliases", []) if _name_key(item)}
    if left_key in right_aliases or right_key in left_aliases or left_aliases.intersection(right_aliases):
        return True
    if _is_independent_institution(left) or _is_independent_institution(right):
        return False

    if types == {"person"}:
        left_words, right_words = _candidate_words(left), _candidate_words(right)
        left_surname, right_surname = _person_surname(left), _person_surname(right)
        titled = {"president", "chairman", "prime", "minister", "secretary"}
        return bool(
            left_surname and left_surname == right_surname and len(left_surname) >= 4
            and (len(left_words) == 1 or len(right_words) == 1
                 or bool(titled.intersection(left_words)) or bool(titled.intersection(right_words)))
        )
    if "government" in types and "person" in types:
        government = left if left_type == "government" else right
        person = right if government is left else left
        stem = _administration_stem(government)
        surname = _person_surname(person)
        person_key = _name_key(person.get("canonical_name") or person.get("name"))
        return bool(stem and len(stem) >= 3 and (stem == surname or person_key.endswith(stem)))
    if types == {"government", "country"}:
        government = left if left_type == "government" else right
        country = right if government is left else left
        government_stem = _country_key(_administration_stem(government))
        country_key = _country_key(country.get("canonical_name"))
        return bool(
            government_stem and government_stem == country_key
            and any(term in _name_key(government.get("canonical_name")) for term in ("government", "centralgovernment"))
        )
    if types == {"government"}:
        keys = {left_key, right_key}
        has_executive_seat = any("whitehouse" in key for key in keys)
        has_named_administration = any(
            key.endswith(("administration", "presidency")) for key in keys
        )
        if has_executive_seat and has_named_administration:
            return True
    return False


def _importance(candidate: dict) -> float:
    try:
        return max(0.0, min(1.0, float(candidate.get("importance_score") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _group_transitively(candidates: list[dict]) -> list[list[dict]]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _same_decision_center(candidates[left], candidates[right]):
                union(left, right)
    groups: dict[int, list[dict]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(find(index), []).append(candidate)
    return list(groups.values())


def _representative(members: list[dict]) -> dict:
    def rank(candidate: dict) -> tuple[int, float]:
        entity_type = _clean(candidate.get("agent_type") or candidate.get("entity_type")).lower()
        name_key = _name_key(candidate.get("canonical_name"))
        if entity_type == "government" and any(
            term in name_key for term in ("administration", "government", "cabinet", "whitehouse", "executive")
        ):
            type_rank = 5
        elif entity_type == "government":
            type_rank = 4
        elif entity_type == "person":
            type_rank = 3
        elif entity_type == "organization":
            type_rank = 2
        elif entity_type == "country":
            type_rank = 1
        else:
            type_rank = 0
        return type_rank, _importance(candidate)
    return max(members, key=rank)


def _group_is_historical(members: list[dict], snapshot_date: str) -> bool:
    snapshot_year = _clean(snapshot_date)[:4]
    latest_years = [
        _clean(item.get("latest_timestamp"))[:4]
        for item in members if _clean(item.get("latest_timestamp"))[:4].isdigit()
    ]
    if not snapshot_year.isdigit() or not latest_years:
        return False
    names = " ".join(_clean(item.get("canonical_name")).casefold() for item in members)
    leadership_specific = "administration" in names or "former " in names or "presidency" in names
    return bool(leadership_specific and max(int(year) for year in latest_years) < int(snapshot_year))


def _group_country(members: list[dict]) -> str:
    countries = [_candidate_country(item) for item in members if _candidate_country(item)]
    return max(set(countries), key=countries.count) if countries else ""


def _group_is_executive(members: list[dict]) -> bool:
    representative = _representative(members)
    if _clean(representative.get("entity_type") or representative.get("agent_type")).lower() != "government":
        return False
    return any(term in _name_key(representative.get("canonical_name")) for term in (
        "administration", "government", "cabinet", "whitehouse", "executive",
    ))


def _should_project_to_executive(members: list[dict], executive_members: list[dict]) -> bool:
    if members is executive_members:
        return False
    representative = _representative(members)
    if _is_independent_institution(representative):
        return False
    if not _group_country(members) or _group_country(members) != _group_country(executive_members):
        return False
    entity_type = _clean(representative.get("entity_type") or representative.get("agent_type")).lower()
    if entity_type in {"country", "government"}:
        return True
    if entity_type == "person":
        title = _clean(representative.get("title")).casefold()
        return any(term in title for term in (
            "president", "chairman", "prime minister", "vice premier", "secretary", "minister",
        ))
    return False


def _project_records_to_executives(records: list[dict]) -> list[dict]:
    executive_by_country: dict[str, dict] = {}
    for record in records:
        country = _candidate_country(record)
        if not country or not _group_is_executive([record]):
            continue
        existing = executive_by_country.get(country)
        if existing is None or _importance(record) > _importance(existing):
            executive_by_country[country] = record
    projected = []
    for record in records:
        executive = executive_by_country.get(_candidate_country(record))
        if executive is not None and record is not executive and _should_project_to_executive([record], [executive]):
            executive["member_candidate_ids"].extend(
                item for item in record.get("member_candidate_ids", [])
                if item not in executive["member_candidate_ids"]
            )
            executive["importance_score"] = max(_importance(executive), _importance(record))
            continue
        projected.append(record)
    return projected


def normalize_agent_registry(
    parsed: dict[str, Any], candidates: list[dict[str, Any]], max_agents: int, snapshot_date: str,
) -> dict[str, Any]:
    """Normalize an LLM registry response exactly as the original pipeline did."""
    candidate_by_id = {
        _clean(candidate.get("candidate_id")): candidate
        for candidate in candidates if _clean(candidate.get("candidate_id"))
    }
    raw_resolutions = list(parsed.get("resolutions", [])) if isinstance(parsed.get("resolutions"), list) else []
    for field, status in (("historical", "historical_actor"), ("context", "context_entity"), ("uncertain", "uncertain")):
        raw_resolutions.extend(
            {"candidate_id": candidate_id, "status": status}
            for candidate_id in parsed.get(field, []) if candidate_id in candidate_by_id
        )
    resolution_by_candidate = {
        _clean(item.get("candidate_id")): item
        for item in raw_resolutions if isinstance(item, dict) and _clean(item.get("candidate_id")) in candidate_by_id
    }
    explicitly_non_agent = {
        candidate_id for candidate_id, resolution in resolution_by_candidate.items()
        if _clean(resolution.get("status")).lower() in {"historical_actor", "context_entity", "uncertain"}
    }
    used: set[str] = set()
    agents: list[dict[str, Any]] = []
    allowed_types = {"person", "country", "government", "organization", "company", "group"}

    for raw_agent in parsed.get("agents", []):
        if not isinstance(raw_agent, dict):
            continue
        raw_agent_id = _clean(raw_agent.get("agent_id"))
        member_ids = [
            _clean(item) for item in raw_agent.get("member_candidate_ids", [])
            if _clean(item) in candidate_by_id
        ]
        member_ids.extend(
            candidate_id for candidate_id, resolution in resolution_by_candidate.items()
            if _clean(resolution.get("agent_id")) == raw_agent_id and candidate_id not in member_ids
        )
        if not member_ids:
            canonical_key = _name_key(raw_agent.get("canonical_name"))
            member_ids = [
                candidate_id for candidate_id, candidate in candidate_by_id.items()
                if _name_key(candidate.get("canonical_name")) == canonical_key
            ]
        member_ids = [item for item in member_ids if item not in used and item not in explicitly_non_agent]
        if not member_ids:
            continue
        members = [candidate_by_id[item] for item in member_ids]
        if _group_is_historical(members, snapshot_date):
            continue
        canonical_key = _name_key(raw_agent.get("canonical_name"))
        representative_candidates = [
            item for item in members
            if _name_key(item.get("canonical_name")) == canonical_key
        ]
        best = max(representative_candidates or members, key=_importance)
        affiliation = _clean(best.get("affiliation")) or next(
            (_clean(item.get("affiliation")) for item in members if _clean(item.get("affiliation"))),
            "",
        )
        title = _clean(best.get("title")) or next(
            (_clean(item.get("title")) for item in members if _clean(item.get("title"))),
            "",
        )
        canonical = _clean(raw_agent.get("canonical_name")) or _clean(best.get("canonical_name"))
        agent_type = _clean(raw_agent.get("agent_type")).lower()
        if agent_type not in allowed_types:
            agent_type = _clean(best.get("entity_type")).lower() or "person"
        proposed = {
            "canonical_name": canonical,
            "agent_type": agent_type,
            "affiliation": affiliation,
            "title": title,
        }
        duplicate = next((agent for agent in agents if _same_decision_center(agent, proposed)), None)
        if duplicate:
            duplicate["member_candidate_ids"].extend(
                item for item in member_ids if item not in duplicate["member_candidate_ids"]
            )
            duplicate["importance_score"] = max(duplicate["importance_score"], max(_importance(item) for item in members))
            if not _clean(duplicate.get("affiliation")) and affiliation:
                duplicate["affiliation"] = affiliation
            if not _clean(duplicate.get("title")) and title:
                duplicate["title"] = title
            if agent_type == "government" and duplicate.get("agent_type") != "government":
                duplicate.update({
                    "canonical_name": canonical,
                    "agent_type": agent_type,
                    "decision_subject": _clean(raw_agent.get("decision_subject")) or canonical,
                })
        else:
            duplicate = {
                "agent_id": f"agent_{len(agents) + 1:03d}",
                "canonical_name": canonical,
                "agent_type": agent_type,
                "decision_subject": _clean(raw_agent.get("decision_subject")) or canonical,
                "member_candidate_ids": list(member_ids),
                "reason": _clean(raw_agent.get("reason")),
                "importance_score": max(_importance(item) for item in members),
                "title": title,
                "affiliation": affiliation,
                "latest_timestamp": _clean(best.get("latest_timestamp")),
            }
            agents.append(duplicate)
        used.update(member_ids)

    coalesced = []
    for group in _group_transitively(agents):
        representative = _representative(group)
        member_ids = []
        for item in group:
            member_ids.extend(value for value in item.get("member_candidate_ids", []) if value not in member_ids)
        coalesced.append({
            **representative,
            "canonical_name": _clean(representative.get("canonical_name")),
            "agent_type": _clean(representative.get("agent_type")) or "unknown",
            "decision_subject": _clean(representative.get("decision_subject")) or _clean(representative.get("canonical_name")),
            "member_candidate_ids": member_ids,
            "importance_score": max(_importance(item) for item in group),
            "affiliation": _clean(representative.get("affiliation")) or next(
                (_clean(item.get("affiliation")) for item in group if _clean(item.get("affiliation"))),
                "",
            ),
            "title": _clean(representative.get("title")) or next(
                (_clean(item.get("title")) for item in group if _clean(item.get("title"))),
                "",
            ),
        })
    agents = sorted(_project_records_to_executives(coalesced), key=_importance, reverse=True)[:max_agents]
    if not agents:
        raise ValueError("Agent Registry selected no usable decision centers")
    for index, agent in enumerate(agents, 1):
        agent["agent_id"] = f"agent_{index:03d}"

    assigned = {candidate_id: agent for agent in agents for candidate_id in agent["member_candidate_ids"]}
    resolutions = []
    name_mapping = {}
    for candidate_id, candidate in candidate_by_id.items():
        raw_resolution = resolution_by_candidate.get(candidate_id, {})
        agent = assigned.get(candidate_id)
        if agent:
            raw_status = _clean(raw_resolution.get("status")).lower()
            status = raw_status if raw_status in {"same_entity", "represented_by", "independent_agent"} else (
                "independent_agent" if _name_key(candidate.get("canonical_name")) == _name_key(agent["canonical_name"])
                else "represented_by"
            )
            resolution = {
                "candidate_id": candidate_id, "status": status, "agent_id": agent["agent_id"],
                "canonical_name": agent["canonical_name"],
                "reason": _clean(raw_resolution.get("reason")) or agent.get("reason", ""),
            }
            for source_name in [candidate.get("canonical_name"), *(candidate.get("aliases") or [])]:
                source = _clean(source_name)
                if source:
                    name_mapping[source] = agent["canonical_name"]
        else:
            raw_status = _clean(raw_resolution.get("status")).lower()
            status = raw_status if raw_status in REGISTRY_STATUSES - {"same_entity", "represented_by", "independent_agent"} else "context_entity"
            resolution = {
                "candidate_id": candidate_id, "status": status, "agent_id": None,
                "canonical_name": _clean(candidate.get("canonical_name")),
                "reason": _clean(raw_resolution.get("reason")) or "Not assigned to an active decision center.",
            }
        resolutions.append(resolution)
    return {"agents": agents, "resolutions": resolutions, "name_mapping": name_mapping, "source": "llm_registry"}
