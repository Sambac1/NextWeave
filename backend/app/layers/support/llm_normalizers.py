"""Validation and normalization helpers for LLM-generated structured data."""

import re
import uuid
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from app.layers.support.layer_rules import infer_entity_type as _infer_entity_type


class ExtractedFigureModel(BaseModel):
    name: str
    entity_type: str = ""
    role: str = ""
    title: str = ""
    affiliation: str = ""
    stance: str = ""
    confidence: float = Field(0.7, ge=0.0, le=1.0)


class ExtractedActionModel(BaseModel):
    source: str
    target: str = ""
    action: str
    result: str = ""
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    cameo_root_code: Optional[str] = None
    cameo_code: Optional[str] = None
    action_label: Optional[str] = None
    quad_class: Optional[str] = None
    action_confidence: Optional[float] = None
    timestamp_text: Optional[str] = None
    normalized_timestamp: Optional[str] = None
    timestamp_precision: Optional[str] = None


class TimelineEventModel(BaseModel):
    event_id: str = ""
    timestamp: str
    description: str
    date_precision: str = "day"
    date_original: str = ""
    date_range: dict[str, str] = Field(default_factory=dict)
    event_type: str = "other"
    actors: list[str] = Field(default_factory=list)
    evidence_text: str = ""
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    importance_score: float = Field(0.5, ge=0.0, le=1.0)
    source_sections: list[str] = Field(default_factory=list)
    source_event_ids: list[int] = Field(default_factory=list)
    source_event_count: int = 0


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _dedupe_texts(values: list[str], limit: int | None = None) -> list[str]:
    """Deduplicate text in input order, optionally keeping only the first limit entries."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _clean_text(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _coerce_confidence(value: object, default: float = 0.7) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


_STANCE_ALLOWED = {"initiator", "opponent", "supporter", "mediator", "neutral"}
_STANCE_RANK = {"initiator": 4, "opponent": 4, "supporter": 3, "mediator": 3, "neutral": 1, "": 0}


def _normalize_stance(value: object) -> str:
    text = _clean_text(value).lower()
    if text in _STANCE_ALLOWED:
        return text
    return ""


def _stance_rank(value: object) -> int:
    return _STANCE_RANK.get(_normalize_stance(value), 0)


def _validate_figure(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    candidate = dict(raw)
    candidate["name"] = _clean_text(
        candidate.get("name") or candidate.get("figure") or candidate.get("canonical_name")
    )
    if not candidate["name"]:
        return None
    candidate["entity_type"] = _infer_entity_type(
        candidate["name"],
        candidate.get("entity_type") or candidate.get("type") or "",
        candidate.get("title") or "",
    )
    candidate["role"] = _clean_text(candidate.get("role") or candidate.get("role_in_event") or "")
    candidate["title"] = _clean_text(candidate.get("title") or "")
    candidate["affiliation"] = _clean_text(candidate.get("affiliation") or "")
    candidate["stance"] = _normalize_stance(candidate.get("stance"))
    candidate["confidence"] = _coerce_confidence(candidate.get("confidence"), 0.7)
    try:
        return ExtractedFigureModel.model_validate(candidate).model_dump()
    except ValidationError:
        return None


def _validate_action(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    candidate = dict(raw)
    candidate["source"] = _clean_text(candidate.get("source") or candidate.get("actor") or "")
    candidate["target"] = _clean_text(candidate.get("target") or candidate.get("object") or "")
    candidate["action"] = _clean_text(candidate.get("action") or candidate.get("relation_type") or "")
    candidate["result"] = _clean_text(candidate.get("result") or candidate.get("description") or "")
    candidate["confidence"] = _coerce_confidence(candidate.get("confidence"), 0.7)
    candidate["cameo_root_code"] = _clean_text(candidate.get("cameo_root_code") or "") or None
    candidate["cameo_code"] = _clean_text(candidate.get("cameo_code") or "") or None
    candidate["action_label"] = _clean_text(candidate.get("action_label") or "") or None
    candidate["quad_class"] = _clean_text(candidate.get("quad_class") or "") or None
    candidate["action_confidence"] = _coerce_confidence(
        candidate.get("action_confidence"), candidate["confidence"]
    ) if candidate.get("action_confidence") not in (None, "") else None
    candidate["timestamp_text"] = _clean_text(
        candidate.get("timestamp_text") or candidate.get("timestamp") or ""
    ) or None
    candidate["normalized_timestamp"] = _normalize_timestamp(
        candidate.get("normalized_timestamp") or candidate.get("timestamp") or ""
    ) or None
    candidate["timestamp_precision"] = _clean_text(candidate.get("timestamp_precision") or "") or None
    if not candidate["source"] or not candidate["action"]:
        return None
    try:
        return ExtractedActionModel.model_validate(candidate).model_dump()
    except ValidationError:
        return None


TIMELINE_EVENT_TYPES = {
    "policy",
    "diplomatic",
    "economic",
    "military",
    "legal",
    "election",
    "protest",
    "communication",
    "agreement",
    "crisis",
    "technology",
    "other",
}

TIMELINE_PRECISIONS = {"day", "month", "year", "range", "approximate"}


def _unique_strings(values: object) -> list[str]:
    if isinstance(values, str):
        text = _clean_text(values)
        if not text:
            return []
        separators = ["\n", ";"]
        if "," in text and len(text) < 180:
            separators.append(",")
        items = [text]
        for sep in separators:
            if any(sep in item for item in items):
                next_items = []
                for item in items:
                    next_items.extend(item.split(sep))
                items = next_items
        values = items
    elif not isinstance(values, list):
        values = [] if values is None else [values]

    result: list[str] = []
    for item in values:
        text = _clean_text(item)
        if text and text not in result:
            result.append(text)
    return result


def _int_list(value: object) -> list[int]:
    if isinstance(value, list):
        items = value
    elif value in (None, ""):
        items = []
    else:
        items = [value]

    result: list[int] = []
    for item in items:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed not in result:
            result.append(parsed)
    return result


def _normalize_timestamp(value: object) -> str:
    import re as _re

    text = _clean_text(value)
    if not text:
        return ""
    match = _re.search(r"\b(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?\b", text)
    if not match:
        return text
    year, month, day = match.group(1), match.group(2), match.group(3)
    if month and day:
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    if month:
        return f"{year}-{month.zfill(2)}"
    return year


def _normalize_date_range(value: object) -> dict[str, str]:
    import re as _re

    start = end = ""
    if isinstance(value, dict):
        start = _normalize_timestamp(
            value.get("start") or value.get("from") or value.get("begin") or value.get("start_date")
        )
        end = _normalize_timestamp(
            value.get("end") or value.get("to") or value.get("finish") or value.get("end_date")
        )
    elif isinstance(value, list):
        if value:
            start = _normalize_timestamp(value[0])
        if len(value) > 1:
            end = _normalize_timestamp(value[-1])
    elif value:
        matches = _re.findall(r"\b\d{4}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b", str(value))
        if matches:
            start = _normalize_timestamp(matches[0])
            end = _normalize_timestamp(matches[-1]) if len(matches) > 1 else ""

    result: dict[str, str] = {}
    if start:
        result["start"] = start
    if end and end != start:
        result["end"] = end
    return result


def _infer_timeline_precision(value: object, timestamp: str, date_range: dict[str, str]) -> str:
    raw = _clean_text(value).lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "date": "day",
        "exact": "day",
        "exact_day": "day",
        "month_only": "month",
        "year_only": "year",
        "date_range": "range",
    }
    raw = aliases.get(raw, raw)
    if raw in TIMELINE_PRECISIONS:
        return raw
    if date_range.get("end"):
        return "range"
    if len(timestamp) >= 10:
        return "day"
    if len(timestamp) >= 7:
        return "month"
    if len(timestamp) >= 4:
        return "year"
    return "approximate"


def _normalize_timeline_type(value: object, description: str) -> str:
    raw = _clean_text(value).lower().replace(" ", "_").replace("-", "_")
    if raw in TIMELINE_EVENT_TYPES:
        return raw

    text = f"{raw} {description}".lower()
    checks = [
        ("legal", ("lawsuit", "court", "wto", "complaint", "ruling", "legal")),
        ("economic", ("tariff", "trade", "import", "export", "sanction", "market", "duty")),
        ("diplomatic", ("summit", "meeting", "negotiation", "talk", "diplomatic")),
        ("agreement", ("agreement", "deal", "truce", "accord", "signed")),
        ("military", ("war", "military", "attack", "troop", "missile")),
        ("protest", ("protest", "strike", "demonstration")),
        ("election", ("election", "vote", "campaign")),
        ("technology", ("technology", "telecom", "chip", "semiconductor", "huawei")),
        ("communication", ("announced", "said", "stated", "tweet", "warned")),
        ("policy", ("policy", "executive order", "regulation", "law", "bill")),
        ("crisis", ("crisis", "escalation", "emergency")),
    ]
    for event_type, keywords in checks:
        if any(keyword in text for keyword in keywords):
            return event_type
    return "other"


def _normalize_timeline_event(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None

    date_range = _normalize_date_range(raw.get("date_range") or raw.get("dateRange") or raw.get("range"))
    timestamp = _normalize_timestamp(
        raw.get("timestamp")
        or raw.get("Timestamp")
        or raw.get("date")
        or raw.get("Date")
        or raw.get("time")
        or raw.get("Time")
    )
    if not timestamp and date_range.get("start"):
        timestamp = date_range["start"]

    description = _clean_text(
        raw.get("description")
        or raw.get("Description")
        or raw.get("Event Description")
        or raw.get("event")
        or raw.get("Event")
    )
    if not timestamp or not description:
        return None

    source_ids = _int_list(
        raw.get("source_event_ids")
        or raw.get("source_ids")
        or raw.get("input_ids")
        or raw.get("input_id")
    )
    candidate = {
        "event_id": _clean_text(raw.get("event_id")) or str(uuid.uuid4()),
        "timestamp": timestamp,
        "description": description,
        "date_precision": _infer_timeline_precision(
            raw.get("date_precision") or raw.get("precision"), timestamp, date_range,
        ),
        "date_original": _clean_text(raw.get("date_original") or raw.get("original_date") or raw.get("date_text")),
        "date_range": date_range,
        "event_type": _normalize_timeline_type(raw.get("event_type") or raw.get("type") or raw.get("category"), description),
        "actors": _unique_strings(raw.get("actors") or raw.get("participants") or raw.get("key_figures")),
        "evidence_text": _clean_text(raw.get("evidence_text") or raw.get("evidence") or raw.get("source_text") or raw.get("quote"))[:500],
        "confidence": _coerce_confidence(raw.get("confidence"), 0.7),
        "importance_score": _coerce_confidence(raw.get("importance_score") or raw.get("importance"), 0.5),
        "source_sections": _unique_strings(raw.get("source_sections") or raw.get("source_section")),
        "source_event_ids": source_ids,
        "source_event_count": len(source_ids),
    }
    if not candidate["date_original"]:
        candidate["date_original"] = str(raw.get("date") or raw.get("timestamp") or timestamp)

    try:
        return TimelineEventModel.model_validate(candidate).model_dump()
    except ValidationError:
        return candidate


def _timeline_sort_key(event: dict) -> tuple[str, str]:
    return (str(event.get("timestamp", "")), str(event.get("description", "")))


def _timeline_signature(event: dict) -> tuple[str, str]:
    desc_key = "".join(str(event.get("description", "")).lower().split())[:220]
    return (str(event.get("timestamp", "")), desc_key)


def _merge_timeline_metadata(event: dict, candidates: list[dict]) -> dict:
    merged = dict(event)
    source_sections = _unique_strings(merged.get("source_sections"))
    actors = _unique_strings(merged.get("actors"))
    source_ids = _int_list(merged.get("source_event_ids"))

    for candidate in candidates:
        source_sections.extend(x for x in _unique_strings(candidate.get("source_sections")) if x not in source_sections)
        actors.extend(x for x in _unique_strings(candidate.get("actors")) if x not in actors)
        source_ids.extend(x for x in _int_list(candidate.get("source_event_ids")) if x not in source_ids)

        if not merged.get("evidence_text") and candidate.get("evidence_text"):
            merged["evidence_text"] = candidate.get("evidence_text", "")
        if not merged.get("date_original") and candidate.get("date_original"):
            merged["date_original"] = candidate.get("date_original", "")
        if not merged.get("date_range") and candidate.get("date_range"):
            merged["date_range"] = candidate.get("date_range", {})
        if merged.get("event_type") in ("", "other") and candidate.get("event_type"):
            merged["event_type"] = candidate.get("event_type")
        merged["confidence"] = max(
            _coerce_confidence(merged.get("confidence"), 0.7),
            _coerce_confidence(candidate.get("confidence"), 0.7),
        )
        merged["importance_score"] = max(
            _coerce_confidence(merged.get("importance_score"), 0.5),
            _coerce_confidence(candidate.get("importance_score"), 0.5),
        )

    merged["source_sections"] = source_sections
    merged["actors"] = actors
    merged["source_event_ids"] = sorted(source_ids)
    merged["source_event_count"] = len(source_ids)
    return merged


def _select_timeline_source_candidates(event: dict, candidates: list[dict]) -> list[dict]:
    source_ids = set(_int_list(event.get("source_event_ids")))
    matched: list[dict] = []
    if source_ids:
        for candidate in candidates:
            if source_ids.intersection(_int_list(candidate.get("source_event_ids"))):
                matched.append(candidate)
    if matched:
        return matched

    timestamp = event.get("timestamp", "")
    description = event.get("description", "")
    for candidate in candidates:
        if candidate.get("timestamp") != timestamp:
            continue
        candidate_desc = candidate.get("description", "")
        if (
            candidate_desc == description
            or candidate_desc in description
            or description in candidate_desc
        ):
            matched.append(candidate)
    if matched:
        return matched

    return [candidate for candidate in candidates if candidate.get("timestamp") == timestamp][:3]


def _dedupe_timeline_events_exact(events: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}
    for raw in events:
        event = _normalize_timeline_event(raw)
        if not event:
            continue
        key = _timeline_signature(event)
        if key in merged:
            merged[key] = _merge_timeline_metadata(merged[key], [event])
        else:
            merged[key] = event
    return sorted(merged.values(), key=_timeline_sort_key)
