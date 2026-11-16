"""One-shot Timeline Agent: the whole recorded history in a single pass."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.knowledge.timeline import _dedupe_exact
from app.llm.client import LLMClient
from app.prompts.knowledge import TIMELINE_AGENT_PROMPT

TIMELINE_AGENT_SECTION_CHARS = 6000
TIMELINE_AGENT_TOTAL_CHARS = 45000
TIMELINE_AGENT_MAX_HISTORICAL = 60
_MONTH_NAMES = {
    name.lower(): index
    for index, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
        1,
    )
}
_MONTH_NAMES.update({name[:3].lower(): index for name, index in list(_MONTH_NAMES.items())})

# Closed sets, mirroring TIMELINE_EXTRACTION_PROMPT. Values outside them are
# normalised back rather than rejected, so one bad label cannot lose an event.
TIMELINE_EVENT_TYPES = {
    "policy", "diplomatic", "economic", "military", "legal", "election",
    "protest", "communication", "agreement", "crisis", "technology", "other",
}
TIMELINE_PRECISIONS = {"day", "month", "year", "range", "approximate"}

_REFERENCE_SECTION_MARKERS = (
    "footnote", "reference", "bibliograph", "citation", "see also",
    "external link", "further reading",
    "参考资料", "参考文献", "脚注", "外部链接", "参见", "延伸阅读",
)


class TimelineAgentHistoricalItem(BaseModel):
    timestamp: str = ""
    description: str = ""
    date_precision: str = "approximate"
    date_original: str = ""
    time_role: str = "occurrence"
    modality: str = "occurred"
    date_evidence: dict[str, Any] = Field(default_factory=dict)
    date_range: dict[str, str] = Field(default_factory=dict)
    evidence_text: str = ""
    actors: list[str] = Field(default_factory=list)
    event_type: str = "other"
    confidence: float = Field(default=0.7, ge=0, le=1)
    importance_score: float = Field(default=0.5, ge=0, le=1)
    source_sections: list[str] = Field(default_factory=list)


class TimelineAgentResult(BaseModel):
    # The list carries no max_length on purpose.
    historical: list[TimelineAgentHistoricalItem] = Field(default_factory=list)
    analysis: str = ""


class TimelineAgentOutcome(BaseModel):
    applied: bool
    reason: str = ""
    historical: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    artifact: dict[str, Any] = Field(default_factory=dict)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _unique_texts(values: Any) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = _clean_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)


def _infer_precision(value: str) -> str:
    if not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", value):
        return "approximate"
    if len(value) == 10:
        return "day"
    if len(value) == 7:
        return "month"
    return "year"


def _resolve_timeline_month(raw: dict[str, Any]) -> str | None:
    """Return YYYY-MM[-DD] only when the payload supports at least a month.

    The Timeline Agent may use a month stated in nearby prose while keeping the
    year in ``timestamp`` (for example, timestamp ``2024`` and evidence
    ``In March, ...``). It may not guess a month from a year alone.
    """
    date_range = raw.get("date_range") if isinstance(raw.get("date_range"), dict) else {}
    timestamp = _clean_text(raw.get("timestamp"))
    if re.search(r"quarter|季度|\bQ[1-4]\b", _clean_text(raw.get("date_original")), re.I):
        return None
    candidates = [timestamp, _clean_text(date_range.get("start"))]
    context_values = [
        timestamp,
        _clean_text(date_range.get("start")),
        _clean_text(raw.get("date_original")),
        _clean_text(raw.get("evidence_text")),
        _clean_text(raw.get("description")),
    ]

    # Explicit numeric ISO-like forms, including Chinese year/month notation.
    explicit_month = re.compile(
        r"(?<!\d)((?:19|20|21)\d{2})\s*(?:[-/.年])\s*"
        r"(0?[1-9]|1[0-2])(?!\d)"
        r"(?:\s*(?:[-/.月])\s*(0?[1-9]|[12]\d|3[01])(?!\d)\s*日?)?"
    )
    for value in candidates + context_values[2:]:
        match = explicit_month.search(value)
        if not match:
            continue
        year, month, day = int(match.group(1)), int(match.group(2)), match.group(3)
        if day:
            try:
                return datetime(year, month, int(day)).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return f"{year:04d}-{month:02d}"

    # Month names with a year, or a month-only phrase paired with the explicit
    # year in timestamp. This lets chronology/evidence clarify a year-only field.
    month_pattern = re.compile(
        r"\b(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")\b"
        r"(?:\s+\d{1,2}(?:st|nd|rd|th)?,?)?\s+((?:19|20|21)\d{2})\b",
        re.IGNORECASE,
    )
    month_only_pattern = re.compile(
        r"\b(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    years = {
        int(match.group())
        for value in context_values
        for match in re.finditer(r"(?<!\d)((?:19|20|21)\d{2})(?!\d)", value)
    }
    for value in context_values:
        match = month_pattern.search(value)
        if match:
            month = _MONTH_NAMES[match.group(1).lower()]
            return f"{int(match.group(2)):04d}-{month:02d}"
    if len(years) == 1:
        year = next(iter(years))
        # Do not scan free-form descriptions for a bare "May": in English it
        # is also a very common modal verb. Date phrases/evidence are safer.
        for value in context_values[2:4]:
            match = month_only_pattern.search(value)
            if match:
                month_name = match.group(1).lower()
                if month_name in {"may"} and not re.search(
                    r"\b(?:in|during)\s+may\b|^may$", value, re.IGNORECASE
                ):
                    continue
                return f"{year:04d}-{_MONTH_NAMES[month_name]:02d}"
    return None


def is_reference_section(meta: dict[str, Any]) -> bool:
    """True for carried apparatus -- footnotes, citations, link farms.

    Every path segment is searched, not only the leaf: a reference *chapter*
    makes each of its pages reference material too. The leading "NN_" ordering
    prefix is stripped first so "09_References" reads as "references".
    """
    path = str(meta.get("path") or "")
    segments = "/".join(re.sub(r"^\d+_", "", part) for part in path.split("/")).lower()
    title = str(meta.get("title") or "").lower()
    apparatus = re.compile(r"^(?:footnotes?|references?(?: and notes)?|bibliography|citations?|see also|external links?|further reading|参考资料|参考文献|脚注|外部链接|参见|延伸阅读)(?: \(part \d+\))?$", re.I)
    return any(apparatus.fullmatch(part.strip()) for part in [*segments.split("/"), title]) or any(part.strip() in _REFERENCE_SECTION_MARKERS for part in [*segments.split("/"), title])


def _rank_for_agent(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same priority as evidence selection: dated first, then longest summary."""
    return sorted(
        sections,
        key=lambda item: (
            not bool(item.get("has_accurate_dates")),
            -len(str(item.get("summary") or "")),
            str(item.get("path") or ""),
        ),
    )


def select_agent_sections(sections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Pick the sections to send, within the per-section and total budget.

    Ordering is not alphabetical: chapters are numbered, so an alphabetical
    budget would drop the most recent chapters first -- exactly the material the
    timeline needs most.
    """
    ranked = _rank_for_agent(sections)
    ordered = [item for item in ranked if not is_reference_section(item)]
    ordered += [item for item in ranked if is_reference_section(item)]

    sent: list[dict[str, Any]] = []
    dropped: list[str] = []
    used = 0
    for meta in ordered:
        path = str(meta.get("path") or "")
        content = str(meta.get("content") or "")[:TIMELINE_AGENT_SECTION_CHARS]
        if not content:
            dropped.append(path)
            continue
        if used + len(content) > TIMELINE_AGENT_TOTAL_CHARS:
            # Keep going rather than stopping: a later, smaller section may
            # still fit inside what the budget has left.
            dropped.append(path)
            continue
        used += len(content)
        sent.append({
            "path": path,
            "title": str(meta.get("title") or ""),
            "summary": str(meta.get("summary") or ""),
            "content": content,
        })
    return sent, dropped


def render_sections(sections: list[dict[str, Any]]) -> str:
    """Plain delimited text. JSON encoding would escape CJK into \\uXXXX."""
    blocks = []
    for index, section in enumerate(sections, 1):
        blocks.append(
            f"===== SECTION {index} =====\n"
            f"path: {section['path']}\n"
            f"title: {section['title']}\n"
            f"summary: {section['summary']}\n"
            f"content:\n{section['content']}"
        )
    return "\n\n".join(blocks)


def normalize_historical_event(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    timestamp = _resolve_timeline_month(raw)
    description = _clean_text(raw.get("description"))
    if not timestamp or not description:
        return None
    date_range = raw.get("date_range")
    event_type = _clean_text(raw.get("event_type"))
    if event_type not in TIMELINE_EVENT_TYPES:
        event_type = "other"
    precision = "day" if len(timestamp) == 10 else "month"
    return {
        "timestamp": timestamp,
        "description": description,
        "date_precision": precision,
        "date_original": _clean_text(raw.get("date_original")),
        "time_role": raw.get("time_role") or "occurrence",
        "modality": raw.get("modality") or "occurred",
        "date_evidence": raw.get("date_evidence") or {},
        "date_range": {str(key): str(value) for key, value in date_range.items()}
        if isinstance(date_range, dict) else {},
        "event_type": event_type,
        "actors": _unique_texts(raw.get("actors")),
        "evidence_text": _clean_text(raw.get("evidence_text"))[:500],
        "confidence": _safe_float(raw.get("confidence"), 0.7),
        "importance_score": _safe_float(raw.get("importance_score"), 0.5),
        "source_sections": _unique_texts(raw.get("source_sections")),
        "source_event_ids": list(raw.get("source_event_ids") or []),
        "source_event_count": int(raw.get("source_event_count") or 0),
    }


def to_events(
    payload: Any,
    valid_paths: set[str] | None = None,
    *, limit: int | None = TIMELINE_AGENT_MAX_HISTORICAL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = payload if isinstance(payload, dict) else {}
    raw_historical = result.get("historical")
    raw_historical = raw_historical if isinstance(raw_historical, list) else []

    normalized = [item for item in (normalize_historical_event(value) for value in raw_historical) if item]
    missing_month = sum(
        1 for value in raw_historical
        if isinstance(value, dict)
        and _clean_text(value.get("description"))
        and _resolve_timeline_month(value) is None
    )
    if valid_paths is not None:
        for event in normalized:
            event["source_sections"] = [path for path in event["source_sections"] if path in valid_paths]
    # _dedupe_exact also unions source_sections/actors and takes the highest
    # confidence across exact duplicates. That union is the cross-section merge
    # this path would otherwise lose by skipping the per-section merge pass.
    historical = _dedupe_exact(normalized)

    truncated = limit is not None and len(historical) > limit
    stats = {
        "historical_returned": len(raw_historical),
        "historical_invalid": len(raw_historical) - len(normalized),
        "historical_missing_month": missing_month,
        "historical_events": min(len(historical), limit) if limit is not None else len(historical),
        "truncated": truncated,
    }
    # The agent returns history oldest-first, so an overflow means the extras are
    # the *latest* events -- and the time window and the simulation context both
    # ask about the recent end, so the tail is what to keep.
    return (historical[-limit:] if limit is not None else historical), stats


def build_artifact(
    event_title: str,
    *,
    applied: bool,
    reason: str,
    sent: list[dict[str, Any]],
    dropped: list[str],
    stats: dict[str, Any] | None = None,
    payload: Any = None,
) -> dict[str, Any]:
    """Audit record for one agent attempt.

    Per-section character counts are kept because the sent paths alone do not
    say how the budget was spent, and the raw lists are kept because the failure
    that most needs explaining -- a payload whose every item was rejected --
    leaves nothing behind in timeline.json or timeline_candidates.json.
    """
    result = payload if isinstance(payload, dict) else {}
    raw_historical = result.get("historical")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_title": event_title,
        "applied": bool(applied),
        "reason": reason,
        "sections_sent": [{"path": item["path"], "chars": len(item["content"])} for item in sent],
        "sections_dropped": list(dropped),
        "stats": dict(stats or {}),
        "analysis": _clean_text(result.get("analysis"))[:2000],
        "raw": {
            "historical": raw_historical if isinstance(raw_historical, list) else [],
        },
    }


async def build_agent_timeline(
    event_title: str,
    sections: list[dict[str, Any]],
    client: LLMClient,
) -> TimelineAgentOutcome:
    """One Agent pass over the selected sections.

    Never raises. An agent failure must cost a warning and a fallback to the
    per-section path, not the whole extraction run.
    """
    sent, dropped = select_agent_sections(sections)
    valid_paths = {str(item.get("path") or "") for item in sections}
    if not sent:
        reason = "no readable content in the selected sections"
        return TimelineAgentOutcome(
            applied=False,
            reason=reason,
            artifact=build_artifact(
                event_title, applied=False, reason=reason, sent=[], dropped=dropped
            ),
        )

    try:
        result = await client.json_call(
            call_name="timeline_agent_build",
            system=(
                "You build the whole event timeline in one pass: the record of what happened, "
                "oldest first, drawn only from the supplied sections."
            ),
            prompt=TIMELINE_AGENT_PROMPT.format(
                event_title=event_title,
                sections=render_sections(sent),
            ),
            schema=TimelineAgentResult,
            temperature=0.2,
            max_tokens=6000,
        )
        payload = result.model_dump()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return TimelineAgentOutcome(
            applied=False,
            reason=reason,
            artifact=build_artifact(
                event_title, applied=False, reason=reason, sent=sent, dropped=dropped
            ),
        )

    historical, stats = to_events(payload, valid_paths)
    # Zero historical events is not success: accepting it would commit a
    # timeline with no history at all.
    applied = bool(historical)
    reason = "" if applied else "TimelineAgent returned no usable historical events"
    return TimelineAgentOutcome(
        applied=applied,
        reason=reason,
        historical=historical,
        stats=stats,
        artifact=build_artifact(
            event_title,
            applied=applied,
            reason=reason,
            sent=sent,
            dropped=dropped,
            stats=stats,
            payload=payload,
        ),
    )
