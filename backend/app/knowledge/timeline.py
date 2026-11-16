from __future__ import annotations

import asyncio
import json
import re

from pydantic import BaseModel, Field, RootModel

from app.llm.client import LLMClient
from app.prompts.knowledge import TIMELINE_DEDUP_PROMPT, TIMELINE_EXTRACTION_PROMPT


class SectionPathSelection(BaseModel):
    selected: list[str] = Field(default_factory=list)


class TimelineDraft(BaseModel):
    timestamp: str
    description: str
    date_precision: str = "approximate"
    date_original: str = ""
    time_role: str = "occurrence"
    modality: str = "occurred"
    date_evidence: dict = Field(default_factory=dict)
    date_range: dict[str, str] = Field(default_factory=dict)
    evidence_text: str = ""
    actors: list[str] = Field(default_factory=list)
    event_type: str = "other"
    confidence: float = Field(default=0.7, ge=0, le=1)
    importance_score: float = Field(default=0.5, ge=0, le=1)
    source_sections: list[str] = Field(default_factory=list)
    source_event_ids: list[int] = Field(default_factory=list)
    source_event_count: int = 0
    input_id: int | None = None


class TimelineBatch(BaseModel):
    events: list[TimelineDraft] = Field(default_factory=list)


class TimelineDeduplication(BaseModel):
    deduplicated: list[TimelineDraft] = Field(default_factory=list)


TIMELINE_DEDUP_BATCH_SIZE = 24
MAX_SELECTED_TIMELINE_SECTIONS = 12


async def select_timeline_sections(
    event_title: str,
    sections: list[dict],
    client: LLMClient,
) -> list[str]:
    """Rank sections locally and select from at most 32 candidates in one model call."""
    if not sections:
        return []

    words = {
        word.casefold()
        for word in re.findall(r"[\w'-]{3,}", event_title, flags=re.UNICODE)
        if word.casefold() not in {"the", "and", "for", "from", "with", "into", "about"}
    }
    temporal = re.compile(
        r"(?:\b(?:19|20)\d{2}\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|"
        r"election|agreement|attack|invasion|protest|war|crisis|ceasefire|sanction|resign|appoint|宣布|签署|袭击|战争|抗议|选举|协议|危机|制裁|辞职|任命)",
        re.IGNORECASE,
    )
    reference = re.compile(r"references|external links|bibliography|further reading|参考文献|外部链接", re.I)

    def rank(item: dict) -> tuple[int, int, int, str]:
        title = str(item.get("title") or "")
        text = f"{title} {item.get('summary') or ''} {item.get('content') or ''}"[:5000].casefold()
        title_matches = sum(1 for word in words if word in title.casefold())
        topic_matches = sum(1 for word in words if word in text)
        temporal_hits = len(temporal.findall(text))
        penalty = 10 if reference.search(title) else 0
        return (
            title_matches * 3 + min(topic_matches, 6) + min(temporal_hits, 5) * 2 - penalty,
            temporal_hits,
            len(str(item.get("content") or "")),
            str(item.get("path") or ""),
        )

    ranked = sorted(sections, key=rank, reverse=True)
    # Preserve coverage across source pages, then fill remaining slots by score.
    candidate_ids: set[str] = set()
    candidates: list[dict] = []
    source_counts: dict[str, int] = {}
    for item in ranked:
        source = str(item.get("path") or "").split("/", 1)[0]
        if source_counts.get(source, 0) >= 2:
            continue
        candidate_ids.add(str(item.get("section_id") or item.get("path")))
        candidates.append(item)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(candidates) >= 16:
            break
    for item in ranked:
        key = str(item.get("section_id") or item.get("path"))
        if key not in candidate_ids:
            candidates.append(item)
            candidate_ids.add(key)
        if len(candidates) >= 32:
            break

    fallback = [
        str(item.get("section_id") or item["path"])
        for item in ranked[:MAX_SELECTED_TIMELINE_SECTIONS]
    ]
    section_list = "\n".join(
        f"- {item.get('section_id') or item['path']} | {item['path']} | {item.get('title', '')} | dates={bool(item.get('has_accurate_dates'))}\n"
        f"  {str(item.get('summary') or item.get('content') or '')[:500]}"
        for item in candidates
    )
    valid_ids = {str(item.get("section_id") or item.get("path") or "") for item in candidates}
    try:
        result = await client.json_call(
            call_name="timeline_section_selection",
            system="Select only supplied sections containing useful historical events for the requested timeline. Return exact section IDs, at most 12, and include multiple periods and source pages when useful.",
            prompt=(
                f"Event: {event_title}\nSelect up to {MAX_SELECTED_TIMELINE_SECTIONS} section IDs from the candidates. "
                "Prefer dated decisions, actions, agreements, conflicts, and outcomes. Avoid references and static descriptions.\n"
                f"Candidates:\n{section_list}"
            ),
            schema=SectionPathSelection,
            temperature=0.1,
            max_tokens=2048,
        )
        selected = list(dict.fromkeys(section_id for section_id in result.selected if section_id in valid_ids))
        return selected[:MAX_SELECTED_TIMELINE_SECTIONS] or fallback
    except Exception:
        return fallback


async def extract_section_timeline(section: dict, client: LLMClient) -> list[dict]:
    result = await client.json_call(
        call_name=f"section_{section['section_id']}_timeline",
        system="Extract a structured timeline using the exact supplied rules.",
        prompt=TIMELINE_EXTRACTION_PROMPT.format(
            section_title=" / ".join(filter(None, (section.get("source_title"), section.get("path") or section["title"]))),
            content=section["content"][:15_000],
        ),
        schema=TimelineBatch,
        temperature=0.2,
        max_tokens=6144,
    )
    events = []
    for item in result.events:
        value = item.model_dump(exclude={"input_id"})
        value["source_sections"] = list(dict.fromkeys([*value["source_sections"], section["path"]]))
        events.append(value)
    return _dedupe_exact(events)


def _signature(event: dict) -> tuple[str, str]:
    description = "".join(str(event.get("description") or "").lower().split())
    return str(event.get("timestamp") or ""), description


def _merge_metadata(event: dict, candidates: list[dict]) -> dict:
    merged = dict(event)
    for field in ("source_sections", "actors", "source_event_ids"):
        merged[field] = list(dict.fromkeys([
            *list(merged.get(field) or []),
            *(item for candidate in candidates for item in list(candidate.get(field) or [])),
        ]))
    for candidate in candidates:
        if not merged.get("evidence_text"):
            merged["evidence_text"] = candidate.get("evidence_text", "")
        if not merged.get("date_original"):
            merged["date_original"] = candidate.get("date_original", "")
        if not merged.get("date_range"):
            merged["date_range"] = candidate.get("date_range", {})
        merged["confidence"] = max(float(merged.get("confidence") or 0.7), float(candidate.get("confidence") or 0.7))
        merged["importance_score"] = max(float(merged.get("importance_score") or 0.5), float(candidate.get("importance_score") or 0.5))
    merged["source_event_count"] = len(merged.get("source_event_ids") or [])
    return merged


def _dedupe_exact(events: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for event in events:
        key = _signature(event)
        if key in merged:
            merged[key] = _merge_metadata(merged[key], [event])
        else:
            merged[key] = dict(event)
    return sorted(merged.values(), key=lambda item: (str(item.get("timestamp") or ""), str(item.get("description") or "")))


def _time_groups(events: list[dict]) -> list[tuple[str, list[dict]]]:
    years = []
    for event in events:
        match = re.search(r"\b\d{4}\b", str(event.get("timestamp") or ""))
        if match:
            years.append(int(match.group()))
    if not years:
        return [("all", events)]
    if max(years) - min(years) >= 3:
        key = lambda item: str(item.get("timestamp") or "")[:4]
    elif max(years) - min(years) >= 1:
        def key(item):
            timestamp = str(item.get("timestamp") or "")
            month = int(timestamp[5:7]) if len(timestamp) >= 7 and timestamp[5:7].isdigit() else 1
            return f"{timestamp[:4]}_{'H1' if month <= 6 else 'H2'}"
    else:
        def key(item):
            timestamp = str(item.get("timestamp") or "")
            if len(timestamp) < 7 or not timestamp[5:7].isdigit():
                return timestamp[:4]
            month = int(timestamp[5:7])
            return f"{timestamp[:4]}_Q{(month - 1) // 3 + 1}"
    grouped: dict[str, list[dict]] = {}
    for event in events:
        grouped.setdefault(key(event), []).append(event)
    return sorted(grouped.items())


def _source_candidates(event: dict, candidates: list[dict]) -> list[dict]:
    source_ids = set(event.get("source_event_ids") or [])
    if source_ids:
        matched = [item for item in candidates if source_ids.intersection(item.get("source_event_ids") or [])]
        if matched:
            return matched
    timestamp = event.get("timestamp")
    description = str(event.get("description") or "")
    matched = [
        item for item in candidates
        if item.get("timestamp") == timestamp and (
            str(item.get("description") or "") == description
            or str(item.get("description") or "") in description
            or description in str(item.get("description") or "")
        )
    ]
    return matched or [item for item in candidates if item.get("timestamp") == timestamp][:3]


async def _deduplicate_batch(
    span: str,
    batch_number: int,
    candidates: list[dict],
    client: LLMClient,
) -> list[dict]:
    prompt_events = []
    for index, event in enumerate(candidates, 1):
        prompt_events.append({**event, "input_id": index})
    try:
        result = await client.json_call(
            call_name=f"timeline_deduplication_{span}_batch_{batch_number:02d}",
            system="Merge exact and near-duplicate historical events while preserving all source metadata.",
            prompt=TIMELINE_DEDUP_PROMPT.format(
                time_span=span,
                events_text=json.dumps(prompt_events, ensure_ascii=False, indent=2),
            ),
            schema=TimelineDeduplication,
            temperature=0.2,
            max_tokens=4096,
        )
        merged = []
        for item in result.deduplicated:
            value = item.model_dump(exclude={"input_id"})
            merged.append(_merge_metadata(value, _source_candidates(value, candidates)))
        if merged:
            return _dedupe_exact(merged)
    except Exception:
        pass
    return _dedupe_exact(candidates)


async def _deduplicate_group(span: str, events: list[dict], client: LLMClient) -> list[dict]:
    normalized = _dedupe_exact(events)
    batches = [
        normalized[index:index + TIMELINE_DEDUP_BATCH_SIZE]
        for index in range(0, len(normalized), TIMELINE_DEDUP_BATCH_SIZE)
    ]
    results = await asyncio.gather(*(
        _deduplicate_batch(span, batch_number, batch, client)
        for batch_number, batch in enumerate(batches, 1)
    ))
    return _dedupe_exact([event for result in results for event in result])


async def merge_timelines(timeline_lists: list[list[dict]], client: LLMClient) -> list[dict]:
    all_events = []
    source_number = 0
    for timeline in timeline_lists:
        for event in timeline:
            source_number += 1
            source_ids = list(dict.fromkeys([*(event.get("source_event_ids") or []), source_number]))
            all_events.append({**event, "source_event_ids": source_ids, "source_event_count": len(source_ids)})
    all_events = _dedupe_exact(all_events)
    if len(all_events) <= 1:
        return all_events
    groups = _time_groups(all_events)
    deduped = await asyncio.gather(*(
        _deduplicate_group(span, values, client) for span, values in groups
    ))
    # Time groups are disjoint. A second whole-timeline LLM pass adds a very
    # large request without a sound reason to merge events across time spans.
    return _dedupe_exact([event for group in deduped for event in group])
