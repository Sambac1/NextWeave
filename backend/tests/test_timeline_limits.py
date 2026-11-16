from __future__ import annotations

import asyncio
import json

from app.knowledge.timeline import (
    TIMELINE_DEDUP_BATCH_SIZE,
    TimelineDeduplication,
    merge_timelines,
    select_timeline_sections,
)


class DedupClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def json_call(self, **kwargs):
        marker = "Events JSON:\n"
        start = kwargs["prompt"].index(marker) + len(marker)
        end = kwargs["prompt"].index("\n\nEach input event", start)
        events = json.loads(kwargs["prompt"][start:end])
        self.calls.append((kwargs["call_name"], len(events)))
        return TimelineDeduplication.model_validate({"deduplicated": events})


def test_timeline_deduplication_uses_small_batches_without_a_global_second_pass() -> None:
    events = [{
        "timestamp": f"2026-01-{index + 1:02d}",
        "description": f"Unique event {index}",
        "confidence": 0.8,
        "importance_score": 0.6,
    } for index in range(55)]
    client = DedupClient()

    result = asyncio.run(merge_timelines([events], client))

    assert len(result) == 55
    assert len(client.calls) == 3
    assert all(size <= TIMELINE_DEDUP_BATCH_SIZE for _, size in client.calls)
    assert all("timeline_deduplication_all" not in name for name, _ in client.calls)


def test_legacy_selector_fallback_preserves_candidates_within_its_bound() -> None:
    calls: list[str] = []

    class Client:
        async def json_call(self, **kwargs):
            calls.append(kwargs["call_name"])
            raise RuntimeError("review unavailable")

    sections = [{
        "path": f"source_01/Chronology/{index}",
        "chapter": "source_01/Chronology",
        "summary": "A dated event with useful detail " * index,
        "has_accurate_dates": True,
        "timeline_extractable": True,
    } for index in range(1, 6)]

    selected = asyncio.run(select_timeline_sections("Event", sections, Client()))

    assert len(selected) == 5
    assert len(calls) == 1
