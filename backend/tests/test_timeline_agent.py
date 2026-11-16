from __future__ import annotations

import asyncio

from app.knowledge.timeline_agent import (
    TIMELINE_AGENT_MAX_HISTORICAL,
    TIMELINE_AGENT_SECTION_CHARS,
    TIMELINE_AGENT_TOTAL_CHARS,
    TimelineAgentResult,
    build_agent_timeline,
    is_reference_section,
    render_sections,
    select_agent_sections,
    to_events,
)

REFERENCE_SECTION_MARKERS = ("footnote", "reference", "bibliograph", "参见", "参考资料")


def section(index: int, *, path: str | None = None, title: str = "", content: str = "Alice acted on 2026-01-01.", **extra):
    return {
        "path": path or f"source_01/02_Chronology/{index:02d}_Year {index}",
        "title": title or f"Year {index}",
        "summary": "A dated event with useful detail.",
        "content": content,
        "has_accurate_dates": True,
        "timeline_extractable": True,
        **extra,
    }


def test_sections_are_capped_per_section_and_in_total() -> None:
    oversize = section(1, content="x" * (TIMELINE_AGENT_SECTION_CHARS + 500))
    full = [section(index, content="y" * TIMELINE_AGENT_SECTION_CHARS) for index in range(2, 10)]
    small = section(20, content="z" * 3000)

    sent, dropped = select_agent_sections([oversize, *full, small])

    assert len(sent[0]["content"]) == TIMELINE_AGENT_SECTION_CHARS
    assert all(len(item["content"]) <= TIMELINE_AGENT_SECTION_CHARS for item in sent)
    assert sum(len(item["content"]) for item in sent) <= TIMELINE_AGENT_TOTAL_CHARS
    assert dropped == ["source_01/02_Chronology/08_Year 8", "source_01/02_Chronology/09_Year 9"]
    # A rejected section does not end the pass: the next, smaller one still fits
    # in what the budget has left.
    assert sent[-1]["path"] == "source_01/02_Chronology/20_Year 20"


def test_reference_sections_are_demoted_but_never_silently_dropped() -> None:
    sections = [section(1, path="source_01/09_References/01_Footnotes", title="Footnotes")] + [
        section(index) for index in range(2, 5)
    ]

    sent, dropped = select_agent_sections(sections)

    assert sent[0]["path"] == "source_01/02_Chronology/02_Year 2"
    assert sent[-1]["path"] == "source_01/09_References/01_Footnotes"
    assert dropped == []
    assert is_reference_section({"path": "source_01/09_References/01_Footnotes"})
    for marker in REFERENCE_SECTION_MARKERS:
        assert is_reference_section({"path": f"source_01/10_{marker}/01_Page"})
        assert is_reference_section({"title": marker})
    # "Notes on the negotiations" is real narrative, not a footnote list.
    assert not is_reference_section({"path": "source_01/03_Notes/01_Notes on the negotiations"})


def test_sections_without_readable_content_are_reported_as_dropped() -> None:
    sent, dropped = select_agent_sections([section(1, content=""), section(2)])

    assert [item["path"] for item in sent] == ["source_01/02_Chronology/02_Year 2"]
    assert dropped == ["source_01/02_Chronology/01_Year 1"]


def test_rendered_sections_keep_cjk_readable() -> None:
    rendered = render_sections([{"path": "source_01/a", "title": "标题", "summary": "摘要", "content": "内容"}])

    assert "标题" in rendered
    assert "\\u" not in rendered
    assert rendered.startswith("===== SECTION 1 =====")


def test_history_is_normalised_and_deduplicated_across_sections() -> None:
    payload = {
        "historical": [
            {"timestamp": "2026-01-01", "description": "Alice  acted", "event_type": "policy",
             "date_precision": "day", "confidence": 0.6, "actors": ["Alice"],
             "source_sections": ["source_01/a"]},
            {"timestamp": "2026-01-01", "description": "alice acted", "event_type": "policy",
             "date_precision": "day", "confidence": 0.9, "actors": ["Bob"],
             "source_sections": ["source_01/b"]},
            {"timestamp": "", "description": "No date, no event"},
            {"timestamp": "2026-02-01", "description": "", "event_type": "not-a-type"},
            {"timestamp": "2026-03-01", "description": "Undated precision", "event_type": "made-up"},
        ],
        "analysis": "Reading.",
    }

    historical, stats = to_events(payload, {"source_01/a", "source_01/b"})

    assert [item["timestamp"] for item in historical] == ["2026-01-01", "2026-03-01"]
    assert historical[0]["actors"] == ["Alice", "Bob"]
    assert historical[0]["source_sections"] == ["source_01/a", "source_01/b"]
    assert historical[0]["confidence"] == 0.9
    assert historical[1]["event_type"] == "other"
    assert historical[1]["date_precision"] == "day"
    assert stats["historical_returned"] == 5
    assert stats["historical_invalid"] == 2
    assert stats["truncated"] is False


def test_timeline_requires_a_resolvable_month_and_normalises_supported_dates() -> None:
    payload = {
        "historical": [
            {"timestamp": "2024-05-17", "description": "Day precision"},
            {"timestamp": "April 2023", "description": "Month precision"},
            {"timestamp": "2022", "date_original": "March 2022", "description": "Month in source phrase"},
            {"timestamp": "2021", "evidence_text": "In September, the agreement was signed.", "description": "Month clarified by context"},
            {"timestamp": "2020", "description": "Year only; month is unknown"},
            {"timestamp": "2020-13", "description": "Invalid month"},
        ],
    }

    historical, stats = to_events(payload, None)

    assert [item["timestamp"] for item in historical] == [
        "2021-09", "2022-03", "2023-04", "2024-05-17",
    ]
    assert [item["date_precision"] for item in historical] == [
        "month", "month", "month", "day",
    ]
    assert stats["historical_missing_month"] == 2


def test_history_keeps_the_latest_events() -> None:
    payload = {
        "historical": [
            {
                "timestamp": f"2010-{index // 28 + 1:02d}-{index % 28 + 1:02d}",
                "description": f"Event {index}",
            }
            for index in range(TIMELINE_AGENT_MAX_HISTORICAL + 1)
        ],
    }

    historical, stats = to_events(payload, None)

    assert len(historical) == TIMELINE_AGENT_MAX_HISTORICAL
    # The newest recorded events are the ones the window and the simulation ask
    # about, so overflow drops the oldest instead.
    assert historical[0]["description"] == "Event 1"
    assert historical[-1]["description"] == f"Event {TIMELINE_AGENT_MAX_HISTORICAL}"
    assert stats["truncated"] is True


def test_source_sections_outside_the_sent_evidence_are_dropped() -> None:
    payload = {
        "historical": [{
            "timestamp": "2026-01-01",
            "description": "Alice acted",
            "source_sections": ["source_01/a", "source_01/never-sent"],
        }],
    }

    historical, _ = to_events(payload, {"source_01/a"})

    assert historical[0]["source_sections"] == ["source_01/a"]


def test_malformed_payload_yields_empty_events_instead_of_raising() -> None:
    for payload in (None, [], {"historical": "nope"}, {"historical": [None, 3]}):
        historical, stats = to_events(payload, None)
        assert historical == []
        assert stats["historical_returned"] in (0, 2)


def test_a_failing_agent_reports_the_reason_instead_of_raising() -> None:
    class Client:
        async def json_call(self, **kwargs):
            raise RuntimeError("model unavailable")

    outcome = asyncio.run(build_agent_timeline("Event", [section(1)], Client()))

    assert outcome.applied is False
    assert "model unavailable" in outcome.reason
    assert outcome.historical == []
    assert outcome.artifact["applied"] is False
    assert outcome.artifact["sections_sent"]


def test_an_agent_that_returns_no_history_is_not_applied() -> None:
    class Client:
        async def json_call(self, **kwargs):
            return TimelineAgentResult.model_validate({
                "historical": [],
                "analysis": "",
            })

    outcome = asyncio.run(build_agent_timeline("Event", [section(1)], Client()))

    assert outcome.applied is False
    assert outcome.historical == []
    # The raw payload survives in the artifact: otherwise "every item was
    # rejected" leaves nothing at all to diagnose.
    assert outcome.artifact["raw"]["historical"] == []
