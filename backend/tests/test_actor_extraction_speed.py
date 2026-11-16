from __future__ import annotations

import asyncio
import json

from app.layers.pipeline import llm_figure_service


def test_malformed_batch_json_retries_two_halves_without_losing_order(monkeypatch) -> None:
    calls = []

    async def fake_call(*args, **kwargs):
        calls.append(args[3])
        if len(calls) == 1:
            return '{"results": [invalid]}', {"finish_reason": "stop"}
        return json.dumps({"results": [
            {"entry_index": 1, "figures": [], "actions": []},
            {"entry_index": 2, "figures": [], "actions": []},
        ]}), {"finish_reason": "stop"}

    monkeypatch.setattr(llm_figure_service, "_call_openrouter_with_meta", fake_call)
    batch = [{"timestamp": "2026-03", "description": f"Entry {i}"} for i in range(4)]
    result = asyncio.run(llm_figure_service.extract_figures_from_batch("key", "model", "event", batch))
    assert len(calls) == 3
    assert [row[0] for row in result] == batch
    assert all(row[3]["status"] == "model_returned_empty" for row in result)


def test_malformed_batch_retry_is_bounded(monkeypatch) -> None:
    calls = 0

    async def fake_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return '{"results": [invalid]}', {"finish_reason": "stop"}

    monkeypatch.setattr(llm_figure_service, "_call_openrouter_with_meta", fake_call)
    batch = [{"timestamp": "2026-03", "description": f"Entry {i}"} for i in range(4)]
    result = asyncio.run(llm_figure_service.extract_figures_from_batch("key", "model", "event", batch))
    assert calls == 3
    assert all(row[3]["status"] == "parse_failed" for row in result)


def test_non_json_model_reasoning_does_not_trigger_recursive_batch_retries(monkeypatch) -> None:
    calls = 0

    async def fake_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return "We need answer only final JSON. Need analyze the event.", {
            "finish_reason": "stop",
            "content_length": 58,
        }

    monkeypatch.setattr(llm_figure_service, "_call_openrouter_with_meta", fake_call)
    batch = [
        {"timestamp": "2026-03", "description": f"Entry {index}"}
        for index in range(4)
    ]

    result = asyncio.run(llm_figure_service.extract_figures_from_batch(
        "key", "model", "event", batch,
    ))

    assert calls == 1
    assert len(result) == 4
    assert all(item[3]["status"] == "parse_failed" for item in result)


def test_truncated_json_still_qualifies_for_batch_splitting() -> None:
    assert llm_figure_service._looks_like_truncated_batch_json(
        '{"results":[{"entry_index":1,', {"finish_reason": "length"},
    )
    assert not llm_figure_service._looks_like_truncated_batch_json(
        "We need answer only final JSON.", {"finish_reason": "length"},
    )


def test_reasoning_only_response_is_not_retried_as_a_split_batch(monkeypatch) -> None:
    calls = 0

    async def fake_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["json_mode"] is True
        return "", {
            "finish_reason": "length",
            "content_length": 0,
            "has_reasoning": True,
        }

    monkeypatch.setattr(llm_figure_service, "_call_openrouter_with_meta", fake_call)
    batch = [
        {"timestamp": "2026-03", "description": f"Entry {index}"}
        for index in range(4)
    ]

    result = asyncio.run(llm_figure_service.extract_figures_from_batch(
        "key", "model", "event", batch,
    ))

    assert calls == 1
    assert all(item[3]["status"] == "reasoning_only" for item in result)
