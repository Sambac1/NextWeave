from __future__ import annotations

import asyncio

from pydantic import BaseModel

from app.domain import LLMOptions
from app.domain.common import ENGLISH_OUTPUT_INSTRUCTION
from app.llm import client as client_module
from app.llm.client import LLMClient


class Result(BaseModel):
    selected_section_ids: list[str]


class TimelineItem(BaseModel):
    timestamp: str
    description: str


class TimelineResult(BaseModel):
    events: list[TimelineItem]


def test_deepseek_structured_calls_disable_thinking(monkeypatch) -> None:
    captured_body: dict = {}

    class Response:
        is_error = False

        @staticmethod
        def json() -> dict:
            return {
                "choices": [{
                    "message": {"content": '{"selected_section_ids":["section_1"]}'},
                    "finish_reason": "stop",
                }],
            }

    class AsyncClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

        async def post(self, url, *, headers, json):
            captured_body.update(json)
            return Response()

    monkeypatch.setattr(client_module.httpx, "AsyncClient", AsyncClient)
    client = LLMClient(LLMOptions(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    ))

    result = asyncio.run(client.json_call(
        call_name="section_selection_group_01",
        system="Select sections.",
        prompt="Select section_1.",
        schema=Result,
        max_tokens=1800,
    ))

    assert result.selected_section_ids == ["section_1"]
    assert ENGLISH_OUTPUT_INSTRUCTION in captured_body["messages"][0]["content"]
    assert captured_body["thinking"] == {"type": "disabled"}
    assert captured_body["response_format"] == {"type": "json_object"}


def test_non_deepseek_endpoint_does_not_receive_provider_specific_thinking(monkeypatch) -> None:
    captured_body: dict = {}

    class Response:
        is_error = False

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": {"content": '{"selected_section_ids":[]}'}}]}

    class AsyncClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

        async def post(self, url, *, headers, json):
            captured_body.update(json)
            return Response()

    monkeypatch.setattr(client_module.httpx, "AsyncClient", AsyncClient)
    client = LLMClient(LLMOptions(
        api_key="test-key",
        model="other-model",
        base_url="https://example.test/v1",
    ))

    asyncio.run(client.json_call(
        call_name="test",
        system="Test.",
        prompt="Test.",
        schema=Result,
    ))

    assert "thinking" not in captured_body
    assert "response_format" not in captured_body


def test_truncated_batch_recovers_complete_items_without_selecting_nested_object(monkeypatch) -> None:
    class Response:
        is_error = False

        @staticmethod
        def json() -> dict:
            return {
                "choices": [{
                    "message": {
                        "content": (
                            '{"events":['
                            '{"timestamp":"2026-01-01","description":"complete"},'
                            '{"timestamp":"2026-01-02","description":"truncated'
                        ),
                    },
                    "finish_reason": "length",
                }],
            }

    class AsyncClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

        async def post(self, url, *, headers, json):
            return Response()

    monkeypatch.setattr(client_module.httpx, "AsyncClient", AsyncClient)
    client = LLMClient(LLMOptions(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    ))

    result = asyncio.run(client.json_call(
        call_name="timeline",
        system="Extract events.",
        prompt="Extract events.",
        schema=TimelineResult,
    ))

    assert [item.timestamp for item in result.events] == ["2026-01-01"]
