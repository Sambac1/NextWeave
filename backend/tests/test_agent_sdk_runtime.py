from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from app.agents import runtime as runtime_module
from app.agents.runtime import AgentSDKClient
from app.domain import LLMOptions
from app.llm.client import LLMError


class Result(BaseModel):
    value: str


class Recorder:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict, float | None]] = []

    def write(self, call_name, phase, value, *, duration_ms=None):
        self.records.append((call_name, phase, value, duration_ms))


def client(recorder=None) -> AgentSDKClient:
    return AgentSDKClient(LLMOptions(
        api_key="test-key",
        model="test-model",
        base_url="https://example.test",
    ), recorder)


def test_agent_sdk_call_has_a_hard_timeout_and_cancels_work(monkeypatch) -> None:
    cancelled = asyncio.Event()
    recorder = Recorder()

    async def never_finishes(**kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    monkeypatch.setattr(runtime_module, "run_structured_agent", never_finishes)
    sdk = client(recorder)
    sdk.timeout = 0.01

    with pytest.raises(LLMError, match="timed out after"):
        asyncio.run(sdk.json_call(
            call_name="timeout_test",
            system="test",
            prompt="test",
            schema=Result,
        ))

    assert cancelled.is_set()
    phases = [record[1] for record in recorder.records]
    assert phases == ["agent_sdk_slot_acquired", "agent_sdk_timeout_error"]
    timeout_data = recorder.records[-1][2]
    assert timeout_data["queue_wait_seconds"] < 0.1
    assert timeout_data["execution_seconds"] >= 0.01


def test_agent_sdk_call_respects_its_concurrency_limit(monkeypatch) -> None:
    active = 0
    peak = 0

    async def bounded_call(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return Result(value=kwargs["name"])

    monkeypatch.setattr(runtime_module, "run_structured_agent", bounded_call)

    async def run() -> None:
        sdk = client()
        sdk._semaphore = asyncio.Semaphore(2)
        await asyncio.gather(*(
            sdk.json_call(
                call_name=f"concurrency_test_{index}",
                system="test",
                prompt="test",
                schema=Result,
            )
            for index in range(6)
        ))

    asyncio.run(run())
    assert peak == 2
