import asyncio
from contextlib import asynccontextmanager

import pytest
from pydantic import BaseModel

from app.domain import LLMOptions
from app.llm import client as module
from app.llm.client import LLMClient, LLMError


class Output(BaseModel):
    events: list[str]


def test_strict_json_rejects_a_truncated_root_and_accounts_each_attempt(monkeypatch):
    calls = []
    class Response:
        is_error = False
        def json(self):
            return {"choices": [{"finish_reason": "length", "message": {"content": '{"events":["first",'}}]}
    class Transport:
        async def post(self, *a, **kw): return Response()
    @asynccontextmanager
    async def pool(**kw): yield Transport()
    monkeypatch.setattr(module, "shared_http_client", pool)
    client = LLMClient(LLMOptions(api_key="fixture", model="fixture", base_url="https://example.org"))
    client.strict_json, client.max_attempts = True, 1
    client.before_attempt = lambda: calls.append(1)
    with pytest.raises(LLMError, match="Truncated"):
        asyncio.run(client.json_call(call_name="test", system="test", prompt="test", schema=Output))
    assert len(calls) == 1
