from __future__ import annotations

import asyncio
import copy

from app.layers.support import openrouter_client


class _Response:
    def __init__(self, status_code: int, body: dict, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text
        self.is_success = 200 <= status_code < 300

    def json(self) -> dict:
        return self._body


class _Client:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.payloads: list[dict] = []

    async def post(self, _url: str, *, json: dict, headers: dict) -> _Response:
        self.payloads.append(copy.deepcopy(json))
        return self.responses.pop(0)


class _ClientContext:
    def __init__(self, client: _Client) -> None:
        self.client = client

    async def __aenter__(self) -> _Client:
        return self.client

    async def __aexit__(self, *_args: object) -> None:
        return None


def _mock_http(monkeypatch, client: _Client) -> None:
    monkeypatch.setattr(
        openrouter_client,
        "shared_http_client",
        lambda **_kwargs: _ClientContext(client),
    )


def test_json_mode_requests_json_and_does_not_use_reasoning_as_content(monkeypatch) -> None:
    client = _Client([_Response(200, {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "", "reasoning_content": "internal reasoning"},
        }],
        "usage": {"completion_tokens": 10},
    })])
    _mock_http(monkeypatch, client)

    content, meta = asyncio.run(openrouter_client._call_openrouter_with_meta(
        "key", "model", None, "return JSON", 100, 0.0,
        base_url="https://api.deepseek.com/v1", json_mode=True,
    ))

    assert client.payloads[0]["response_format"] == {"type": "json_object"}
    assert client.payloads[0]["thinking"] == {"type": "disabled"}
    assert content == ""
    assert meta["has_reasoning"] is True


def test_json_mode_falls_back_only_when_provider_rejects_response_format(monkeypatch) -> None:
    client = _Client([
        _Response(400, {"error": {"message": "response_format is unsupported"}},
                  "response_format is unsupported"),
        _Response(200, {
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        }),
    ])
    _mock_http(monkeypatch, client)

    content, _meta = asyncio.run(openrouter_client._call_openrouter_with_meta(
        "key", "model", None, "return JSON", 100, 0.0,
        base_url="https://provider.example/v1", json_mode=True,
    ))

    assert len(client.payloads) == 2
    assert "response_format" in client.payloads[0]
    assert "response_format" not in client.payloads[1]
    assert content == "{}"


def test_deepseek_disables_thinking_for_non_json_alias_and_profile_calls(monkeypatch) -> None:
    client = _Client([_Response(200, {
        "choices": [{"finish_reason": "stop", "message": {"content": "direct answer"}}],
    })])
    _mock_http(monkeypatch, client)
    content, _meta = asyncio.run(openrouter_client._call_openrouter_with_meta(
        "key", "deepseek-flash", None, "merge aliases", 4096, .2,
        base_url="https://api.deepseek.com", json_mode=False,
    ))
    assert client.payloads[0]["thinking"] == {"type": "disabled"}
    assert "response_format" not in client.payloads[0]
    assert content == "direct answer"


def test_provider_specific_thinking_option_is_not_sent_to_other_hosts(monkeypatch) -> None:
    client = _Client([_Response(200, {
        "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
    })])
    _mock_http(monkeypatch, client)
    asyncio.run(openrouter_client._call_openrouter_with_meta(
        "key", "model", None, "return JSON", 100, .0,
        base_url="https://gateway.example/api.deepseek.com", json_mode=False,
    ))
    assert "thinking" not in client.payloads[0]
