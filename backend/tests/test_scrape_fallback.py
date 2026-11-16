"""Two-tier scraping: primary fetch fails -> Agent SDK fallback -> disk."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from app.domain import LLMOptions
from app.knowledge import sdk_scrape as sdk_module
from app.knowledge.scraper import SourceCollector, _is_usable


def _collector(fetch) -> SourceCollector:
    """A SourceCollector that skips __init__ (no httpx client) and uses ``fetch``."""
    collector = SourceCollector.__new__(SourceCollector)
    collector.fetch = fetch
    return collector


def _llm() -> LLMOptions:
    return LLMOptions(api_key="test-key", model="test-model", base_url="https://example.test")


# ── _is_usable ──────────────────────────────────────────────────────────────


def test_is_usable_predicate() -> None:
    assert _is_usable({"content": "x" * 200}) is True
    assert _is_usable({"content": "short"}) is False
    assert _is_usable({"content": ""}) is False
    assert _is_usable({"content": "x" * 500, "error": "timeout"}) is False
    assert _is_usable(None) is False


# ── fetch_with_fallback ─────────────────────────────────────────────────────


def test_primary_result_is_used_when_usable(monkeypatch) -> None:
    async def fake_fetch(url, save_directory):
        return {"title": "Page", "url": url, "content": "x" * 300}

    async def must_not_run(*args, **kwargs):
        raise AssertionError("sdk_scrape should not be called for a usable primary result")

    monkeypatch.setattr(sdk_module, "sdk_scrape", must_not_run)

    collector = _collector(fake_fetch)
    result = asyncio.run(collector.fetch_with_fallback(
        "https://x.example/wiki/Page", Path("."), _llm(),
    ))
    assert result["title"] == "Page"
    assert "sdk_fallback" not in result


def test_failed_primary_without_llm_options_returns_error(monkeypatch) -> None:
    async def fake_fetch(url, save_directory):
        return {"url": url, "error": "timeout"}

    async def must_not_run(*args, **kwargs):
        raise AssertionError("sdk_scrape requires llm_options and must not be called without it")

    monkeypatch.setattr(sdk_module, "sdk_scrape", must_not_run)

    collector = _collector(fake_fetch)
    result = asyncio.run(collector.fetch_with_fallback(
        "https://x.example/wiki/Page", Path("."), None,
    ))
    assert result == {"url": "https://x.example/wiki/Page", "error": "timeout"}


def test_failed_primary_uses_sdk_fallback(monkeypatch) -> None:
    async def fake_fetch(url, save_directory):
        return {"url": url, "error": "404"}

    received: dict = {}

    async def fake_sdk(url, options, save_directory):
        received["url"] = url
        received["options"] = options
        received["save_directory"] = save_directory
        return {
            "title": "Reconstructed", "url": "https://alt.example/wiki/Page",
            "domain": "alt.example", "summary": "s", "content": "x" * 300,
            "sections": [], "sdk_fallback": True, "sources": ["https://alt.example/wiki/Page"],
        }

    monkeypatch.setattr(sdk_module, "sdk_scrape", fake_sdk)

    options = _llm()
    collector = _collector(fake_fetch)
    result = asyncio.run(collector.fetch_with_fallback(
        "https://x.example/wiki/Page", Path("."), options,
    ))
    assert result["sdk_fallback"] is True
    assert received["url"] == "https://x.example/wiki/Page"
    assert received["options"] is options
    assert received["save_directory"] == Path(".")


def test_sdk_fallback_error_is_merged_with_primary_error(monkeypatch) -> None:
    async def fake_fetch(url, save_directory):
        return {"url": url, "error": "timeout"}

    async def broken_sdk(*args, **kwargs):
        raise RuntimeError("sdk down")

    monkeypatch.setattr(sdk_module, "sdk_scrape", broken_sdk)

    collector = _collector(fake_fetch)
    result = asyncio.run(collector.fetch_with_fallback(
        "https://x.example/wiki/Page", Path("."), _llm(),
    ))
    assert "timeout" in result["error"]
    assert "SDK fallback error" in result["error"]
    assert "sdk down" in result["error"]


def test_sdk_fallback_that_returns_short_content_is_treated_as_unusable(monkeypatch) -> None:
    async def fake_fetch(url, save_directory):
        return {"url": url, "error": "empty"}

    async def weak_sdk(*args, **kwargs):
        return {"title": "", "content": "tiny", "sdk_fallback": True}

    monkeypatch.setattr(sdk_module, "sdk_scrape", weak_sdk)

    collector = _collector(fake_fetch)
    result = asyncio.run(collector.fetch_with_fallback(
        "https://x.example/wiki/Page", Path("."), _llm(),
    ))
    # Fallback content is too short, so the original error result is returned.
    assert result == {"url": "https://x.example/wiki/Page", "error": "empty"}


# ── sdk_scrape (mocked Agent SDK runtime) ───────────────────────────────────


def test_sdk_scrape_writes_artifacts_and_returns_dict(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sdk_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_api_key="", llm_model="", llm_base_url="https://example.test",
            agent_sdk_timeout_seconds=5,
        ),
    )
    received: dict = {}

    async def fake_run_structured_agent(**kwargs):
        received.update(kwargs)
        return sdk_module.SDKScrapeResult(
            title="Alt title",
            summary="Alt summary",
            content="Reconstructed content " * 30,
            source_url="https://alt.example/wiki/Page",
            sources=["https://alt.example/wiki/Page"],
        )

    monkeypatch.setattr(sdk_module, "run_structured_agent", fake_run_structured_agent)

    save_dir = tmp_path / "source"
    result = asyncio.run(sdk_module.sdk_scrape(
        "https://x.example/wiki/Original", _llm(), save_dir,
    ))

    assert result["sdk_fallback"] is True
    assert result["content"]
    assert result["domain"] == "alt.example"
    assert (save_dir / "content.txt").read_text(encoding="utf-8") == result["content"]
    assert result["summary"] == "Alt summary"
    assert not (save_dir / "summary.txt").exists()

    # The agent got exactly the two retrieval tools and a bounded turn budget.
    assert received["max_turns"] == 12
    tool_names = [item[0] for item in received["extra_tools"]]
    assert tool_names == ["search_web", "fetch_url"]
