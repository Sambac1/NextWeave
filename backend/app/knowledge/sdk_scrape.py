"""Agent-assisted fallback collection with source-specific excerpts and provenance."""

from __future__ import annotations

import asyncio
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from app.agents import run_structured_agent
from app.domain import LLMOptions
from app.domain.common import ENGLISH_OUTPUT_INSTRUCTION
from app.knowledge.scraper import USER_AGENT, split_sections
from app.settings import get_settings


_MIN_USABLE_CONTENT_CHARS = 200


class SDKScrapeResult(BaseModel):
    title: str = ""
    summary: str = ""
    content: str = ""
    source_url: str = ""
    sources: list[str] = Field(default_factory=list)
    source_excerpts: list[dict[str, str]] = Field(default_factory=list)


SEARCH_WEB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search keywords."},
        "top_k": {"type": "integer", "description": "Number of results to return. Defaults to 5."},
    },
    "required": ["query"],
}

FETCH_URL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "URL from search_web to fetch in full."},
    },
    "required": ["url"],
}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:  # noqa: BLE001
        return ""


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _block(source_id: str, body: str) -> str:
    return f"[source: {source_id}]\n{body}"


def _sections_from_excerpts(excerpts: list[dict[str, str]], source_number: int) -> list[dict]:
    """Build bounded source sections while preserving each excerpt's original URL."""
    sections: list[dict] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for item in excerpts or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        body = str(item.get("text") or "").strip()
        if not url or not body or url in seen_urls:
            continue
        seen_urls.add(url)
        title = str(item.get("title") or "").strip() or url
        if title in seen_titles:
            title = f"{title} ({urlparse(url).netloc or len(seen_titles) + 1})"
        seen_titles.add(title)
        parts = split_sections(body, source_number)
        multi = len(parts) > 1
        for index, part in enumerate(parts, 1):
            suffix = f" (part {index})" if multi else ""
            sections.append({
                "section_id": f"source_{source_number:02d}_section_{len(sections) + 1:03d}",
                "title": f"{title}{suffix}",
                "path": f"source_{source_number:02d}/{title}{suffix}",
                "chapter": f"source_{source_number:02d}",
                "content": part["content"],
                "url": url,
            })
    return sections


async def _search_tavily(query: str, top_k: int, api_key: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": top_k},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content") or item.get("snippet", ""),
        }
        for item in data.get("results", [])[:top_k]
    ]


async def _search_wikipedia(query: str, top_k: int) -> list[dict]:
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    headers = {"User-Agent": USER_AGENT}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": str(top_k),
        "srprop": "snippet",
    }
    try:
        async with httpx.AsyncClient(
            timeout=25.0, headers=headers, proxy=proxy, follow_redirects=True
        ) as client:
            resp = await client.get("https://en.wikipedia.org/w/api.php", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:  # noqa: BLE001
        return []
    return [
        {
            "title": item.get("title", ""),
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(str(item.get('title', '')).replace(' ', '_'))}",
            "snippet": _strip_html(item.get("snippet", "")),
        }
        for item in (data.get("query", {}) or {}).get("search", [])
    ][:top_k]


def _ddg_decode(href: str) -> str:
    if "uddg=" in href:
        match = re.search(r"uddg=([^&]+)", href)
        if match:
            return urllib.parse.unquote(match.group(1))
    return href


async def _search_ddg(query: str, top_k: int) -> list[dict]:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }
    try:
        async with httpx.AsyncClient(
            timeout=25.0, headers=headers, proxy=proxy, follow_redirects=True
        ) as client:
            resp = await client.post("https://lite.duckduckgo.com/lite/", data={"q": query})
            html = resp.text
    except Exception:  # noqa: BLE001
        return []
    titles = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S)
    snippets = re.findall(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', html, re.S)
    out: list[dict] = []
    for index in range(min(top_k, len(titles))):
        href, title = titles[index]
        out.append({
            "title": _strip_html(title),
            "url": _ddg_decode(href),
            "snippet": _strip_html(snippets[index]) if index < len(snippets) else "",
        })
    return out


async def _search_web(query: str, top_k: int = 5) -> list[dict]:
    """Search through Tavily, Wikipedia, then DuckDuckGo fallbacks."""
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            return await _search_tavily(query, top_k, tavily_key)
        except Exception:  # noqa: BLE001
            pass
    results = await _search_wikipedia(query, top_k)
    if results:
        return results
    return await _search_ddg(query, top_k)


# ── fetch_url ─────────────────────────────────────────────────────────────


def _fetch_page_text(url: str, max_chars: int = 30000) -> str:
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=25.0, headers=headers, proxy=proxy, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            raw = resp.text
            without_noise = re.sub(
                r"<(script|style|nav|header|footer|aside|form|noscript)[^>]*>.*?</\1>",
                " ",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            text = re.sub(r"<[^>]+>", " ", without_noise)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
    except Exception as exc:  # noqa: BLE001
        return f"[fetch_url failed: {exc}]"


def _make_search_web_tool():
    async def handler(args: Any) -> dict:
        query = str((args or {}).get("query") or "").strip()
        top_k = int((args or {}).get("top_k") or 5)
        if not query:
            return {"content": [{"type": "text", "text": "NO_QUERY"}]}
        results = await _search_web(query, top_k)
        if not results:
            return {"content": [{"type": "text", "text": "WEB_SEARCH_UNAVAILABLE"}]}
        blocks = [
            _block(item.get("url", "?"), f"Title: {item.get('title', '')}\nSnippet: {item.get('snippet', '')}")
            for item in results
        ]
        return {"content": [{"type": "text", "text": "\n\n".join(blocks)}]}

    return handler


def _make_fetch_url_tool():
    async def handler(args: Any) -> dict:
        url = str((args or {}).get("url") or "").strip()
        if not url:
            return {"content": [{"type": "text", "text": "NO_URL"}]}
        text = await asyncio.to_thread(_fetch_page_text, url)
        return {"content": [{"type": "text", "text": _block(url, text)}]}

    return handler


async def sdk_scrape(url: str, options: LLMOptions, save_directory: Path) -> dict:
    """Collect alternative sources and return the collector-compatible result with provenance."""
    settings = get_settings()
    api_key = (options.api_key or settings.llm_api_key or "").strip()
    model = (options.model or settings.llm_model or "").strip()
    base_url = (options.base_url or settings.llm_base_url or "").rstrip("/")

    search_web = _make_search_web_tool()
    fetch_url = _make_fetch_url_tool()

    instructions = (
        "You are a web-scraping fallback agent. The primary scraper failed to fetch the "
        "given URL (network error, 404, blocked, or empty content). Reconstruct the page's "
        "content from alternative web sources. Use search_web to find alternative sources "
        "about the same subject, then fetch_url to read promising pages. Base your output "
        "ONLY on what you actually retrieved — do not invent facts, dates, or names. Put "
        "every URL you drew on into 'sources'."
    )
    if (options.language or "en-US").lower().startswith("en"):
        instructions += "\n" + ENGLISH_OUTPUT_INSTRUCTION
    prompt = f"""
Original URL (failed to scrape): {url}

Steps:
1. Infer the subject/title from the URL (or search_web for it).
2. Call search_web to find alternative sources covering the same subject.
3. Call fetch_url on the most relevant results to read full text.
4. Submit a structured result:
   - title: reconstructed page/section title
   - summary: short summary of the subject
   - content: a substantial plain-text body with the key facts (becomes the background corpus)
   - source_url: the single most representative alternative URL you used
   - sources: every URL you drew content from
   - source_excerpts: one entry per URL you drew on, in the form
     {{"url": "...", "title": "...", "text": "..."}} where "text" is the relevant
     VERBATIM passage you read from THAT url. This is what gives the pipeline a
     per-source section tree, so do not merge several URLs into one entry and do
     not leave "text" empty for a URL you actually used.
Do not fabricate. If nothing usable is found, put a brief honest note in content.
""".strip()

    output = await asyncio.wait_for(
        run_structured_agent(
            api_key=api_key,
            model=model,
            base_url=base_url,
            name="scrape_fallback",
            instructions=instructions,
            output_type=SDKScrapeResult,
            prompt=prompt,
            max_turns=12,
            extra_tools=[
                (
                    "search_web",
                    "Search the web for alternative sources about a topic.",
                    SEARCH_WEB_SCHEMA,
                    search_web,
                ),
                (
                    "fetch_url",
                    "Read the full plain-text content of a URL.",
                    FETCH_URL_SCHEMA,
                    fetch_url,
                ),
            ],
        ),
        timeout=settings.agent_sdk_timeout_seconds,
    )

    payload = output.model_dump()
    source_url = str(payload.get("source_url") or url).strip()

    source_number = 1
    if save_directory is not None:
        match = re.match(r"(\d+)", save_directory.name)
        if match:
            source_number = int(match.group(1))
    sections = _sections_from_excerpts(payload.get("source_excerpts") or [], source_number)

    if save_directory is not None:
        save_directory.mkdir(parents=True, exist_ok=True)
        content = str(payload.get("content") or "").strip()
        if content:
            temporary = save_directory / "content.txt.tmp"
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(save_directory / "content.txt")

    return {
        "title": str(payload.get("title") or "").strip(),
        "url": source_url,
        "domain": _domain(source_url) or _domain(url),
        "summary": str(payload.get("summary") or "").strip(),
        "content": str(payload.get("content") or "").strip(),
        "sections": sections,
        "sdk_fallback": True,
        "sources": payload.get("sources") or [],
    }
