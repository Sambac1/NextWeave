from __future__ import annotations

import asyncio
import html as html_module
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

import httpx
from app.knowledge.sentences import complete_excerpt


WIKI_URL = re.compile(
    r"https?://(?P<language>[a-z\-]+)\.wikipedia\.org/wiki/(?P<title>.+?)(?:#.*)?$",
    re.IGNORECASE,
)
UNSAFE_PATH = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def filesystem_name(value: str) -> str:
    return UNSAFE_PATH.sub("_", value).strip(". ") or "source"


def _write_source_body(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


_MIN_USABLE_CONTENT_CHARS = 200

# Wikimedia requests identify the client and a project contact URL.
USER_AGENT = os.getenv("WIKIMEDIA_USER_AGENT", "").strip() or (
    "NextWeave (https://everything-is-possible-demo.readytetra5.chatgpt.site/)"
)

_MIN_SECTIONS_FOR_FALLBACK = 2

SECTION_CHARS = 15_000


def _failure_reason(result: dict) -> str:
    """Why a scrape result is unusable, for the caller's error message."""
    if not isinstance(result, dict):
        return "not a result object"
    if result.get("error"):
        return str(result["error"])
    content = str(result.get("content") or "").strip()
    if not content:
        return "no content"
    return f"content too short ({len(content)} chars, need {_MIN_USABLE_CONTENT_CHARS})"


def _mark_failed(result: dict, error: str) -> dict:
    """Ensure a failed result carries an error, so downstream reads it as one.

    A result that already states why it failed keeps its own message -- it is
    the more precise one. Only a result that failed *silently* (unusable
    content, no error field) gets one attached.
    """
    if not isinstance(result, dict):
        return {"url": "", "error": error}
    if result.get("error"):
        return result
    merged = dict(result)
    merged["error"] = error
    return merged


def _is_usable(result: dict) -> bool:
    """True when a scrape result is good enough to feed the downstream pipeline."""
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    content = str(result.get("content") or "").strip()
    if len(content) >= _MIN_USABLE_CONTENT_CHARS:
        return True
    sections = [item for item in (result.get("sections") or []) if isinstance(item, dict)]
    sections = [
        item for item in sections
        if len(str(item.get("content") or "").strip()) >= _MIN_USABLE_CONTENT_CHARS
    ]
    return len(sections) >= _MIN_SECTIONS_FOR_FALLBACK


def split_sections(content: str, source_number: int, *, max_chars: int | None = SECTION_CHARS) -> list[dict]:
    """Preserve source structure while bounding individual LLM inputs."""
    heading = re.compile(r"(?m)^\s*(={2,6})\s*(.+?)\s*\1\s*$")
    matches = list(heading.finditer(content))
    raw_sections: list[tuple[str, str, str, list[tuple[int, str]]]] = []
    source_root = f"source_{source_number:02d}"
    if matches:
        lead = content[:matches[0].start()].strip()
        if lead:
            raw_sections.append(("Lead", f"{source_root}/Lead", lead, [(2, "Lead")]))
        hierarchy: list[tuple[int, str]] = []
        for index, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            while hierarchy and hierarchy[-1][0] >= level:
                hierarchy.pop()
            hierarchy.append((level, title))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            if body:
                path = "/".join([source_root, *(item[1] for item in hierarchy)])
                raw_sections.append((title, path, body, hierarchy.copy()))
    else:
        raw_sections.append(("Document", f"{source_root}/Document", content.strip(), [(2, "Document")]))
    sections = []
    serial = 0
    for title, path, body, headings in raw_sections:
        chunk_size = max_chars if max_chars is not None else max(1, len(body))
        for offset in range(0, len(body), chunk_size):
            chunk = body[offset:offset + chunk_size].strip()
            if not chunk:
                continue
            serial += 1
            suffix = f" (part {offset // chunk_size + 1})" if len(body) > chunk_size else ""
            sections.append({
                "section_id": f"source_{source_number:02d}_section_{serial:03d}",
                "title": f"{title}{suffix}",
                "path": f"{path}{suffix}",
                "chapter": "/".join(path.split("/")[:2]),
                "headings": headings,
                "content": chunk,
            })
    return sections


class SourceCollector:
    """Collects Wikipedia pages through its API and other pages through HTTP."""

    def __init__(self) -> None:
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        kwargs: dict = {
            "timeout": 30,
            "follow_redirects": True,
            "headers": {"User-Agent": USER_AGENT},
        }
        if proxy:
            kwargs["proxy"] = proxy
        self.http = httpx.AsyncClient(**kwargs)

    async def close(self) -> None:
        await self.http.aclose()

    async def fetch(self, url: str, save_directory: Path) -> dict:
        match = WIKI_URL.match(url.strip())
        if match:
            return await self._fetch_wikipedia(
                url,
                match.group("language"),
                unquote(match.group("title")).replace("_", " "),
                save_directory,
            )
        return await self._fetch_web_page(url, save_directory)

    async def fetch_with_fallback(
        self,
        url: str,
        save_directory: Path,
        llm_options=None,
    ) -> dict:
        """Fetch a URL; fall back to the Agent SDK when the primary scraper fails."""
        result = await self.fetch(url, save_directory)
        if _is_usable(result):
            return result
        if llm_options is not None:
            try:
                from app.knowledge.sdk_scrape import sdk_scrape

                fallback = await sdk_scrape(url, llm_options, save_directory)
                if _is_usable(fallback):
                    return fallback
            except Exception as exc:  # noqa: BLE001
                # Report both failures: the primary one explains why the
                # fallback was attempted, and the SDK error why it did not help.
                merged = dict(result)
                merged["error"] = (
                    f"{result.get('error') or 'primary scrape failed'}; "
                    f"SDK fallback error: {exc}"
                )
                return merged
        return _mark_failed(
            result,
            f"primary scrape failed ({_failure_reason(result)})"
            + ("" if llm_options is not None else "; no LLM configuration for SDK fallback"),
        )

    async def fetch_all(self, urls: list[str], save_directory: Path, llm_options=None) -> list[dict]:
        save_directory.mkdir(parents=True, exist_ok=True)
        tasks = [
            self.fetch_with_fallback(url, save_directory / f"{index + 1:02d}_source", llm_options)
            for index, url in enumerate(urls)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            result if isinstance(result, dict) else {"url": urls[index], "error": str(result)}
            for index, result in enumerate(results)
        ]

    async def _fetch_wikipedia(
        self,
        url: str,
        language: str,
        title: str,
        save_directory: Path,
    ) -> dict:
        try:
            response = await self.http.get(
                f"https://{language}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts|categories|info",
                    "titles": title,
                    "redirects": "1",
                    "explaintext": "1",
                    "inprop": "url",
                    "cllimit": "max",
                    "format": "json",
                    "formatversion": "2",
                },
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", [])
            page = pages[0] if pages else {}
            if page.get("missing") or not page.get("extract"):
                return {"url": url, "error": f"Wikipedia page not found: {title}"}
            content = str(page.get("extract") or "")
            summary = complete_excerpt(content.split("\n\n", 1)[0], 3000)
            save_directory.mkdir(parents=True, exist_ok=True)
            _write_source_body(save_directory / "content.txt", content)
            return {
                "title": str(page.get("title") or title),
                "url": str(page.get("fullurl") or url),
                "domain": f"{language}.wikipedia.org",
                "page_id": page.get("pageid"), "revision_id": page.get("lastrevid"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "content": content,
                "sections": split_sections(content, int(save_directory.name.split("_")[0])),
                "categories": [
                    str(item.get("title", "")).split(":", 1)[-1]
                    for item in page.get("categories", [])
                    if item.get("title")
                ],
            }
        except httpx.TimeoutException:
            return {"url": url, "error": "Timeout while calling Wikipedia API"}
        except httpx.HTTPStatusError as exc:
            return {"url": url, "error": f"Wikipedia API HTTP {exc.response.status_code}"}
        except Exception as exc:
            return {"url": url, "error": f"Wikipedia API error: {exc}"}

    async def _fetch_web_page(self, url: str, save_directory: Path) -> dict:
        try:
            response = await self.http.get(url)
            response.raise_for_status()
            raw = response.text
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
            title = html_module.unescape(title_match.group(1).strip()) if title_match else url
            without_noise = re.sub(
                r"<(script|style|nav|header|footer|aside|form|noscript)[^>]*>.*?</\1>",
                " ",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            text = html_module.unescape(re.sub(r"<[^>]+>", " ", without_noise))
            text = re.sub(r"\s+", " ", text).strip()[:100_000]
            save_directory.mkdir(parents=True, exist_ok=True)
            _write_source_body(save_directory / "content.txt", text)
            return {
                "title": title,
                "url": str(response.url),
                "domain": urlparse(str(response.url)).netloc,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "summary": complete_excerpt(text, 1200),
                "content": text,
                "sections": split_sections(text, int(save_directory.name.split("_")[0])),
            }
        except httpx.TimeoutException:
            return {"url": url, "error": "Timeout while fetching URL"}
        except httpx.HTTPStatusError as exc:
            return {"url": url, "error": f"HTTP {exc.response.status_code}"}
        except Exception as exc:
            return {"url": url, "error": str(exc)}
