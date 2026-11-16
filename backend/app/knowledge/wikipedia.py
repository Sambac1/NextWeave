"""Cutoff-safe Wikipedia discovery and source snapshots for event forecasts."""
from __future__ import annotations

import asyncio
import hashlib
import html
import re
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlsplit

import httpx

from app.knowledge.scraper import USER_AGENT
from app.knowledge.sentences import sentence_records
from app.knowledge.wiki_titles import TimelineTitles

WIKI_HOST = re.compile(r"[a-z][a-z0-9-]{0,23}\.wikipedia\.org")
LINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")
COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
REFERENCES = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*>.*?</ref\s*>", re.I | re.S)
TEMPLATE_BOUNDARY = re.compile(r"\{\{|\}\}")
DETAIL_ARTICLE = re.compile(r"\{\{\s*(?:main|further(?: information)?|details)\s*\|([^{}]+)\}\}", re.I)
REDIRECT = re.compile(r"^\s*#redirect\s*\[\[([^\]\n]+)\]\]", re.I)
STOPWORDS = set(
    "about after against and are before between can does for from have how into may "
    "the their there this through was were what when where which who will with would yes no".split()
)
MARKET_ODDS_CUE = re.compile(
    r"\b(?:odds|(?:implied\s+)?probabilit(?:y|ies)|priced|trading|price|shares|contracts)\b"
    r"|盘口|赔率|隐含概率|交易价格",
    re.I,
)
MARKET_ODDS_VALUE = re.compile(r"(?:\d+(?:\.\d+)?\s*(?:%|¢|cents\b|percent\b)|\b0\.\d+\b)", re.I)


class WikiUnavailable(ValueError):
    """The requested page does not provide usable evidence for the cutoff."""


def utc_timestamp(value: datetime | str, field: str) -> datetime:
    try:
        instant = datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO 8601 date and time with a timezone.") from exc
    if not isinstance(instant, datetime) or instant.utcoffset() is None:
        raise ValueError(f"{field} must be an ISO 8601 date and time with a timezone.")
    return instant.astimezone(timezone.utc)


def _wiki_location(url: str) -> tuple[str, str]:
    parsed = urlsplit(url.strip())
    host = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or not WIKI_HOST.fullmatch(host)
        or parsed.netloc != host
        or not parsed.path.startswith("/wiki/")
    ):
        raise ValueError("Use a Wikipedia seed URL in the form https://<language>.wikipedia.org/wiki/<article>.")
    title = unquote(parsed.path[len("/wiki/"):]).replace("_", " ").strip()
    if not title or len(title) > 255 or "|" in title or re.search(r"[\x00-\x1f]", title):
        raise ValueError("Invalid Wikipedia article title.")
    return host, title


def _url(host: str, title: str) -> str:
    return f"https://{host}/wiki/{quote(title.replace(' ', '_'), safe='()')}"


def _strip_templates(raw: str, detail_titles: list[str] | None = None) -> str:
    """Discard transclusions rather than expand a potentially newer template."""
    parts = []
    depth = 0
    start = 0
    template_start = 0
    for match in TEMPLATE_BOUNDARY.finditer(raw):
        if match[0] == "{{":
            if depth == 0:
                parts.append(raw[start:match.start()])
                template_start = match.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                start = match.end()
                if detail_titles is not None and (detail := DETAIL_ARTICLE.fullmatch(raw[template_start:start])):
                    detail_titles.extend(title for param in detail[1].split("|")[:3]
                                         if (title := param.split("#", 1)[0].strip())
                                         and len(title) <= 255 and not re.search(r"[:=\[\]<>\n]", title))
    if not depth:
        parts.append(raw[start:])
    return "".join(parts)


def _article_markup(raw: str, detail_titles: list[str] | None = None) -> str:
    raw = REFERENCES.sub("", COMMENT.sub("", raw))
    # Reference sections are metadata; their outbound links do not drive selection.
    raw = re.split(
        r"(?im)^\s*={2,6}\s*(?:references|notes|further reading|external links|see also|参考资料|参考文献|外部链接)\s*={2,6}\s*$",
        raw, maxsplit=1,
    )[0]
    return _strip_templates(raw, detail_titles)


def _titles(markup: str) -> list[str]:
    result = []
    for match in LINK.finditer(markup):
        title = match[1].split("|", 1)[0].split("#", 1)[0].strip().replace("_", " ")
        # Ignore namespaces, interwiki links, and anchors.
        if title and ":" not in title and len(title) <= 255:
            result.append(title)
    return list(dict.fromkeys(result))


def _plain_text(markup: str) -> str:
    def link_text(match: re.Match[str]) -> str:
        target, _, label = match[1].partition("|")
        if re.match(r"^(?:file|image|category|template):", target.lstrip(":"), re.I):
            return ""
        return label or target.split("#", 1)[0]

    text = LINK.sub(link_text, markup)
    text = re.sub(r"\[(?:https?://)[^\s\]]+(?:\s+([^\]]+))?\]", r"\1", text)
    text = re.sub(r"<[^>\n]+>", "", text).replace("'''", "").replace("''", "")
    return html.unescape(text).strip()


def _without_direct_market_odds(text: str) -> tuple[str, int]:
    """Remove explicit Polymarket quote sentences from the evidence-only arm."""
    lines = []
    removed = 0
    for line in text.split("\n"):
        kept = []
        start = 0
        endings = [row["end"] for row in sentence_records(line)]
        for end in dict.fromkeys([*endings, len(line)]):
            span = line[start:end]
            if ("polymarket" in span.casefold() and MARKET_ODDS_CUE.search(span)
                    and MARKET_ODDS_VALUE.search(span)):
                removed += 1
            else:
                kept.append(span)
            start = end
        lines.append("".join(kept))
    return "\n".join(lines).strip(), removed


def _terms(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[^\W_]+", text.casefold())
        if token not in STOPWORDS and len(token) > 2
    }


async def _wiki_json(client: httpx.AsyncClient, host: str, params: dict) -> dict:
    response = await client.get(
        f"https://{host}/w/api.php", params=params | {"format": "json", "formatversion": 2},
        headers={"User-Agent": USER_AGENT}, follow_redirects=False,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or result.get("error"):
        raise WikiUnavailable("MediaWiki returned an error or unavailable content.")
    return result


async def fetch_revision(client: httpx.AsyncClient, url: str, as_of: datetime) -> dict:
    """Read the latest revision at or before cutoff, using its own wikitext links."""
    host, title = _wiki_location(url)
    cutoff = utc_timestamp(as_of, "as_of")
    visited = set()
    for _ in range(3):
        if title in visited:
            break
        visited.add(title)
        result = await _wiki_json(client, host, {
            "action": "query", "prop": "revisions", "titles": title,
            "rvprop": "ids|timestamp|content", "rvslots": "main", "rvlimit": 1,
            "rvdir": "older", "rvstart": cutoff.isoformat(),
        })
        pages = result.get("query", {}).get("pages", [])
        page = pages[0] if pages else {}
        revisions = page.get("revisions", [])
        if not revisions or page.get("ns") != 0:
            raise WikiUnavailable(f"No public version of the Wikipedia article {title} was available before the cutoff.")
        revision = revisions[0]
        revision_at = utc_timestamp(revision.get("timestamp"), "revision_at")
        if revision_at > cutoff:
            raise WikiUnavailable(f"The Wikipedia article {title} returned a revision after the cutoff.")
        raw = revision.get("slots", {}).get("main", {}).get("content")
        if not isinstance(raw, str) or not raw.strip():
            raise WikiUnavailable(f"The pre-cutoff revision of {title} has no article text.")
        redirect = REDIRECT.match(COMMENT.sub("", raw))
        if redirect:
            targets = _titles(redirect[0])
            if not targets:
                break
            title = targets[0]
            continue
        revision_id = revision.get("revid")
        page_id = page.get("pageid")
        if type(page_id) is not int or type(revision_id) is not int or min(page_id, revision_id) <= 0:
            raise WikiUnavailable(f"The Wikipedia article {title} has no traceable revision ID.")
        detail_titles: list[str] = []
        markup = _article_markup(raw, detail_titles)
        text, removed = _without_direct_market_odds(_plain_text(markup))
        if not text:
            raise WikiUnavailable(f"The Wikipedia article {title} has no usable static article text.")
        actual_title = page["title"]
        return {
            "url": _url(host, actual_title),
            "revision_url": f"https://{host}/w/index.php?oldid={revision_id}",
            "title": actual_title,
            "page_id": page_id,
            "revision_id": revision_id,
            "revision_at": revision_at.isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "revision_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "redacted_market_odds": removed,
            "_links": [_url(host, target) for target in _titles(markup)],
            "_detail_links": list(dict.fromkeys(_url(host, target) for target in detail_titles)),
        }
    raise WikiUnavailable(f"The historical redirect for the Wikipedia article {title} could not be resolved safely.")


async def discover_wikipedia(
    client: httpx.AsyncClient,
    *,
    question: str,
    rules: str,
    as_of: datetime,
    seed_urls: list[str],
    max_pages: int = 10,
    live_search: bool = False,
    title_discovery: bool = False,
    include_search_with_seeds: bool = False,
    max_link_attempts: int | None = None,
    exclude_urls: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Search for the timeline, screen by title, then freeze the selected revisions."""
    if not 1 <= max_pages <= 12 or len(seed_urls) > max_pages:
        raise ValueError("The evidence bundle supports up to 12 Wikipedia pages. Use at most as many seeds as the page limit.")
    cutoff = utc_timestamp(as_of, "as_of")
    topic = TimelineTitles(question, cutoff.year)
    excluded = exclude_urls or set()
    seeds = list(dict.fromkeys(_url(*_wiki_location(url)) for url in seed_urls))
    search_titles = not seeds or include_search_with_seeds
    queries = topic.queries() if search_titles else []
    if not seeds and not (live_search or title_discovery):
        raise ValueError("Provide at least one Wikipedia seed page for a historical cutoff.")
    if not seeds and not queries:
        raise ValueError("Wikipedia search terms could not be derived from the question. Provide seed pages.")

    slots = asyncio.Semaphore(4)
    evidence, manifest = [], []
    pages_seen, fingerprints = set(), set()
    rate_limit = None

    async def collect(url):
        async with slots:
            try:
                return await fetch_revision(client, url, cutoff), ""
            except WikiUnavailable as exc:
                return None, str(exc)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    raise
                return None, "The Wikipedia API is temporarily unavailable or returned unusable content."
            except (httpx.HTTPError, ValueError):
                return None, "The Wikipedia API is temporarily unavailable or returned unusable content."

    async def freeze(rows):
        nonlocal rate_limit
        responses = await asyncio.gather(*(collect(row["url"]) for row in rows), return_exceptions=True)
        for row, value in zip(rows, responses):
            row["revision_attempted"] = True
            if isinstance(value, httpx.HTTPStatusError) and value.response.status_code == 429:
                rate_limit = value
                row["reason"] = "Wikipedia Rate Limit"
                continue
            if isinstance(value, BaseException):
                raise value
            result, error = value
            if result is None:
                row["reason"] = error
                continue
            row.update(revision_id=result["revision_id"], canonical_title=result["title"])
            if result["url"] in excluded or result["url"] in pages_seen or result["sha256"] in fingerprints:
                row["reason"] = "Duplicate or excluded revision"
            elif row["discovered_from"] == "wiki_search" and not topic.assess(result["title"])[0]:
                row["reason"] = "Redirect resolved to a title outside the event"
            else:
                result.update(discovered_from=row["discovered_from"], is_seed=row["is_seed"])
                if row["discovered_from"] == "wiki_search":
                    result["title_match_score"] = row["title_score"]
                evidence.append(result)
                fingerprints.add(result["sha256"])
                pages_seen.add(result["url"])
                row.update(selected=True, reason="Cutoff-safe revision available")

    seed_rows = [{"stage": "revision", "url": url, "discovered_from": "user_seed",
                  "is_seed": True, "selected": False} for url in seeds if url not in excluded]
    manifest.extend(seed_rows)
    await freeze(seed_rows)
    allowance = max_pages if max_link_attempts is None else max(0, max_link_attempts)
    if search_titles and len(evidence) < max_pages:
        results = await asyncio.gather(*(_wiki_json(client, "en.wikipedia.org", {
            "action": "query", "list": "search", "srsearch": query,
            "srnamespace": 0, "srlimit": 6, "srprop": "",
        }) for query in queries), return_exceptions=True)
        candidates = {}
        search_error = None
        for query, result in zip(queries, results):
            if isinstance(result, (httpx.HTTPError, WikiUnavailable)):
                search_error = result
                manifest.append({"stage": "search", "query": query, "titles": [], "status": "failed"})
                continue
            if isinstance(result, BaseException):
                raise result
            titles = [row["title"] for row in result.get("query", {}).get("search", [])
                      if isinstance(row.get("title"), str)]
            manifest.append({"stage": "search", "query": query, "titles": titles, "status": "completed"})
            for rank, title in enumerate(titles):
                url = _url("en.wikipedia.org", title)
                if url in excluded or url in seeds or url in pages_seen:
                    continue
                score, reason = topic.assess(title)
                row = candidates.setdefault(url.casefold(), {
                    "stage": "title_review", "url": url, "title": title, "discovered_from": "wiki_search",
                    "is_seed": True, "selected": False, "title_score": score, "title_reason": reason,
                    "reason": reason, "matched_queries": [], "best_rank": rank,
                })
                row["matched_queries"].append(query)
                row["best_rank"] = min(row["best_rank"], rank)
        manifest.extend(candidates.values())
        ordered = sorted((row for row in candidates.values() if row["title_score"]),
                         key=lambda row: (-row["title_score"], -len(row["matched_queries"]), row["best_rank"]))
        for row in ordered:
            row["reason"] = "Ranked below the page or revision budget"
        # Fetch only titles that passed review. Refill unused slots after missing revisions or redirects.
        budget = min(24, max_pages - len(evidence) + allowance)
        cursor = 0
        while cursor < min(budget, len(ordered)) and len(evidence) < max_pages:
            end = min(cursor + max_pages - len(evidence), budget)
            await freeze(ordered[cursor:end])
            cursor = end
        if not evidence and search_error and all(isinstance(result, BaseException) for result in results):
            raise search_error
    elif not search_titles and len(evidence) < max_pages:
        # Historical imports with explicit seeds can use their dated one-hop links.
        relevant = {}
        question_terms, rule_terms = _terms(question), _terms(rules)
        for seed in evidence:
            for url in dict.fromkeys([*seed["_detail_links"], *seed["_links"]]):
                if url in excluded or url in seeds or url in pages_seen:
                    continue
                words = _terms(_wiki_location(url)[1])
                score = 3 * len(words & question_terms) + len(words & rule_terms)
                if score:
                    relevant.setdefault(url, (score, seed["revision_url"]))
        ordered = sorted(relevant.items(), key=lambda row: (-row[1][0], row[0]))
        rows = [{"stage": "revision", "url": url, "discovered_from": origin,
                 "is_seed": False, "selected": False}
                for url, (_, origin) in ordered[:min(allowance, max_pages - len(evidence))]]
        manifest.extend(rows)
        await freeze(rows)

    if not evidence:
        if rate_limit is not None:
            raise rate_limit
        raise ValueError("No strongly related Wikipedia title had a usable revision at the cutoff. Try a Wikipedia seed page.")
    for row in evidence:
        row.pop("_links")
        row.pop("_detail_links")
    return evidence, manifest
