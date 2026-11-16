"""Read-only import of a Polymarket contract and its bounded Wiki evidence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from app.knowledge.scraper import USER_AGENT

from .dynamics import outcome_window
from app.knowledge.wikipedia import discover_wikipedia, utc_timestamp

GAMMA = "https://gamma-api.polymarket.com"
SLUG = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,249}")


class MarketSelectionRequired(ValueError):
    """An event URL points to multiple independently resolved contracts."""

    def __init__(self, choices: list[dict]):
        self.choices = choices
        examples = " · ".join(f"{row['question']} {row['url']}" for row in choices[:5])
        super().__init__(f"This event contains multiple contracts. Choose a specific market URL. {examples}")


def _location(market_url: str) -> tuple[str, str, str | None]:
    url = urlsplit(market_url.strip())
    parts = url.path.strip("/").split("/")
    if (
        url.scheme != "https"
        or url.netloc not in {"polymarket.com", "www.polymarket.com"}
        or len(parts) not in {2, 3}
        or parts[0] not in {"event", "market"}
        or not all(SLUG.fullmatch(p) for p in parts[1:])
    ):
        raise ValueError("Enter an HTTPS Polymarket event or contract URL.")
    kind, slug = parts[:2]
    if kind == "market" and len(parts) != 2:
        raise ValueError("A contract URL must contain a single market identifier.")
    selected = parse_qs(url.query).get("marketSlug", [])
    if len(selected) > 1 or (selected and not SLUG.fullmatch(selected[0])):
        raise ValueError("Invalid Polymarket submarket identifier.")
    explicit = parts[2] if len(parts) == 3 else None
    if explicit and selected and explicit != selected[0]:
        raise ValueError("The Polymarket URL contains conflicting submarket identifiers.")
    return kind, slug, explicit or (selected[0] if selected else None)


def _market_url(slug: str, event_slug: str | None) -> str:
    if event_slug:
        return f"https://polymarket.com/event/{event_slug}?" + urlencode({"marketSlug": slug})
    return f"https://polymarket.com/market/{slug}"


async def _gamma(client: httpx.AsyncClient, kind: str, slug: str) -> dict:
    try:
        response = await client.get(f"{GAMMA}/{kind}s/slug/{slug}", follow_redirects=False)
        if response.status_code == 404:
            raise ValueError("Polymarket could not find the event or market.")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ValueError("The Polymarket API is temporarily unavailable. Try again later.") from exc
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Polymarket returned an unrecognized market record.")
    return result


def _select_market(event: dict, slug: str, selected: str | None) -> dict:
    markets = [
        market for market in (event.get("markets") or [])
        if isinstance(market, dict) and SLUG.fullmatch(str(market.get("slug") or ""))
    ]
    if selected:
        market = next((market for market in markets if market["slug"] == selected), None)
        if market is None:
            raise ValueError("The selected submarket does not belong to this Polymarket event.")
        return market
    if len(markets) == 1:
        return markets[0]
    if not markets:
        raise ValueError("This Polymarket event has no importable markets.")
    raise MarketSelectionRequired([
        {"slug": market["slug"], "question": str(market.get("question") or market["slug"]),
         "url": _market_url(market["slug"], slug)}
        for market in markets[:30]
    ])


def _rules_timestamp(market: dict, cutoff: datetime, capture: datetime | None) -> datetime:
    if capture is not None:
        if capture > cutoff:
            raise ValueError("The market archive was captured after the forecast cutoff.")
        return capture
    if not market.get("createdAt") or not market.get("updatedAt"):
        raise ValueError("The rules could not be verified at the forecast cutoff. Provide a timestamped market archive.")
    created = utc_timestamp(market["createdAt"], "createdAt")
    updated = utc_timestamp(market["updatedAt"], "updatedAt")
    if created > cutoff or updated > cutoff or updated < created:
        raise ValueError("The current market version does not apply at this historical cutoff. Provide an archived copy of the rules.")
    return updated


def _contract(market: dict, cutoff: datetime, event_slug: str | None) -> dict:
    question = str(market.get("question") or "").strip()
    rules = str(market.get("description") or "").strip()
    market_id = str(market.get("id") or "").strip()
    condition_id = str(market.get("conditionId") or "").strip()
    slug = str(market.get("slug") or "")
    outcomes = market.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except ValueError as exc:
            raise ValueError("Polymarket returned unrecognized outcome options.") from exc
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != 2
        or any(not isinstance(value, str) for value in outcomes)
        or {value.strip().casefold() for value in outcomes} != {"yes", "no"}
    ):
        raise ValueError("Only contracts with Yes and No outcomes are supported.")
    if not question or not rules or not market_id or not condition_id or not SLUG.fullmatch(slug):
        raise ValueError("The market lacks a clear question, rules, or unique identifier.")
    if market.get("closed") is True:
        raise ValueError("The market was closed in the selected snapshot.")
    deadline = utc_timestamp(market.get("endDate"), "market endDate")
    if deadline <= cutoff:
        raise ValueError("The forecast cutoff must precede the published market end time.")
    return {
        "group_id": event_slug or slug,
        "condition_id": condition_id,
        "market_id": market_id,
        "market_url": _market_url(slug, event_slug),
        "question": question,
        "rules": rules,
        "outcomes": ["Yes", "No"],
        "resolution_source": str(market.get("resolutionSource") or ""),
        "as_of": cutoff.isoformat(),
        "outcome_deadline": deadline.isoformat(),
        "deadline_basis": "market_end_date_requires_rule_review",
        **outcome_window(rules, market),
    }


def _event_slug(market: dict) -> str | None:
    events = market.get("events") or []
    for event in events:
        if isinstance(event, dict) and SLUG.fullmatch(str(event.get("slug") or "")):
            return event["slug"]
    return None


async def resolve_market_url(client: httpx.AsyncClient, market_url: str) -> tuple[dict, str | None]:
    kind, slug, selected = _location(market_url)
    if kind == "event":
        event = await _gamma(client, "event", slug)
        return _select_market(event, slug, selected), slug
    market = await _gamma(client, "market", slug)
    if market.get("slug") != slug:
        raise ValueError("Polymarket returned a different market.")
    return market, _event_slug(market)


async def resolve_contract(
    client: httpx.AsyncClient,
    market_url: str,
    *,
    as_of: datetime | None = None,
    archived_market: dict | None = None,
) -> dict:
    """Read the frozen rule record before acquiring any evidence."""
    kind, slug, selected = _location(market_url)
    cutoff = utc_timestamp(as_of, "as_of") if as_of is not None else None
    if cutoff and cutoff > datetime.now(timezone.utc):
        raise ValueError("The forecast cutoff cannot be in the future.")
    event_slug = slug if kind == "event" else None
    capture = None
    if archived_market is not None:
        if cutoff is None or not isinstance(archived_market.get("market"), dict):
            raise ValueError("A historical market archive requires as_of, captured_at, and the market snapshot.")
        capture = utc_timestamp(archived_market.get("captured_at"), "captured_at")
        market = archived_market["market"]
        archived_event = archived_market.get("event_slug") or _event_slug(market)
        if archived_event is not None and not SLUG.fullmatch(str(archived_event)):
            raise ValueError("The archived event identifier is invalid.")
        if (event_slug and archived_event != event_slug) or (
            (selected or (slug if kind == "market" else None))
            not in {None, market.get("slug")}
        ):
            raise ValueError("The market archive does not match the Polymarket URL.")
        event_slug = archived_event
    else:
        market, event_slug = await resolve_market_url(client, market_url)
    if kind == "market" and not event_slug:
        event_slug = _event_slug(market)
    live = cutoff is None
    if live:
        capture = cutoff = datetime.now(timezone.utc)
    assert cutoff is not None
    rule_time = _rules_timestamp(market, cutoff, capture)
    contract = _contract(market, cutoff, event_slug)
    return contract | {"market_rules_at": rule_time.isoformat()}


async def import_market(
    market_url: str,
    *,
    as_of: datetime | None,
    seed_wiki_urls: list[str],
    client: httpx.AsyncClient | None = None,
    max_pages: int = 10,
    archived_market: dict | None = None,
) -> dict:
    """Import a contract. Historical rules require a cutoff-compatible snapshot."""
    if client is None:
        async with httpx.AsyncClient(timeout=12, headers={"User-Agent": USER_AGENT}) as session:
            return await import_market(
                market_url, as_of=as_of, seed_wiki_urls=seed_wiki_urls, client=session,
                max_pages=max_pages, archived_market=archived_market,
            )
    if as_of is not None and not seed_wiki_urls:
        raise ValueError("Provide at least one Wikipedia seed page for a historical cutoff.")
    resolved = await resolve_contract(client, market_url, as_of=as_of, archived_market=archived_market)
    contract = {key: value for key, value in resolved.items() if key != "market_rules_at"}
    cutoff = utc_timestamp(contract["as_of"], "as_of")
    rule_time = resolved["market_rules_at"]
    evidence, manifest = await discover_wikipedia(
        client, question=contract["question"], rules=contract["rules"], as_of=cutoff,
        seed_urls=seed_wiki_urls, max_pages=max_pages, live_search=as_of is None,
    )
    case_identity = {
        "contract": contract,
        "market_rules_at": rule_time,
        "evidence": [
            {key: row[key] for key in ("url", "page_id", "revision_id", "revision_at", "sha256",
                                       "revision_sha256", "is_seed", "discovered_from")}
            | ({"title_match_score": row["title_match_score"]} if "title_match_score" in row else {})
            for row in evidence
        ],
    }
    identity_bytes = json.dumps(case_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return contract | {
        "case_id": "market_" + hashlib.sha256(identity_bytes).hexdigest()[:24],
        "market_rules_at": rule_time,
        "evidence": evidence,
        "retrieval_manifest": manifest,
    }
