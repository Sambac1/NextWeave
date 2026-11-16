"""Collect Polymarket cohorts and keep market outcomes outside forecast inputs."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from app.knowledge.scraper import USER_AGENT
from app.storage.json_store import read_json, write_json

from app.knowledge.wikipedia import discover_wikipedia, utc_timestamp
from .markets import GAMMA, _contract, _event_slug, _gamma, resolve_market_url
from .runner import frozen_sources

POLITICS = re.compile(
    r"\b(?:politics|political|election|electoral|presiden\w*|senate|congress|parliament|minister|"
    r"governor|mayor|referendum|cabinet|legislature|geopolitic\w*|diploma\w*|"
    r"ceasefire|invasion|military|sanction\w*|nuclear deal|government|impeach\w*|"
    r"tariff\w*|veto|campaign|primary election|white house|democrat\w*|republican\w*)\b", re.I
)
NON_POLITICS = re.compile(r"\b(?:sports|esports|crypto|weather|entertainment)\b", re.I)
_capture_lock = asyncio.Lock()
HISTORICAL_READY_VERSION = "published-cutoff-v2"


class CohortWikiClient:
    """Serialize cohort requests and honor Wikimedia's Retry-After signal."""
    def __init__(self, client):
        self.client = client
        self.lock = asyncio.Lock()
        self.next_request = 0.0

    async def get(self, url, **kwargs):
        async with self.lock:
            loop = asyncio.get_running_loop()
            for _ in range(2):
                await asyncio.sleep(max(0, self.next_request - loop.time()))
                response = await self.client.get(url, **kwargs)
                self.next_request = loop.time() + 1.2
                if response.status_code != 429:
                    return response
                raw = response.headers.get("Retry-After", "")
                try:
                    delay = float(raw)
                except ValueError:
                    try:
                        delay = (parsedate_to_datetime(raw) - _now()).total_seconds()
                    except (ValueError, TypeError):
                        delay = 20
                self.next_request = loop.time() + max(5, delay)
                if delay > 60:
                    return response
            return response


def _now():
    return datetime.now(timezone.utc)


def _time(value):
    try:
        return utc_timestamp(value, "timestamp")
    except ValueError:
        return None


def _list(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return value if isinstance(value, list) else []


def _political(market):
    if market.get("_politics_filter") is True:
        return True
    groups = [market, *(row for row in market.get("events") or [] if isinstance(row, dict))]
    categories = " ".join(str(row.get(key) or "") for row in groups for key in ("category", "subcategory"))
    tags = " ".join(str(tag.get("label") or tag.get("slug") or "") for row in groups
                    for tag in row.get("tags") or [] if isinstance(tag, dict))
    return not NON_POLITICS.search(categories) and bool(POLITICS.search(f"{categories} {tags} {market.get('question', '')}"))


def _market_key(market):
    return str(market.get("conditionId") or "")


def _valid_binary(market):
    return sorted(str(value).strip().lower() for value in _list(market.get("outcomes"))) == ["no", "yes"]


def _prices(market):
    try:
        values = [float(value) for value in _list(market.get("outcomePrices"))]
    except (ValueError, TypeError):
        return []
    return values if len(values) == 2 and all(math.isfinite(value) and 0 <= value <= 1 for value in values) else []


def market_probability(market):
    values = _prices(market)
    if not values or not _valid_binary(market) or not math.isclose(sum(values), 1, abs_tol=.02):
        return None
    return values[[str(v).strip().lower() for v in _list(market["outcomes"])].index("yes")]


def official_outcome(market):
    """Accept a provider-resolved binary payout; closure or a last quote alone is insufficient."""
    if market.get("closed") is not True:
        return {"status": "Awaiting Resolution"}
    value = market_probability(market)
    prices = _prices(market)
    resolved_at = _time(market.get("closedTime"))
    if (
        str(market.get("umaResolutionStatus") or "").casefold() != "resolved"
        or market.get("resolutionStatus") in ("cancelled", "canceled", "invalid")
        or sorted(prices) != [0, 1] or not resolved_at or not market.get("slug")
    ):
        return {"status": "Review Required", "reason": "Ambiguous Resolution"}
    return {"status": "Resolved", "observed_yes": value == 1, "resolved_at": resolved_at.isoformat(),
            "source": f"{GAMMA}/markets/slug/{market['slug']}"}


async def list_markets(client, *, closed, limit, politics_tag=None, offset=0):
    response = await client.get(f"{GAMMA}/markets", params={
        "closed": str(closed).lower(), "limit": limit,
        **({"offset": offset} if offset else {}),
        "order": "endDate" if closed else "volume24hr", "ascending": "false",
        **({} if closed else {"active": "true"}),
        **({"tag_id": politics_tag} if politics_tag else {}),
    })
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, list):
        raise ValueError("Polymarket returned an invalid market list.")
    return [row for row in result if isinstance(row, dict)][:limit]


async def daily_snapshot(client, root: Path):
    async with _capture_lock:
        now = _now()
        path = root / "daily" / f"{now.date().isoformat()}.json"
        if snapshot := read_json(path, None):
            return snapshot
        markets = await list_markets(client, closed=False, limit=30)
        if not markets:
            raise ValueError("Polymarket returned an empty Top 30 list.")
        snapshot = {"observed_at": _now().isoformat(), "ranking": "Polymarket 24-hour volume, descending", "markets": markets}
        write_json(path, snapshot)
        return snapshot


def _live_contracts(snapshot):
    for row in snapshot["markets"]:
        yield row, row, snapshot["observed_at"], snapshot["observed_at"], None


def _historical_contracts(markets, root):
    archives = {}
    for path in sorted((root / "daily").glob("*.json")):
        snapshot = read_json(path, {})
        for row in snapshot.get("markets", []):
            archives.setdefault(_market_key(row), []).append((row, snapshot["observed_at"]))
    for current in markets:
        deadline = _time(current.get("endDate"))
        if deadline is None:
            yield current, None, None, None, None
            continue
        settlement = official_outcome(current)
        closed_at = _time(settlement.get("resolved_at"))
        cutoff = min(deadline, closed_at) - timedelta(days=14) if closed_at else deadline - timedelta(days=14)
        matches = [(row, observed) for row, observed in archives.get(_market_key(current), [])
                   if _time(observed) and _time(observed) <= cutoff]
        if matches:
            original, captured = max(matches, key=lambda pair: pair[1])
        elif (updated := _time(current.get("updatedAt"))) and updated <= cutoff:
            # The closed record may only be used when its own version predates the cutoff.
            original, captured = current | {"closed": False}, updated.isoformat()
        else:
            original, captured = None, None
        yield current, original, cutoff.isoformat(), captured, settlement


def load_ready_cohort(root, limit, minimum_cutoff, *, condition_id=None):
    saved = read_json(root / "historical-ready.json", None)
    if not saved or saved.get("version") != HISTORICAL_READY_VERSION:
        return None
    try:
        cases = []
        for case in saved["cases"]:
            if condition_id and str(case.get("condition_id", "")).casefold() != condition_id.casefold():
                continue
            cutoff = _time(case["as_of"])
            if not cutoff or (minimum_cutoff and cutoff <= minimum_cutoff):
                continue
            label = saved["settlements"][case["case_id"]]
            rules_at = _time(case["market_rules_at"])
            settled = _time(label["resolved_at"])
            if not rules_at or rules_at > cutoff or not settled or settled <= cutoff or label["status"] != "Resolved":
                return None
            cases.append(case)
        if len(cases) < limit:
            return None
        selected = cases[:limit]
        if any(not frozen_sources(case, "expanded") for case in selected):
            return None
        labels = {case["case_id"]: saved["settlements"][case["case_id"]] for case in selected}
        return selected, labels, saved["selection"] | {"target": limit, "accepted": limit, "available": len(cases)}
    except (KeyError, TypeError, ValueError):
        return None


def save_ready_cohort(root, cases, labels, selection):
    if len(cases) < 30:
        return
    existing = read_json(root / "historical-ready.json", None)
    if existing and existing.get("version") == HISTORICAL_READY_VERSION and len(existing["cases"]) > len(cases):
        return
    write_json(root / "historical-ready.json", {"version": HISTORICAL_READY_VERSION, "cases": cases,
              "settlements": labels, "selection": selection})


async def collect_cohort(mode, limit, root, *, minimum_cutoff=None, client=None, market_url=None):
    if not market_url and mode == "historical" and (ready := load_ready_cohort(root, limit, minimum_cutoff)):
        return ready
    if client is None:
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": USER_AGENT}) as session:
            return await collect_cohort(mode, limit, root, minimum_cutoff=minimum_cutoff, client=session, market_url=market_url)

    selected_event = None
    if market_url:
        limit = 1
        market, selected_event = await resolve_market_url(client, market_url)
        policy = "Contract supplied by URL. Wikipedia evidence is frozen at the forecast cutoff."
        if mode == "live":
            captured = _now().isoformat()
            pool = [(market, market, captured, captured, None)]
        else:
            condition_id = _market_key(market)
            settlement = official_outcome(market)
            if condition_id and settlement["status"] == "Resolved" and (ready := load_ready_cohort(
                root, 1, minimum_cutoff, condition_id=condition_id,
            )):
                cases, labels, _ = ready
                return cases, labels, {"policy": policy, "target": 1, "collected": 1,
                                      "accepted": 1, "excluded": [], "not_screened": 0}
            pool = list(_historical_contracts([market], root))
            if condition_id and settlement["status"] == "Resolved" and pool[0][1] is None:
                from .historical_archive import published_contracts
                archived = await published_contracts(
                    client, root, now=_now(), limit=1,
                    political=lambda row: row["id"].casefold() == condition_id.casefold(),
                )
                if archived:
                    current, original, at = archived[0]
                    pool = [(current, original, at, at, official_outcome(current))]
    elif mode == "live":
        daily = await daily_snapshot(client, root)
        pool = list(_live_contracts(daily))
        policy = daily["ranking"]
    else:
        # This is a resolved-market cohort. A present-day rank never stands in for a historical rank.
        tag = None
        try:
            response = await client.get(f"{GAMMA}/tags/slug/politics")
            response.raise_for_status()
            tag = response.json().get("id")
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        rows = []
        for offset in range(0, 500, 100):
            page = await list_markets(client, closed=True, limit=100, politics_tag=tag, offset=offset)
            rows.extend(page)
            if len(page) < 100:
                break
        if tag:
            rows = [row | {"_politics_filter": True} for row in rows]
        pool = list(_historical_contracts(rows, root))
        if sum(row[1] is not None and (row[4] or {}).get("status") == "Resolved" for row in pool) < limit * 2:
            from .historical_archive import published_contracts
            try:
                archived = await published_contracts(client, root, now=_now(), political=_political)
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                archived = []
            pool = [(current, original, at, at, official_outcome(current)) for current, original, at in archived] + pool
        pool.sort(key=lambda row: row[1] is None)
        policy = "Resolved political markets. Prefer 8–30 day horizons and published pre-resolution rules."

    cases, labels, exclusions = [], {}, []
    seen = set()
    pending = []
    wiki_client = CohortWikiClient(client) if mode == "historical" else client

    async def freeze(current, original, cutoff, captured, settlement):
        case = _contract(original, cutoff, selected_event or _event_slug(original))
        case["resolution_source"] = ""
        identity = json.dumps([HISTORICAL_READY_VERSION, case, captured, original.get("_rules_archive")], sort_keys=True)
        cache = root / "historical_evidence" / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
        saved = read_json(cache, None) if mode == "historical" else None
        if saved is not None:
            case = saved
        else:
            evidence, manifest = await discover_wikipedia(
                wiki_client, question=case["question"], rules=case["rules"], as_of=cutoff,
                seed_urls=[], title_discovery=True, max_pages=2 if mode == "historical" else 5,
                max_link_attempts=2 if mode == "historical" else None,
            )
            fingerprint = hashlib.sha256(json.dumps([
                identity, [(page["revision_id"], page["sha256"], page.get("title_match_score")) for page in evidence],
            ], sort_keys=True).encode()).hexdigest()
            case.update(case_id=f"market_{fingerprint[:24]}", evidence=evidence, retrieval_manifest=manifest,
                        market_rules_at=captured,
                        **({"market_rules_archive": original["_rules_archive"]} if "_rules_archive" in original else {}))
        if not frozen_sources(case, "expanded"):
            return None, None
        if saved is None and mode == "historical":
            write_json(cache, case)
        at_cutoff = utc_timestamp(captured, "quote capture") == cutoff
        quote = market_probability(original) if at_cutoff and (
            mode == "live" or not original.get("_published_rules") and original.get("closed") is not True
            and original.get("updatedAt") != current.get("updatedAt")
        ) else None
        label = {
            **(settlement or {"status": "Awaiting Resolution"}),
            "slug": original["slug"], "condition_id": _market_key(current),
            "market_probability": quote, "market_observed_at": captured if quote is not None else None,
        }
        return case, label

    async def flush():
        results = await asyncio.gather(*(freeze(*candidate[3:]) for candidate in pending), return_exceptions=True)
        for (question, key, condition, *_), result in zip(pending, results):
            if len(cases) >= limit:
                break
            if key in seen or condition in seen:
                exclusions.append({"question": question, "reason": "Duplicate Contract"})
            elif isinstance(result, Exception):
                if isinstance(result, httpx.HTTPStatusError) and result.response.status_code == 429:
                    reason = "Wikipedia Rate Limit"
                else:
                    reason = "Historical Revision Unavailable" if mode == "historical" else "Insufficient Evidence"
                exclusions.append({"question": question, "reason": reason})
            else:
                case, label = result
                if case is None:
                    exclusions.append({"question": question, "reason": "Insufficient Evidence"})
                else:
                    seen.update((key, condition))
                    cases.append(case)
                    labels[case["case_id"]] = label
        pending.clear()

    for current, original, cutoff_value, captured, settlement in pool:
        question = str(current.get("question") or current.get("slug") or "Unknown contract")
        def excluded(reason):
            exclusions.append({"question": question, "reason": reason})

        if not market_url and not _political(current):
            excluded("Outside Politics")
            continue
        if not _market_key(current) or not _valid_binary(current):
            excluded("Unsupported Market Type")
            continue
        key = re.sub(r"\W+", " ", question.casefold()).strip()
        condition = _market_key(current).lower()
        if key in seen or condition in seen:
            excluded("Duplicate Contract")
            continue
        if mode == "historical" and (settlement or {}).get("status") != "Resolved":
            excluded("Ambiguous Resolution")
            continue
        if original is None or cutoff_value is None or captured is None:
            excluded("Cutoff-valid Market Rules Unavailable")
            continue
        cutoff = utc_timestamp(cutoff_value, "forecast cutoff")
        created = _time(original.get("createdAt"))
        if created is None or created > cutoff:
            excluded("Market Creation After Forecast Cutoff or Unverified")
            continue
        if minimum_cutoff and cutoff <= minimum_cutoff:
            excluded("Forecast Cutoff Not After Verified Model Knowledge Cutoff")
            continue
        if mode == "historical" and utc_timestamp(settlement["resolved_at"], "resolution") <= cutoff:
            excluded("Market Settled Before Forecast Cutoff")
            continue
        if mode == "live":
            try:
                latest = await _gamma(client, "market", original["slug"])
            except (ValueError, KeyError):
                excluded("Current Market Status Unavailable")
                continue
            if _market_key(latest) != _market_key(original) or latest.get("closed") is not False or (
                (_time(latest.get("endDate")) or cutoff) <= _now()
            ) or market_probability(latest) in (0, 1):
                excluded("Market Already Closed or Outcome Indicated")
                continue
        pending.append((question, key, condition, current, original, cutoff, captured, settlement))
        if len(pending) >= (3 if mode == "historical" else 1):
            await flush()
            if len(cases) >= limit:
                break
    if pending and len(cases) < limit:
        await flush()
    selection = {"policy": policy, "target": limit, "collected": len(pool), "accepted": len(cases),
                 "excluded": exclusions, "not_screened": len(pool) - len(cases) - len(exclusions)}
    if mode == "historical":
        save_ready_cohort(root, cases, labels, selection)
    return cases, labels, selection
