"""Discover pre-outcome Polymarket rule snapshots published by ForecastBench."""
from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import re
from collections import defaultdict
from datetime import timedelta

import httpx

from app.storage.json_store import read_json, write_json

from .evidence import utc_timestamp
from .markets import GAMMA

REPO = "forecastingresearch/forecastbench-datasets"
API = f"https://api.github.com/repos/{REPO}"
FILENAME = re.compile(r"20\d{2}-\d{2}-\d{2}-llm\.json")
CONDITION = re.compile(r"0x[a-fA-F0-9]{64}")
COMMIT = re.compile(r"[a-f0-9]{40}")


def _verified_time(value):
    try:
        return utc_timestamp(value, "archive timestamp")
    except ValueError:
        return None


async def _json(client, url, **kwargs):
    response = await client.get(url, **kwargs)
    response.raise_for_status()
    return response.json()


class QuestionPrefix:
    """Decode the complete Polymarket rows in the publisher's grouped question sets."""
    def __init__(self, filename):
        self.filename = filename
        self.text = ""
        self.offset = None
        self.rows = []
        self.decoder = json.JSONDecoder()

    def feed(self, chunk):
        self.text += chunk
        if self.offset is None:
            start = re.search(r'"questions"\s*:\s*\[', self.text)
            if start is None:
                return False
            header = json.loads(self.text[:start.start()].rstrip().rstrip(",") + "}")
            if header.get("question_set") != self.filename:
                raise ValueError("Archive filename does not match its question set.")
            self.offset = start.end()
        while self.offset < len(self.text):
            while self.offset < len(self.text) and self.text[self.offset] in " ,\n\r\t":
                self.offset += 1
            if self.offset == len(self.text):
                return False
            if self.text[self.offset] == "]":
                return True
            try:
                row, end = self.decoder.raw_decode(self.text, self.offset)
            except json.JSONDecodeError:
                return False
            self.offset = end
            if row.get("source") != "polymarket":
                if self.rows:
                    return True
                continue
            if CONDITION.fullmatch(str(row.get("id") or "")) and row.get("question") and row.get("background"):
                self.rows.append({key: row[key] for key in ("id", "question", "background")})
        return False


async def _read_release(client, url, filename):
    reader = QuestionPrefix(filename)
    decode = codecs.getincrementaldecoder("utf-8")()
    async with client.stream("GET", url, timeout=httpx.Timeout(50, connect=12)) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if reader.feed(decode.decode(chunk)):
                return reader.rows
            if len(reader.text) > 5_000_000:
                raise ValueError("Historical question-set download exceeded its size limit.")
    raise ValueError("Historical question-set download ended before the market rows were complete.")


async def _release(client, root, filename, blob=None):
    target = root / "published_rules" / filename
    if cached := read_json(target, None):
        return cached
    path = f"datasets/question_sets/{filename}"
    history = await _json(client, f"{API}/commits?path={path}&per_page=30", timeout=20)
    # A pinned first publication prevents a later edit from becoming historical evidence.
    if not history or len(history) >= 30:
        return None
    revision = history[-1]
    commit = revision["sha"]
    published = _verified_time(revision["commit"]["committer"]["date"])
    if not COMMIT.fullmatch(commit) or published is None:
        return None
    raw_url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}"
    rows = None
    if len(history) == 1 and isinstance(blob, str) and COMMIT.fullmatch(blob):
        try:
            async with asyncio.timeout(45):
                mirror = await client.get(f"https://cdn.jsdelivr.net/gh/{REPO}@{commit}/{path}", timeout=35)
                mirror.raise_for_status()
                body = mirror.content
                reader = QuestionPrefix(filename)
                if (len(body) <= 5_000_000
                    and hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest() == blob
                    and reader.feed(body.decode())):
                    rows = reader.rows
        except (httpx.HTTPError, ValueError, TimeoutError):
            pass
    if rows is None:
        async with asyncio.timeout(55):
            rows = await _read_release(client, raw_url, filename)
    result = {"published_at": published.isoformat(), "url": raw_url, "questions": rows}
    write_json(target, result)
    return result


async def published_contracts(client, root, *, now, political, limit=450):
    blobs = {}
    names = {path.name for path in (root / "published_rules").glob("*.json")}
    try:
        index = await _json(client, f"{API}/contents/datasets/question_sets?ref=main", timeout=15)
        blobs = {row["name"]: row.get("sha") for row in index
                 if isinstance(row, dict) and FILENAME.fullmatch(row.get("name", ""))}
        names.update(blobs)
    except (httpx.HTTPError, ValueError, TypeError):
        pass
    # A local pin remains usable when GitHub rate-limits the directory lookup.
    latest = (now - timedelta(days=8)).date().isoformat()
    names = sorted((name for name in names if FILENAME.fullmatch(name) and name[:10] <= latest), reverse=True)[:14]
    releases = []
    for start in range(0, len(names), 4):
        batch = await asyncio.gather(*(_release(client, root, name, blobs.get(name)) for name in names[start:start + 4]),
                                     return_exceptions=True)
        releases.extend(item for item in batch if isinstance(item, dict))
        if len({row["id"] for release in releases for row in release["questions"] if political(row)}) >= limit:
            break
    by_id = defaultdict(list)
    for release in releases:
        for row in release["questions"]:
            if political(row):
                by_id[row["id"].lower()].append((row, release["published_at"], release["url"]))
    ids = list(by_id)[:limit]
    async def closed_markets(chunk):
        return await _json(client, f"{GAMMA}/markets", params=[
            ("closed", "true"), ("limit", str(len(chunk))), *(("condition_ids", identifier) for identifier in chunk),
        ], timeout=25)

    batches = await asyncio.gather(*(closed_markets(ids[start:start + 45]) for start in range(0, len(ids), 45)),
                                   return_exceptions=True)
    markets = {str(row.get("conditionId", "")).lower(): row for batch in batches if isinstance(batch, list)
               for row in batch if isinstance(row, dict)}
    entries = []
    for identifier, snapshots in by_id.items():
        current = markets.get(identifier)
        if not current:
            continue
        closed = _verified_time(current.get("closedTime"))
        nominal = _verified_time(current.get("endDate"))
        if not closed or not nominal or closed > now:
            continue
        bound = min(closed, nominal)
        valid = [(row, at, url) for row, at, url in snapshots
                 if (captured := _verified_time(at)) and timedelta(days=3) <= bound - captured <= timedelta(days=90)
                 and " ".join(row["background"].split()) == " ".join(str(current.get("description") or "").split())]
        if not valid:
            continue
        def preference(item):
            horizon = bound - _verified_time(item[1])
            return (not timedelta(days=8) <= horizon <= timedelta(days=30), abs(horizon - timedelta(days=14)))
        row, at, url = min(valid, key=preference)
        original = current | {"closed": False, "question": row["question"], "description": row["background"],
                              "_rules_archive": url, "_published_rules": True}
        entries.append((preference((row, at, url)), current, original, at))
    entries.sort(key=lambda item: (item[0], item[1]["conditionId"]))
    return [(current, original, at) for _, current, original, at in entries]
