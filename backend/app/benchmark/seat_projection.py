"""A bounded national polling model for supported seat contracts."""
from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import random
import re

from app.knowledge.scraper import split_sections


def build_projection(question, sources, cutoff: date):
    target = re.search(r"\(([A-Z][A-Za-z0-9]{0,7})\)", question)
    rules = []
    for source in sources:
        for section in split_sections(source["content"], 1, max_chars=None):
            text = section["content"]
            seats = re.search(r"\b(?:made up of|consists of|comprises)\s+([\d,]+)\s+seats\b", text, re.I)
            threshold = re.search(r"\bthreshold of\s+(\d+(?:\.\d+)?)%\s+of the national vote\b", text, re.I)
            if seats and threshold and re.search(r"proportional representation", text, re.I):
                rules.append((int(seats[1].replace(",", "")), float(threshold[1]), source["revision_id"]))
    if not target or len({row[:2] for row in rules}) != 1:
        return None
    size, floor, rules_revision = rules[0]
    if not 0 < size <= 2000 or not 0 <= floor < 100:
        return None
    target = target[1]
    polls = []
    for i, source in enumerate(sources):
        for section in split_sections(source["content"], i + 1, max_chars=None):
            if section["title"] != str(cutoff.year):
                continue
            for table in re.findall(r"(?ms)^\{\|.*?^\|\}", section["content"]):
                groups = re.split(r"(?m)^\|-\s*$", table)
                if not all(label in table for label in ("!Polling firm", "!Fieldwork date", "!Sample size")):
                    continue
                parties = next((labels for group in groups
                                if target in (labels := re.findall(r"(?m)^!\s*([A-Z][A-Za-z0-9]{0,7})\s*$", group))
                                and len(set(labels)) == len(labels) >= 3), [])
                if not parties:
                    continue
                for group in groups:
                    cells = [s.strip() for s in re.findall(r"(?m)^\|([^\n}]*)$", group)]
                    if len(cells) < len(parties) + 3:
                        continue
                    match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s*$", cells[1])
                    if not match:
                        continue
                    try:
                        month = datetime.strptime(match[2][:3], "%b").month
                        observed = date(cutoff.year, month, int(match[1]))
                        sample_size = int(cells[2].replace(",", ""))
                        shares = list(map(float, cells[3:3 + len(parties)]))
                    except ValueError:
                        continue
                    if (cutoff - timedelta(days=45) <= observed <= cutoff and sample_size >= 100
                            and all(0 <= share <= 100 for share in shares)
                            and max(0., 100 - sum(shares)) < max(floor, .01) and sum(shares) <= 101):
                        polls.append({"firm": cells[0], "fieldwork_end": observed.isoformat(),
                                      "sample_size": sample_size, "shares": dict(zip(parties, shares)),
                                      "revision_id": source["revision_id"]})
    latest = {}
    for poll in sorted(polls, key=lambda row: row["fieldwork_end"]):
        latest[poll["firm"].casefold()] = poll
    if len(latest) < 2 or len({frozenset(row["shares"]) for row in latest.values()}) != 1:
        return None
    return {"model": "poll_bootstrap_national_hamilton_v1", "target": target,
            "total_seats": size, "threshold_pct": floor, "polls": list(latest.values()),
            "rules_revision_id": rules_revision,
            "assumption": "National proportional approximation. District exceptions and turnout shifts are not modeled."}


def draw_count(projection, *, seed, sample):
    rng = random.Random(f"benchmark-seats-v1:{seed}:{sample}")
    poll = rng.choice(projection["polls"])
    shares = {party: max(0., rng.gauss(p, math.sqrt(p * (100 - p) / poll["sample_size"])))
              for party, p in poll["shares"].items()}
    other = max(0., 100 - sum(poll["shares"].values()))
    total = sum(shares.values()) + other
    shares = {party: p / total * 100 for party, p in shares.items()}
    eligible = {party: p for party, p in shares.items() if p >= projection["threshold_pct"] and p > 0}
    if not eligible:
        return None
    weights = {party: p / sum(eligible.values()) * projection["total_seats"] for party, p in eligible.items()}
    seats = {party: int(weights.get(party, 0)) for party in shares}
    remainder = projection["total_seats"] - sum(seats.values())
    for party in sorted(weights, key=lambda p: (-(weights[p] - seats[p]), -shares[p], p))[:remainder]:
        seats[party] += 1
    ranking = sorted(seats, key=lambda party: (-seats[party], -shares[party], party))
    return {"seats": seats, "ranking": ranking, "poll_revision_id": poll["revision_id"],
            "poll_fieldwork_end": poll["fieldwork_end"]}
