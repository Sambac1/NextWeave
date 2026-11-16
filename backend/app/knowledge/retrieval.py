"""Shared, deterministic retrieval over one immutable public knowledge release.

SQLite/OKF remain the durable store. This small inverted index is built once
per experiment, so actor queries never reread Markdown or call a model. The
same tokenizer, BM25 ranking and filters serve knowledge-page searches.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import math
import re

VERSION = "evidence-retrieval-1"
_STOP = set("a an and are as at be been by for from has have in is it of on or that the their this to was were will with would".split())


def terms(text: str, *, limit: int | None = None) -> list[str]:
    """Unicode words plus CJK bigrams; no model, external dictionary or stems."""
    tokens = []
    for run in re.findall(r"[\u3400-\u9fff]+|[^\W_]+", str(text).casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", run):
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
            if len(run) == 1:
                tokens.append(run)
        elif run not in _STOP and (len(run) > 1 or run.isdigit()):
            tokens.append(run)
    if limit is not None:
        return list(dict.fromkeys(tokens))[:limit]
    return tokens


def _named_in(name: str, query: str) -> bool:
    name = str(name).strip().casefold()
    if not name:
        return False
    if not re.search(r"[\u3400-\u9fff]", name):
        return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", query.casefold()) is not None
    return name in query.casefold()


def event_upper_bound(metadata: dict) -> str:
    """A month-only claim is not known to precede an intra-month cutoff."""
    from calendar import monthrange
    value = str(metadata.get("timestamp") or "")
    try:
        if re.fullmatch(r"\d{4}-\d{2}", value):
            y, m = map(int, value.split("-"))
            return date(y, m, monthrange(y, m)[1]).isoformat()
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return ""


class EvidenceIndex:
    """Read-only public documents, scoped term statistics and cheap queries."""

    def __init__(self, documents: list[dict], *, version: str | None = None):
        self.version = version
        self._documents = {}
        self._postings = defaultdict(dict)
        self._lengths = {}
        self._actors = {}
        self._actor_events = defaultdict(set)
        self._section_events = defaultdict(set)
        for source in documents:
            document = deepcopy(source)
            identifier = str(document["id"])
            if identifier in self._documents:
                raise ValueError("Duplicate knowledge document ID")
            kind = document.get("kind") or document.get("type")
            document["kind"] = kind
            m = document["metadata"]
            document.setdefault("title", m.get("title", ""))
            self._documents[identifier] = document
            # Links and hashes are navigation, not useful lexical evidence.
            body = re.sub(r"\]\([^)]*\)", "]", str(document.get("body") or ""))
            text = " ".join([str(document["title"]), str(m.get("description") or ""),
                             body, *map(str, m.get("aliases") or [])])
            counts = Counter(terms(text))
            self._lengths[identifier] = sum(counts.values()) or 1
            for token, count in counts.items():
                self._postings[token][identifier] = count
            if kind == "actor":
                self._actors[identifier] = [document["title"], *(m.get("aliases") or [])]
            if kind == "timeline":
                for actor in m.get("actor_ids") or []:
                    self._actor_events[actor].add(identifier)
                for ref in m.get("evidence_refs") or []:
                    for path in ref.get("section_paths") or []:
                        self._section_events[path].add(identifier)
                for path in m.get("source_sections") or []:
                    self._section_events[path].add(identifier)
        self._average_length = sum(self._lengths.values()) / max(1, len(self._lengths))

    def document(self, identifier: str) -> dict | None:
        value = self._documents.get(identifier)
        return deepcopy(value) if value else None

    def participant_card(self, identifier: str) -> dict:
        document = self._documents.get(identifier, {})
        return {"actor_id": identifier, "name": document.get("metadata", {}).get("title", identifier),
                "knowledge_path": document.get("path", "")}

    def related_events(self, section_path: str) -> set[str]:
        return set(self._section_events.get(section_path, ()))

    def actor_ids(self, names: list[str]) -> set[str]:
        exact = {str(n).casefold() for n in names}
        return {identifier for identifier, aliases in self._actors.items()
                if identifier.casefold() in exact or exact & {a.casefold() for a in aliases}}

    def search(self, query: str = "", *, kind: str = "all", actor_id: str = "",
               from_month: str = "", to_month: str = "", limit: int = 30,
               boost_actor_ids=(), before_date: str = "", validated_only: bool = False,
               allowed_ids: set[str] | None = None) -> list[dict]:
        query_terms = terms(query, limit=48)
        scores = defaultdict(float)
        matched_terms = defaultdict(list)
        n = len(self._documents)
        for token in query_terms:
            posting = self._postings.get(token, {})
            idf = math.log(1 + (n - len(posting) + .5) / (len(posting) + .5))
            for identifier, tf in posting.items():
                norm = tf + 1.2 * (.25 + .75 * self._lengths[identifier] / self._average_length)
                scores[identifier] += idf * tf * 2.2 / norm
                matched_terms[identifier].append(token)
        query_actors = {identifier for identifier, names in self._actors.items()
                        if any(_named_in(name, query) for name in names)} if query.strip() else set()
        boosts = set(boost_actor_ids) | query_actors
        hits = []
        for identifier, document in self._documents.items():
            if allowed_ids is not None and identifier not in allowed_ids:
                continue
            m = document["metadata"]
            if kind != "all" and document["kind"] != kind:
                continue
            if document["kind"] == "background" and (from_month or to_month):
                continue
            associated = set(m.get("actor_ids") or []) | {m.get("actor_id")}
            if actor_id and actor_id not in associated:
                continue
            occurred = str(m.get("occurred_month") or "")
            if occurred and ((from_month and occurred < from_month) or (to_month and occurred > to_month)):
                continue
            if validated_only and document["kind"] == "timeline":
                if m.get("origin", "historical") != "historical" or m.get("date_validation", {}).get("status") not in {"supported", "caller_supplied"}:
                    continue
                upper = event_upper_bound(m)
                if not upper or (before_date and upper > before_date):
                    continue
            direct = bool(boosts & associated) or identifier in boosts
            score = scores.get(identifier, 0.0)
            if query_terms and not score and not direct:
                continue
            reasons = (["fulltext_match"] if score else []) + (["actor_link"] if direct else [])
            hits.append({"id": identifier, "kind": document["kind"], "path": document.get("path", ""),
                         "title": document["title"], "metadata": m,
                         "score": round(score + 2 * direct, 6), "version": self.version,
                         "retrieval_reasons": reasons, "matched_terms": matched_terms[identifier]})
        selected = sorted(hits, key=lambda h: (-h["score"], h["kind"] != "timeline",
                      not h["metadata"].get("simulation_agent", False),
                      h["metadata"].get("timestamp") or h["metadata"].get("occurred_month") or "", h["id"]))[:limit]
        return deepcopy(selected)
