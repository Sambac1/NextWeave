"""Bounded decision evidence from a frozen OKF release, never other samples."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time as wall_time, timezone
import re
import time

from app.knowledge.retrieval import EvidenceIndex, VERSION, event_upper_bound, terms
from app.knowledge.sentences import complete_excerpt, sentence_records
from app.knowledge.table_evidence import parse_table_evidence
from app.simulation.context import behavior_refs

PASSAGE_RETRIEVAL_VERSION = "frozen-source-passages-2"
PASSAGE_LIMIT = 4
PASSAGE_MAX_CHARS = 1200
PASSAGE_TOTAL_CHARS = 4800


def _instant(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.utcoffset() is not None else None


def _source_available(metadata, cutoff, *, require_revision):
    if cutoff is None or metadata.get("source_sdk_fallback"):
        return False
    offset = metadata.get("source_span_start")
    if type(offset) is not int or offset < 0 or not all(
        metadata.get(key) for key in ("resource", "source_id", "snapshot_sha256")
    ):
        return False
    revision = _instant(metadata.get("source_revision_at"))
    if require_revision:
        return bool(metadata.get("source_revision_id") and revision and revision <= cutoff)
    available = revision or _instant(metadata.get("source_retrieved_at"))
    return bool(available and available <= cutoff)


def _passage_sentences(document):
    body = str(document.get("body") or "")
    title = str(document.get("title") or "")
    # Storage can split large sections at arbitrary offsets. Skip the first
    # sentence of a continuation, which may start mid-sentence.
    skip_first = bool(re.search(r"\(part (?:[2-9]|\d{2,})\)$", title))
    table_ranges = []
    cursor = 0
    for table in parse_table_evidence(body):
        raw = table['original_quote']
        offset = body.find(raw, cursor)
        if offset < 0:
            continue
        cursor = offset + len(raw)
        table_ranges.append((offset, cursor))
        row_cursor = 0
        for row in table['rows']:
            quote = row['original_quote']
            start = raw.find(quote, row_cursor)
            if start < 0:
                continue
            # Inherited rowspan values can require an earlier contiguous span.
            row_cursor = start
            if not 12 <= len(quote) <= 1800:
                continue
            schema = {"columns": [field['column_path'] for field in row['fields']],
                      "values": [field['value'] for field in row['fields']],
                      "ambiguity": list(dict.fromkeys([*table['ambiguity'], *row['ambiguity']]))}
            unit = {"id": row['row_id'], "start": offset + start, "end": offset + start + len(quote),
                    "quote": quote, "kind": "table_row", "table_schema": schema}
            yield unit, set(terms(quote + " " + str(schema)))
    for index, row in enumerate(sentence_records(body)):
        if index == 0 and skip_first:
            continue
        quote = row["quote"]
        if any(row['start'] < end and row['end'] > start for start, end in table_ranges):
            continue
        if 12 <= len(quote) <= PASSAGE_MAX_CHARS and not any(mark in quote for mark in ("{|", "|}", "|-")):
            yield row, set(terms(quote))


def declared_documents(snapshot) -> list[dict]:
    """Custom scenarios share retrieval, while clearly retaining caller provenance."""
    aliases = {n.casefold(): a.agent_id for a in snapshot.agents
               for n in [a.agent_id, a.name, *a.aliases]}
    documents = []
    for event in snapshot.timeline:
        if event.origin != "historical":
            continue
        metadata = {"title": event.description, "description": event.description,
                    "timeline_id": event.timeline_id, "timestamp": event.timestamp,
                    "occurred_month": event.timestamp[:7], "origin": "historical",
                    "actor_ids": [aliases.get(n.casefold(), n) for n in event.actors],
                    "source_sections": event.source_sections, "evidence_refs": [],
                    "date_validation": {"status": "caller_supplied"}}
        documents.append({"id": event.timeline_id, "kind": "timeline", "path": "",
                          "metadata": metadata, "body": event.description + "\n" + event.evidence_text})
    for actor in snapshot.agents:
        documents.append({"id": actor.agent_id, "kind": "actor", "path": "", "body": "",
                          "metadata": {"title": actor.name, "actor_id": actor.agent_id,
                                       "aliases": actor.aliases, "description": actor.description}})
    return documents


def _short(text: str, limit=600) -> str:
    # This is a query, not a quotation or a replacement for stored evidence.
    return str(text)[:limit]


class DecisionEvidenceRetriever:
    def __init__(self, frozen, documents: list[dict], *, version: str | None = None,
                 cutoff_date: str = "", cutoff_at: str | None = None, source_passages: bool = True):
        self.index = EvidenceIndex(documents, version=version)
        self.history = {e.timeline_id: e.model_copy(deep=True) for e in frozen.timeline
                        if e.origin == "historical"}
        self.cutoff = cutoff_date or max((event_upper_bound({"timestamp": e.timestamp})
                                         for e in self.history.values()), default="")
        self.version = version
        self._eligible = {h["id"] for h in self.index.search(kind="timeline", limit=len(self.history) + 1,
                         validated_only=True, before_date=self.cutoff) if h["id"] in self.history}
        if cutoff_at is not None:
            cutoff = _instant(cutoff_at)
        else:
            try:
                cutoff = datetime.combine(date.fromisoformat(self.cutoff), wall_time.max, timezone.utc)
            except ValueError:
                cutoff = None
        self._source_sections = {
            document["id"] for document in documents
            if source_passages and (document.get("kind") or document.get("type")) == "section"
            and _source_available(document["metadata"], cutoff, require_revision=cutoff_at is not None)
        }
        self._background = {
            document["id"] for document in documents
            if source_passages and (document.get("kind") or document.get("type")) == "background"
            and _source_available(document["metadata"], cutoff, require_revision=cutoff_at is not None)
        }
        self._passage_cache = {}

    def _source_passages(self, queries, actor):
        if not self._source_sections and not self._background:
            return []
        names = " ".join([actor.name, *actor.aliases[:4]])
        ranked = {}
        for query in [names, *queries.values()]:
            if not query.strip():
                continue
            query_terms = set(terms(query, limit=48))
            for rank, hit in enumerate(self.index.search(query, kind="section", limit=4,
                                           allowed_ids=self._source_sections), 1):
                identifier = hit["id"]
                if identifier not in self._passage_cache:
                    document = self.index.document(identifier)
                    self._passage_cache[identifier] = list(_passage_sentences(document))
                for row, words in self._passage_cache[identifier]:
                    overlap = len(query_terms & words)
                    if not overlap:
                        continue
                    key = f"{identifier}_{row['id']}"
                    score, _, _ = ranked.get(key, (0, hit, row))
                    ranked[key] = (score + min(overlap, 8) / (8 + rank), hit, row)
        # Use one existing passage slot for the best matching structured
        # background quote. Keep the same context and call budgets.
        background_query = " ".join([names, queries["situation"], queries["recent_events"]])
        for hit in self.index.search(background_query, kind="background", limit=3, allowed_ids=self._background):
            document = self.index.document(hit["id"])
            quote = document["metadata"]["quote"]
            if len(quote) > PASSAGE_MAX_CHARS or len(hit["matched_terms"]) < 2:
                continue
            row = {"quote": quote, "start": 0, "end": len(quote), "kind": "background"}
            ranked[hit["id"]] = (float("inf"), hit, row)
            break
        passages, seen_quotes, remaining = [], set(), PASSAGE_TOTAL_CHARS
        for identifier, (_, hit, row) in sorted(ranked.items(), key=lambda pair: (-pair[1][0], pair[0])):
            if row["quote"] in seen_quotes:
                continue
            cost = len(row['quote']) + len(str(row.get('table_schema', '')))
            if cost > remaining:
                continue
            metadata = hit["metadata"]
            offset = metadata["source_span_start"]
            passages.append({"passage_id": identifier, "quote": row["quote"],
                             "section": metadata["title"], "knowledge_path": hit["path"],
                             "url": metadata["resource"], "source_id": metadata["source_id"],
                             "snapshot_sha256": metadata["snapshot_sha256"],
                             "revision_id": metadata.get("source_revision_id"),
                             "available_at": metadata.get("source_revision_at") or metadata.get("source_retrieved_at"),
                             "span_start": offset + row["start"], "span_end": offset + row["end"]})
            if row.get('kind') == 'table_row':
                passages[-1].update(kind='table_row', table_schema=row['table_schema'])
            elif row.get('kind') == 'background':
                passages[-1].update(kind='background', fact_status=metadata['fact_status'],
                                    date_precision=metadata['date_precision'],
                                    date_original=metadata['date_original'],
                                    reported_bounds=metadata['reported_bounds'])
            seen_quotes.add(row["quote"])
            remaining -= cost
            if len(passages) == PASSAGE_LIMIT:
                break
        return passages

    def retrieve(self, snapshot, actor, scenario: str, k: int = 6) -> tuple[list[dict], dict]:
        started = time.perf_counter()
        k = max(1, min(12, k))
        recent = [e.description for e in snapshot.timeline if e.origin == "simulated"][-3:]
        queries = {
            "situation": _short(scenario, 1200),
            "recent_events": _short(" ".join(recent), 900),
            "objectives": _short(" ".join(d.description for d in actor.desires[:3])),
            "beliefs": _short(" ".join(v.assessment for v in actor.beliefs.situation_views[:2])),
            "constraints": _short(" ".join(v.description for v in actor.decision_core.limits[:4])) if actor.decision_core else _short(" ".join(actor.constraints[:4])),
        }
        focus = self.index.actor_ids([actor.agent_id, actor.name, *actor.aliases]) | {actor.agent_id}
        pool = {}
        routes = {}

        def add(hit, route, rank):
            identifier = hit["id"]
            if identifier not in self._eligible:
                return
            item = pool.setdefault(identifier, {"hit": hit, "score": 0.0, "routes": []})
            item["score"] += 1 / (20 + rank)
            if route not in item["routes"]:
                item["routes"].append(route)

        for rank, hit in enumerate(self.index.search(kind="timeline", boost_actor_ids=focus,
                                  validated_only=True, before_date=self.cutoff, limit=24), 1):
            if hit["score"] > 0:
                add(hit, "actor_history", 40 + rank)

        for route, query in queries.items():
            if not query.strip():
                continue
            hits = self.index.search(query, kind="timeline", boost_actor_ids=focus,
                                     validated_only=True, before_date=self.cutoff, limit=24)
            routes[route] = [h["id"] for h in hits if h["score"] > 0 and h["matched_terms"]]
            for rank, hit in enumerate(hits, 1):
                if hit["score"] > 0:
                    add(hit, route, rank)
            # This route recalls events, while source passages are selected
            # separately with a source-availability check.
            for section in self.index.search(query, kind="section", limit=3):
                for identifier in sorted(self.index.related_events(section["metadata"]["section_path"])):
                    document = self.index.document(identifier)
                    if document:
                        add(document, "section_link", 25)

        topic = " ".join([queries["situation"], queries["recent_events"]])
        # No actor filter here: counterpart evidence should not be crowded out
        # by all of the focal actor's historical mentions. Not a contradiction detector.
        for rank, hit in enumerate(self.index.search(topic, kind="timeline", limit=24,
                         validated_only=True, before_date=self.cutoff), 1):
            if hit["score"] > 0:
                add(hit, "wider_context", rank)
        for rank, identifier in enumerate(actor.relevant_timeline_refs, 1):
            document = self.index.document(identifier)
            if document:
                add(document, "actor_profile_link", 40 + rank)

        required = {r for v in [*actor.beliefs.situation_views, *actor.beliefs.actor_models]
                    for r in v.timeline_refs} | behavior_refs(actor)
        historical_required = required & self.history.keys()
        invalid = historical_required - self._eligible
        if invalid:
            raise ValueError("Required historical evidence is outside the frozen date/validation boundary: " + ", ".join(sorted(invalid)))
        ranked = sorted(pool, key=lambda i: (-pool[i]["score"], i))
        selected = []

        def choose(identifier):
            if identifier and identifier not in selected and len(selected) < k:
                selected.append(identifier)

        choose(ranked[0] if ranked else "")
        if queries["recent_events"]:
            choose(next((i for i in routes.get("recent_events", []) if i in pool), ""))
        # Explicit constraints get a reserved opportunity, without labeling a
        # lexical match as a verified restriction or counterargument.
        if queries["constraints"]:
            choose(next((i for i in routes.get("constraints", [])
                         if i in pool), ""))
        if queries["objectives"]:
            choose(next((i for i in routes.get("objectives", []) if i in pool), ""))
        choose(next((i for i in ranked if "wider_context" in pool[i]["routes"]
                     and not focus.intersection(pool[i]["hit"]["metadata"].get("actor_ids", []))), ""))
        for identifier in ranked:
            choose(identifier)
        selected.extend(sorted(historical_required - set(selected)))

        evidence = []
        for identifier in selected:
            document = self.index.document(identifier)
            m = document["metadata"]
            event = self.history[identifier]
            item = pool.get(identifier, {})
            references = deepcopy(m.get("evidence_refs") or [])
            evidence.append({"timeline_id": identifier, "timestamp": event.timestamp,
                "description": event.description, "origin": "historical",
                "knowledge_version": self.version, "knowledge_path": document.get("path", ""),
                "date_validation": m.get("date_validation", {}),
                "evidence": references, "evidence_text": event.evidence_text,
                "participants": [self.index.participant_card(a) for a in (m.get("actor_ids") or [])[:6]],
                "retrieval_score": round(item.get("score", 0), 6),
                "retrieval_reasons": [*item.get("routes", []),
                                      *(["required_reference"] if identifier in required else [])]})
        # This audit belongs to this actor/sample; it is never exported to OKF
        # or copied to another actor's public context.
        audit = {"version": VERSION, "knowledge_version": self.version, "cutoff_date": self.cutoff,
                 "queries": queries, "candidate_count": len(pool), "selected_ids": selected,
                 "required_ids": sorted(historical_required),
                 "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                 "model_calls": 0, "section_policy": "recall_linked_validated_events_only"}
        if passages := self._source_passages(queries, actor):
            audit["source_passages"] = passages
            audit["section_policy"] = PASSAGE_RETRIEVAL_VERSION
        return evidence, audit


def evidence_card(item: dict) -> dict:
    """Small source-located excerpts; preserve full quotes in the run snapshot."""
    references = []
    for ref in item.get("evidence", [])[:2]:
        quote = str(ref.get("quote") or "")
        excerpt = quote if len(quote) <= 600 else complete_excerpt(quote, 600)
        # When the sentence is too long, retain the locator and disclose omission.
        references.append({k: ref.get(k) for k in
                           ("source_id", "snapshot_sha256", "url", "section_paths", "span_start", "span_end", "attribution")}
                          | {"quote": excerpt, "excerpt_omitted": bool(quote and not excerpt),
                             "excerpt_shortened": len(excerpt) < len(quote)})
    return {"timeline_id": item["timeline_id"], "knowledge_version": item.get("knowledge_version"),
            "date_validation": item.get("date_validation"), "sources": references,
            "participants": item.get("participants", []),
            "retrieval_reasons": item.get("retrieval_reasons", [])}
