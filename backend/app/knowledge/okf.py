"""Versioned, public OKF read model and rebuildable local search index.

Generated frontmatter is JSON, a YAML 1.2 subset. No model call is needed to
publish, search, follow evidence, or rebuild the index. Runtime BDI is never
exported. Historical releases are immutable so experiments can pin a version.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.storage.json_store import EventRepository, read_json, write_json
from app.storage.knowledge_lock import knowledge_mutation
from app.knowledge.hierarchy import public_profiles, actor_graph
from app.knowledge.evidence import VERSION as EVIDENCE_VERSION, locate_quote, retained_background, validate_history
from app.knowledge.retrieval import EvidenceIndex

PROJECTION_VERSION = "knowledge-9"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def month(value: str) -> str | None:
    match = re.match(r"^(\d{4})-(\d{2})(?:$|-)", str(value))
    if match and 1 <= int(match[2]) <= 12:
        return f"{match[1]}-{match[2]}"
    return None


def public_actor(value: dict) -> dict:
    """An explicit whitelist; nested public_state can also contain assumptions."""
    return {key: value.get(key) for key in (
        "agent_id", "name", "aliases", "agent_type", "role", "title", "affiliation",
        "description", "relevant_timeline_refs",
    ) if value.get(key) is not None}


def _document(path: Path, metadata: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "---\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n---\n\n" + body + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_document(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("Generated knowledge document lacks frontmatter")
    header, separator, body = text[4:].partition("\n---\n")
    if not separator:
        raise ValueError("Unclosed knowledge frontmatter")
    return json.loads(header), body.strip()


class KnowledgeStore:
    def __init__(self, events: EventRepository | None = None):
        self.events = events or EventRepository()
        self.root = self.events.root.parent / "knowledge"
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.events.root.parent / "knowledge_search.sqlite"

    def _case(self, case_id: str) -> Path:
        self.events.directory(case_id)  # validate and scope the identifier
        return self.root / "cases" / case_id

    def manifest(self, case_id: str, *, version: str | None = None) -> dict:
        if version is None:
            return read_json(self._case(case_id) / "current.json", {})
        if not re.fullmatch(r"[0-9a-f]{24}", version):
            raise ValueError("Invalid knowledge version")
        value = read_json(self._case(case_id) / "versions" / version / "manifest.json", {})
        if value.get("case_id") != case_id or value.get("version") != version:
            raise ValueError("Knowledge version does not belong to this case")
        return value

    @knowledge_mutation
    def publish(self, case_id: str) -> dict:
        case = read_json(self.events.directory(case_id) / "event.json", {})
        sources = self.events.load_sources(case_id)
        # Full history is independent of a UI/forecast time-window view.
        history = self.events.load_artifact(case_id, "timeline_full", None)
        if history is None:
            history = self.events.load_artifact(case_id, "timeline", [])
        full_history = history
        from app.knowledge.extraction import _sections
        sections = _sections(sources) if sources else []
        supported,rejected=validate_history(history,sections) if sources else ([],[])
        review = self.events.load_artifact(case_id, "timeline_evidence_review", {})
        retained_facts = retained_background(review.get("rejected", [])) if sources else []
        supported_by_id={e["timeline_id"]:e for e in supported}
        rejected_by_id={e["timeline_id"]:e["reason"] for e in rejected}
        history = [e for e in history if e.get("origin", "historical") == "historical"
                   and month(e.get("timestamp", ""))]
        from app.knowledge.decision_seats import restore_seat_type
        seat_plan = self.events.load_artifact(case_id, "decision_seat_plan", {})
        agents = self.events.load_artifact(case_id, "agents", [])
        actors,known = public_profiles([restore_seat_type(agent, seat_plan) for agent in agents], case_id, case.get("title",""))
        extraction=self.events.load_artifact(case_id,"extraction_manifest",{})
        requires_reanalysis = bool(sources) and (
            extraction.get("evidence_policy") != EVIDENCE_VERSION
            or extraction.get("history_fingerprint") != digest(full_history)
            or extraction.get("profile_fingerprint") != digest(agents)
            or extraction.get("profile_source_fingerprint") != digest([{ "url":s.get("url"),"content":s.get("content"),"sections":s.get("sections",[])} for s in sources]))
        actor_by_name = {}
        for a in actors:
            for name in [a.get("name", ""), *(a.get("aliases") or [])]:
                actor_by_name[str(name).casefold()] = a["agent_id"]
        # Mentioned entities remain discoverable even when not selected as one
        # of the bounded simulation principals. Do not fabricate rich profiles.
        for e in history:
            for mentioned in e.get("retrieval_actors") or []:
                name = str(mentioned.get("name") or "").strip()
                if not name:
                    continue
                if name.casefold() in actor_by_name:
                    existing_actor=next((a for a in actors if a["agent_id"]==actor_by_name[name.casefold()]),None)
                    if existing_actor and existing_actor.get("decision_role")=="mentioned":
                        for key in ("title","affiliation"):
                            if not existing_actor.get(key) and mentioned.get(key): existing_actor[key]=str(mentioned[key])
                    continue
                identifier = "actor_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{case_id}:{name.casefold()}").hex[:20]
                actor_by_name[name.casefold()] = identifier
                actors.append({"agent_id": identifier, "name": name, "aliases": [],
                               "agent_type":str(mentioned.get("entity_type") or "entity"),"description":"",
                               "title":str(mentioned.get("title") or ""),"affiliation":str(mentioned.get("affiliation") or ""),
                               "role_description":str(mentioned.get("role_in_event") or mentioned.get("role") or ""),"profile_scope":"evidence_summary",
                               "relevant_timeline_refs": [],"decision_role":"mentioned","parent_actor_id":None,"membership_status":"unassigned"})
        # Reuse identity facts already extracted for non-simulation entities.
        # Never export the source record wholesale: it may contain model state.
        extracted=self.events.load_artifact(case_id,"figures_merged",{})
        identity_rows=extracted.get("figures",[]) if isinstance(extracted,dict) else extracted
        identity_rows = [*(identity_rows if isinstance(identity_rows,list) else []), *self.events.load_artifact(case_id,"participating_objects",[])]
        identity_facts={}
        for row in identity_rows if isinstance(identity_rows,list) else []:
            if not isinstance(row,dict): continue
            for name in (row.get("canonical_name"),row.get("name")):
                if name: identity_facts[str(name).casefold()]=row
        for actor in actors:
            if actor.get("decision_role")!="mentioned": continue
            fact=identity_facts.get(actor.get("name","").casefold(),{})
            for key in ("title","affiliation"):
                if not actor.get(key) and fact.get(key): actor[key]=str(fact[key])
            if not actor.get("role_description") and fact.get("role_in_event"):
                actor["role_description"]=str(fact["role_in_event"])
            if actor.get("agent_type")=="entity" and fact.get("entity_type"):
                actor["agent_type"]=str(fact["entity_type"])
            from app.domain.agency import non_agent_kind
            object_kind = non_agent_kind(fact) or non_agent_kind(actor)
            if object_kind:
                actor.update(agent_type=object_kind, entity_type_status="non_decision_object", profile_scope="evidence_brief")
        version = digest({"projection":PROJECTION_VERSION,"validator":EVIDENCE_VERSION,"schema": 2,"requires_reanalysis":requires_reanalysis,"evidence_policy":extraction.get("evidence_policy"), "history": history, "actors": actors,
                          "background": retained_facts, "sources": [{"url": s.get("url"), "body": digest(s.get("content", "")),
                                       "revision_id": s.get("revision_id"), "revision_at": s.get("revision_at"),
                                       "retrieved_at": s.get("retrieved_at"), "sdk_fallback": s.get("sdk_fallback"),
                                       "sections": s.get("sections", [])} for s in sources]})[:24]
        case_dir = self._case(case_id)
        release = case_dir / "versions" / version
        existing = read_json(release / "manifest.json", {})
        if existing:
            self._activate(case_id, existing)
            return existing
        now = datetime.now(timezone.utc).isoformat()
        source_records = []
        for index, source in enumerate(sources, 1):
            fragments = source.get("sections") if source.get("sdk_fallback") else None
            materials = fragments or [source]
            for item in materials:
                body = str(item.get("content") or "")
                if not body:
                    continue
                url = str(item.get("url") or source.get("url") or "")
                source_key = "src_" + digest(url or body)[:20]
                snapshot = hashlib.sha256(body.encode()).hexdigest()
                raw = self.root / "references" / source_key / f"{snapshot}.txt"
                raw.parent.mkdir(parents=True, exist_ok=True)
                if not raw.exists():
                    raw.write_text(body, encoding="utf-8")
                relative = f"sources/{source_key}/{snapshot}.md"
                meta = {"type": "Source Snapshot", "title": source.get("title") or url,
                        "description": "Captured source text; extracted claims must cite a span or disclose page-only attribution.",
                        "resource": url, "source_id": source_key, "snapshot_sha256": snapshot,
                        "snapshot_file": raw.relative_to(self.root).as_posix(),
                        "retrieved_at": source.get("retrieved_at"), "revision_id": source.get("revision_id"),
                        "status": "stable", "generated": {"by": "process:source-export", "at": now}}
                if not (self.root / relative).exists():
                    _document(self.root / relative, meta, f"# {meta['title']}\n\nOriginal: {url}\n\nSnapshot: {snapshot}")
                source_records.append({"alias": f"source_{index:02d}", "path": relative,
                                       "source_id": source_key, "snapshot": snapshot,
                                       "text": body, "title": meta["title"], "url": url,
                                       "revision_id": source.get("revision_id"), "revision_at": source.get("revision_at"),
                                       "retrieved_at": source.get("retrieved_at"), "sdk_fallback": bool(source.get("sdk_fallback"))})
        root_path = release.relative_to(self.root).as_posix()
        actor_refs: dict[str, list[str]] = {a["agent_id"]: [] for a in actors}
        docs = []
        for original_event in history:
            e=supported_by_id.get(original_event["timeline_id"],original_event)
            identifier = str(e["timeline_id"])
            # IDs are data, never filenames supplied by a model.
            file_key = digest(identifier)[:24]
            path = f"{root_path}/timeline/{file_key}.md"
            ids = []
            for name in [*(e.get("actors") or []), *[a.get("name", "") for a in e.get("retrieval_actors") or []]]:
                actor_id = name if name in actor_refs else actor_by_name.get(str(name).casefold())
                if actor_id and actor_id not in ids:
                    ids.append(actor_id)
                    actor_refs[actor_id].append(identifier)
            paths = e.get("source_sections") or []
            quote = str(e.get("evidence_text") or "")
            references = []
            for s in source_records:
                if paths and not any(str(p).split("/", 1)[0] == s["alias"] for p in paths):
                    continue
                span=locate_quote(s["text"],quote) if quote else None
                start=span[0] if span else -1
                if source_records and not paths and start < 0:
                    continue
                references.append({"id": f"e{len(references)+1}", "resource": "/" + s["path"],
                                   "source_id": s["source_id"], "snapshot_sha256": s["snapshot"],
                                   "section_paths": [p for p in paths if str(p).startswith(s["alias"] + "/")],
                                   "span_start": start if start >= 0 else None,
                                   "span_end": span[1] if span else None,
                                   "quote": s["text"][span[0]:span[1]] if span else "", "url": s["url"],
                                   "attribution": "exact_span" if start >= 0 else "page_only"})
            exact = any(r["attribution"] == "exact_span" for r in references)
            metadata = {"type": "Historical Event", "title": e["description"][:100],
                        "description": e["description"], "case_id": case_id, "timeline_id": identifier,
                        "occurred_month": month(e["timestamp"]) if identifier in supported_by_id else None,
                        "timestamp":e["timestamp"] if identifier in supported_by_id else "",
                        "proposed_timestamp":e["timestamp"] if identifier not in supported_by_id else "",
                        "date_precision": e.get("date_precision", "month"), "actor_ids": ids,
                        "actor_names":{a["agent_id"]:a.get("name",a["agent_id"]) for a in actors if a["agent_id"] in ids},
                        "time_role":e.get("time_role","occurrence"),"modality":e.get("modality","occurred"),
                        "date_evidence":e.get("date_evidence",{}),
                        "date_validation":{"status":"supported" if identifier in supported_by_id else "needs_review","reason":rejected_by_id.get(identifier,"")},
                        "actions": e.get("actions") or [], "origin": "historical",
                        "source_sections": paths,
                        "sources": [{"id": r["id"], "resource": r["resource"]} for r in references],
                        "evidence_refs": references, "status": "stable" if exact else "draft",
                        "generated": {"by": "process:timeline-export", "at": now}}
            body = f"# {e['description']}\n\nDate: {metadata['timestamp'] or 'not established; original proposed date is not accepted'}\n\n"
            body += "\n".join(f"- [Actor](/{'/'.join([root_path, 'actors', digest(a)[:24] + '.md'])})" for a in ids)
            for r in references:
                body += f"\n\nEvidence [{r['attribution']}]: {r['quote']}[^{r['id']}]\n\n[^{r['id']}]: {r['resource']}"
            _document(self.root / path, metadata, body)
            docs.append({"id": identifier, "type": "timeline", "path": path, "metadata": metadata, "body": body})
        for a in actors:
            identifier = a["agent_id"]
            path = f"{root_path}/actors/{digest(identifier)[:24]}.md"
            refs = list(dict.fromkeys(actor_refs[identifier]))
            event_map={d["id"]:d["metadata"] for d in docs if d["type"]=="timeline"}
            related=[event_map[r] for r in refs if r in event_map]
            related.sort(key=lambda e:e["timestamp"],reverse=True)
            dated=[e for e in related if e["date_validation"]["status"]=="supported"]
            background=[{"timeline_id":e["timeline_id"],"description":e["description"],"reason":e["date_validation"]["reason"]} for e in related if e["date_validation"]["status"]!="supported"]
            activities=[{"timeline_id":e["timeline_id"],"timestamp":e["timestamp"],"description":e["description"],"date_validation":e["date_validation"]} for e in dated[:8]]
            evidence=[]
            seen_evidence=set()
            for event in related:
                for ref in event["evidence_refs"]:
                    key=(ref["source_id"],ref["span_start"],ref["span_end"])
                    if key not in seen_evidence: evidence.append(ref);seen_evidence.add(key)
            card={**a,"relevant_timeline_refs":refs,"observed_activities":activities,
                  "event_count":len(dated),"background_records":background,"pending_count":len(background),
                  "source_count":len({(ref["source_id"],ref["snapshot_sha256"]) for ref in evidence}),
                  "date_range":[min((e["timestamp"] for e in dated),default=""),max((e["timestamp"] for e in dated),default="")]}
            if not card.get("description"):
                card["description"]=a.get("role_description") or (related[0]["description"] if related else "No activities with traceable sources are recorded.")
            card["activity_summary"]=f"Linked to {len(dated)} dated events and {len(background)} background records"
            metadata = {"type": "Actor Profile", "title": a.get("name", identifier),
                        "description": card["description"], "case_id": case_id,
                        "actor_id": identifier, "aliases": a.get("aliases") or [],
                        "simulation_agent": identifier in known, "timeline_ids": refs,
                        "profile": card,"evidence_refs":evidence[:8], "status": "draft", "generated": {"by": "process:actor-export", "at": now},
                        "sources": [{"id": f"t{i}", "resource": f"/{root_path}/timeline/{digest(ref)[:24]}.md"}
                                    for i, ref in enumerate(refs)]}
            body = f"# {metadata['title']}\n\n{metadata['description']}\n\n" + "\n".join(
                f"- [Historical event](/{'/'.join([root_path, 'timeline', digest(ref)[:24] + '.md'])})" for ref in refs)
            _document(self.root / path, metadata, body)
            docs.append({"id": identifier, "type": "actor", "path": path, "metadata": metadata, "body": body})
        # Chapters can recall linked dated events and retain separate, dated
        # source material for decision-time context.
        background_by_section = {}
        for item in retained_facts:
            for path in dict.fromkeys(item["source_sections"]):
                background_by_section.setdefault(path, []).append(item)
        background_ids = set()
        for section in sections:
            source = next((s for s in source_records if s["url"] == section["source_url"]
                           and section["path"].startswith(s["alias"] + "/")), None)
            if not source:
                continue
            identifier = "section_" + digest({"path": section["path"], "section_id": section["section_id"], "source": source["snapshot"]})[:24]
            path = f"{root_path}/sections/{identifier}.md"
            linked = [d["id"] for d in docs if d["type"] == "timeline"
                      and section["path"] in d["metadata"].get("source_sections", [])]
            start = source["text"].find(section["content"])
            exact = start >= 0 and source["text"].rfind(section["content"]) == start
            metadata = {"type": "Source Section", "title": section["title"],
                        "description": section["path"], "case_id": case_id,
                        "section_id": identifier, "section_path": section["path"],
                        "timeline_ids": linked, "resource": source["url"],
                        "source_id": source["source_id"], "snapshot_sha256": source["snapshot"],
                        "source_revision_id": source["revision_id"], "source_revision_at": source["revision_at"],
                        "source_retrieved_at": source["retrieved_at"], "source_sdk_fallback": source["sdk_fallback"],
                        "source_span_start": start if exact else None,
                        "status": "source_material", "date_validation": {"status": "not_an_event"},
                        "generated": {"by": "process:section-export", "at": now}}
            _document(self.root / path, metadata, section["content"])
            docs.append({"id": identifier, "type": "section", "path": path,
                         "metadata": metadata, "body": section["content"]})
            if not exact:
                continue
            for item in background_by_section.get(section["path"], []):
                span = locate_quote(section["content"], item["quote"])
                if not span:
                    continue
                absolute = start + span[0]
                quote = section["content"][span[0]:span[1]]
                fact_id = "background_" + digest([source["source_id"], source["snapshot"], absolute, quote])[:24]
                if fact_id in background_ids:
                    continue
                background_ids.add(fact_id)
                fact_path = f"{root_path}/background/{fact_id}.md"
                fact = {key: metadata[key] for key in (
                    "case_id", "section_path", "resource", "source_id", "snapshot_sha256",
                    "source_revision_id", "source_revision_at", "source_retrieved_at",
                )}
                fact.update({key: item[key] for key in (
                    "fact_status", "date_original", "date_precision", "reported_bounds", "reason",
                )})
                fact.update(type="Historical Background", title=section["title"],
                            source_span_start=absolute, source_span_end=start + span[1],
                            section_id=identifier, status="background_only", quote=quote)
                _document(self.root / fact_path, fact, quote)
                docs.append({"id": fact_id, "type": "background", "path": fact_path,
                             "metadata": fact, "body": quote})
        manifest = {"schema_version":2,"projection_version":PROJECTION_VERSION,"case_id": case_id, "version": version,
                    "evidence_policy":extraction.get("evidence_policy","legacy_unchecked"),
                    "requires_reanalysis":requires_reanalysis, "title": case.get("title", ""),
                    "created_at": now, "timeline_count": len(history), "actor_count": len(actors),
                    "source_count": len(source_records), "section_count": sum(d["type"] == "section" for d in docs),
                    "background_count": len(background_ids),
                    "documents": [{k: d[k] for k in ("id", "type", "path")} for d in docs]}
        release.mkdir(parents=True, exist_ok=True)
        (release / "index.md").write_text("# " + manifest["title"] + "\n\n" + "\n".join(
            f"- [{d['metadata']['title']}](/{'/'.join(d['path'].split('/'))})" for d in docs), encoding="utf-8")
        write_json(release / "manifest.json", manifest)
        self._index(case_id, version, docs)
        self._activate(case_id, manifest)
        return manifest

    def _activate(self, case_id: str, manifest: dict):
        write_json(self._case(case_id) / "current.json", manifest)
        (self._case(case_id) / "index.md").write_text(
            f"# {manifest['title']}\n\n[Current knowledge](versions/{manifest['version']}/index.md)\n", encoding="utf-8")
        (self.root / "index.md").write_text('---\nokf_version: "0.2"\n---\n\n# Knowledge\n\n' + "\n".join(
            f"- [{p.parent.name}](cases/{p.parent.name}/index.md)" for p in sorted((self.root / "cases").glob("*/current.json"))), encoding="utf-8")

    def _connection(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS docs (case_id TEXT, version TEXT, id TEXT, kind TEXT, path TEXT, title TEXT, body TEXT, month TEXT, metadata TEXT, PRIMARY KEY(case_id, version, id))")
        db.execute("CREATE INDEX IF NOT EXISTS doc_filters ON docs(case_id,version,kind,month)")
        # Chinese character substrings and English prose. Very short names are
        # also matched by exact aliases below; they need not be trigram tokens.
        try:
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(title, body, tokenize='trigram')")
        except sqlite3.OperationalError:
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(title, body)")
        return db

    def _index(self, case_id: str, version: str, documents: list[dict]):
        with self._connection() as db:
            old = db.execute("SELECT rowid FROM docs WHERE case_id=? AND version=?", (case_id, version)).fetchall()
            for row in old:
                db.execute("DELETE FROM search_fts WHERE rowid=?", (row[0],))
            db.execute("DELETE FROM docs WHERE case_id=? AND version=?", (case_id, version))
            for d in documents:
                m = d["metadata"]
                cursor = db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?)", (
                    case_id, version, d["id"], d["type"], d["path"], m["title"], d["body"],
                    m.get("occurred_month", ""), json.dumps(m, ensure_ascii=False)))
                db.execute("INSERT INTO search_fts(rowid,title,body) VALUES(?,?,?)", (cursor.lastrowid, m["title"], d["body"]))

    @knowledge_mutation
    def rebuild(self, case_id: str, version: str | None = None) -> dict:
        manifest = self.manifest(case_id)
        if version is None and manifest.get("schema_version")!=2: manifest=self.publish(case_id)
        version = version or manifest.get("version")
        if not version or not re.fullmatch(r"[0-9a-f]{24}", version):
            raise ValueError("Invalid or missing knowledge version")
        manifest = read_json(self._case(case_id) / "versions" / version / "manifest.json", {})
        docs = []
        for entry in manifest.get("documents", []):
            metadata, body = _read_document(self.root / entry["path"])
            docs.append(entry | {"metadata": metadata, "body": body})
        self._index(case_id, version, docs)
        return manifest

    def snapshot(self, case_id: str, *, version: str | None = None) -> tuple[str, list[dict]]:
        """Read one release once. The caller may freeze it without future disk access."""
        manifest = self.manifest(case_id)
        if version is None and (manifest.get("schema_version") != 2 or manifest.get("projection_version") != PROJECTION_VERSION):
            manifest = self.publish(case_id)
        version = version or manifest.get("version")
        if not version:
            manifest = self.publish(case_id)
            version = manifest["version"]
        if not re.fullmatch(r"[0-9a-f]{24}", version):
            raise ValueError("Invalid knowledge version")
        self.manifest(case_id, version=version)
        with self._connection() as db:
            if not db.execute("SELECT 1 FROM docs WHERE case_id=? AND version=? LIMIT 1", (case_id, version)).fetchone():
                self.rebuild(case_id, version)
            rows = db.execute("SELECT id,kind,path,title,body,metadata FROM docs WHERE case_id=? AND version=? ORDER BY id",
                              (case_id, version)).fetchall()
        return version, [{"id": r["id"], "kind": r["kind"], "path": r["path"], "title": r["title"],
                          "body": r["body"], "metadata": json.loads(r["metadata"])} for r in rows]

    def search(self, case_id: str, query: str = "", *, kind: str = "all", actor_id: str = "",
               from_month: str = "", to_month: str = "", limit: int = 30, version: str | None = None) -> list[dict]:
        version, documents = self.snapshot(case_id, version=version)
        return EvidenceIndex(documents, version=version).search(query, kind=kind, actor_id=actor_id,
            from_month=from_month, to_month=to_month, limit=limit)

    def record(self, case_id: str, record_id: str, *, version: str | None = None) -> dict:
        version, _ = self.snapshot(case_id, version=version)
        with self._connection() as db:
            row=db.execute("SELECT id,kind,path,title,metadata FROM docs WHERE case_id=? AND version=? AND id=?",(case_id,version,record_id)).fetchone()
        if row is None: raise FileNotFoundError("Knowledge record not found")
        return {"id":row["id"],"kind":row["kind"],"path":row["path"],"title":row["title"],"metadata":json.loads(row["metadata"]),"version":version}

    def graph(self, case_id: str, query: str = "", limit: int = 30, *, view: str = "evidence", focus_id: str = "", version: str | None = None, **filters) -> dict:
        version, _ = self.snapshot(case_id, version=version)
        if view=="actors":
            profiles=self.search(case_id,version=version,kind="actor",limit=100000)
            matched={p["id"] for p in self.search(case_id,query,version=version,kind="actor",limit=100000)} if query else None
            if filters.get("from_month") or filters.get("to_month"):
                dated={actor for hit in self.search(case_id,version=version,kind="timeline",limit=100000,**filters) for actor in hit["metadata"]["actor_ids"]}
                matched = dated if matched is None else matched & dated
            result=actor_graph(profiles,case_id,self.manifest(case_id, version=version).get("title",""),actor_id=filters.get("actor_id",""),match_ids=matched,limit=min(100,limit))
            return result|{"version":version,"view":view}
        hits = self.search(case_id, query, version=version, kind="timeline", limit=100000 if focus_id else limit, **filters)
        if view=="evidence": hits=[h for h in hits if h["id"]==focus_id] if focus_id else hits[:1]
        profiles = {p["id"]: p for p in self.search(case_id, version=version, kind="actor", limit=100000)}
        nodes, edges = {}, []
        for hit in hits:
            event = hit["metadata"]
            nodes[hit["id"]] = {"id": hit["id"], "label": hit["title"], "type": "timeline", "detail": event}
            for actor in event["actor_ids"]:
                profile = profiles.get(actor)
                nodes[actor] = {"id": actor, "label": profile["title"] if profile else actor, "type": "actor", "detail": profile["metadata"] if profile else {}}
                edges.append({"source": hit["id"], "target": actor, "label": "participant"})
            for ref in event["evidence_refs"]:
                key = ref["source_id"] + ":" + ref["snapshot_sha256"]
                nodes[key] = {"id": key, "label": ref["url"], "type": "source", "detail": ref}
                edges.append({"source": hit["id"], "target": key, "label": ref["attribution"]})
        if view=="timeline":
            nodes={k:v for k,v in nodes.items() if v["type"]!="source"}
            edges=[e for e in edges if e["source"] in nodes and e["target"] in nodes]
        total_nodes=len(nodes)
        keep=set(list(nodes)[:max(1,limit)])
        return {"nodes": [v for k,v in nodes.items() if k in keep], "edges": [e for e in edges if e["source"] in keep and e["target"] in keep], "version": version,"view":view,"truncated":total_nodes>len(keep),"total_nodes":total_nodes}
