"""Recoverable deletion of one local case and its isolated experiments."""
from __future__ import annotations

import os
import hashlib
import re
import shutil
import uuid
from datetime import datetime, timezone

from app.knowledge.okf import KnowledgeStore, _read_document, digest
from app.storage.json_store import BENCHMARK_EVENT_ORIGIN, EventRepository, read_json, write_json
from app.storage.knowledge_lock import knowledge_mutation


class AnalysisTrash:
    _LOG_CATEGORIES = ("request_logs", "event_logs", "graph_logs", "knowledge_runs", "forecast_runs", "process_logs")

    def __init__(self, events: EventRepository | None = None):
        self.events = events or EventRepository()
        self.data = self.events.root.parent
        self.root = self.data / "analysis_trash"
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self):
        entries = (read_json(path, {}) for path in self.root.glob("*/trash.json"))
        return sorted((entry for entry in entries
                       if entry.get("status") in {"deleted", "purging"} and entry.get("origin") != BENCHMARK_EVENT_ORIGIN),
                      key=lambda entry: entry.get("deleted_at", ""), reverse=True)

    @knowledge_mutation
    def delete(self, case_id: str, active_runs=()):
        directory = self.events.directory(case_id)
        event = read_json(directory / "event.json", {})
        if event.get("origin") == BENCHMARK_EVENT_ORIGIN:
            raise ValueError("This preparation belongs to a Market Evaluation.")
        plans = [(directory, "analysis")]
        knowledge = KnowledgeStore(self.events)
        case_knowledge = knowledge.root / "cases" / case_id
        if case_knowledge.exists():
            plans.append((case_knowledge, "knowledge"))
        experiments = []
        for path in (self.data / "ensemble_runs").glob("*/config.json"):
            config = read_json(path, {})
            if config.get("request", {}).get("event_id") != case_id:
                continue
            identifier = path.parent.name
            running = identifier in active_runs
            lock = path.parent / ".running"
            if lock.exists():
                try:
                    os.kill(int(lock.read_text()), 0)
                    running = True
                except (ProcessLookupError, ValueError):
                    pass
            if running:
                raise ValueError("Experiments are still running for this analysis. Stop them and save progress first.")
            experiments.append(identifier)
            plans.append((path.parent, f"experiments/{identifier}"))
        trash_id = str(uuid.uuid4())
        target = self.root / trash_id
        target.mkdir()
        manifest = {"trash_id": trash_id, "case_id": case_id, "title": event.get("title", case_id),
                    "deleted_at": datetime.now(timezone.utc).isoformat(), "status": "moving",
                    "experiment_ids": experiments,
                    "moves": [{"original": p.relative_to(self.data).as_posix(), "stored": name} for p, name in plans]}
        write_json(target / "trash.json", manifest)
        moved = []
        try:
            for source, name in plans:
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
                moved.append((source, destination))
            with knowledge._connection() as db:
                rows = db.execute("SELECT rowid FROM docs WHERE case_id=?", (case_id,)).fetchall()
                for row in rows:
                    db.execute("DELETE FROM search_fts WHERE rowid=?", (row[0],))
                db.execute("DELETE FROM docs WHERE case_id=?", (case_id,))
            manifest["status"] = "deleted"
            write_json(target / "trash.json", manifest)
            self._refresh_root(knowledge)
            return manifest
        except Exception:
            for original, stored in reversed(moved):
                original.parent.mkdir(parents=True, exist_ok=True)
                stored.rename(original)
            manifest["status"] = "rolled_back"
            write_json(target / "trash.json", manifest)
            raise

    @knowledge_mutation
    def restore(self, trash_id: str):
        directory = self._entry_directory(trash_id)
        manifest = read_json(directory / "trash.json", {})
        if manifest.get("status") == "purging":
            raise ValueError("Permanent deletion has already started. Retry the deletion to finish it.")
        if manifest.get("status") != "deleted":
            raise FileNotFoundError("This analysis is no longer in Trash. It may have been restored.")
        plans = [(self.data / m["original"], directory / m["stored"]) for m in manifest["moves"]]
        for original, stored in plans:
            if self.data.resolve() not in original.resolve().parents or directory.resolve() not in stored.resolve().parents:
                raise ValueError("Invalid recycle-bin path")
            if original.exists() or not stored.exists():
                raise ValueError("Restore failed because the destination is occupied or the saved data is missing. The existing data was preserved.")
        restored = []
        try:
            for original, stored in plans:
                original.parent.mkdir(parents=True, exist_ok=True)
                stored.rename(original)
                restored.append((original, stored))
            knowledge = KnowledgeStore(self.events)
            if knowledge.manifest(manifest["case_id"]):
                knowledge.rebuild(manifest["case_id"])
            self._refresh_root(knowledge)
            manifest["status"] = "restored"
            manifest["restored_at"] = datetime.now(timezone.utc).isoformat()
            write_json(directory / "trash.json", manifest)
            return manifest
        except Exception:
            for original, stored in reversed(restored):
                stored.parent.mkdir(parents=True, exist_ok=True)
                original.rename(stored)
            raise

    def _entry_directory(self, trash_id: str):
        try:
            if str(uuid.UUID(trash_id)) != trash_id:
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError("Invalid recycle-bin entry") from None
        directory = self.root / trash_id
        if directory.is_symlink():
            raise ValueError("Invalid recycle-bin path")
        return directory

    @staticmethod
    def _source_snapshots(root):
        """Include every retained knowledge version and unpublished source body."""
        snapshots = set()
        for path in root.rglob("*.md"):
            if not path.read_text(encoding="utf-8").startswith("---\n"):
                continue
            metadata, _ = _read_document(path)
            for ref in [metadata, *(metadata.get("evidence_refs") or [])]:
                source, snapshot = ref.get("source_id", ""), ref.get("snapshot_sha256", "")
                if re.fullmatch(r"src_[0-9a-f]{20}", source) and re.fullmatch(r"[0-9a-f]{64}", snapshot):
                    snapshots.add((source, snapshot))
        for path in root.rglob("sources.json"):
            stored = read_json(path, {})
            sources = stored if isinstance(stored, list) else stored.get("sources", [])
            for source in sources:
                fragments = source.get("sections") if source.get("sdk_fallback") else None
                if source.get("sdk_fallback") and not fragments:
                    number = str(source.get("source_id", "")).rsplit("_", 1)[-1]
                    if number.isdigit():
                        fragments = read_json(path.parent / "sources" / f"{int(number):02d}_source" / "sections.json", [])
                for item in fragments or [source]:
                    body = str(item.get("content") or "")
                    if not body and item.get("body_file"):
                        body_path = path.parent / item["body_file"]
                        if path.parent.resolve() not in body_path.resolve().parents:
                            raise ValueError("Invalid source body path")
                        body = body_path.read_text(encoding="utf-8")
                    if body:
                        url = str(item.get("url") or source.get("url") or "")
                        snapshots.add(("src_" + digest(url or body)[:20], hashlib.sha256(body.encode()).hexdigest()))
        return snapshots

    def _related_logs(self, case_id, experiment_ids):
        """Match explicit ownership and inherited request IDs, never topic text."""
        from app.storage.run_logger import current_request_run_id
        records = []
        owned_runs = set(experiment_ids)
        identifiers = {case_id, *experiment_ids}
        for category in self._LOG_CATEGORIES:
            for folder in (self.data / category).glob("*"):
                if not folder.is_dir() or folder.is_symlink():
                    continue
                owned = False
                run_ids = set()
                # Run-started, configuration/HTTP request are the first files;
                # subsequent files can duplicate very large model contexts.
                for path in sorted(folder.glob("*.json"))[:3]:
                    record = read_json(path, {})
                    metadata = record.get("metadata") or {}
                    data = record.get("data") or {}
                    if metadata.get("run_id"):
                        run_ids.add(metadata["run_id"])
                    if metadata.get("event_id") == case_id:
                        owned = True
                    if isinstance(data, dict):
                        body = data.get("body") or {}
                        if data.get("event_id") == case_id or (isinstance(body, dict) and body.get("event_id") == case_id):
                            owned = True
                        if identifiers.intersection(str(data.get("path", "")).split("/")):
                            owned = True
                if current_request_run_id() in run_ids:
                    continue
                records.append((folder, run_ids, owned))
                if owned:
                    owned_runs.update(run_ids)
        return [folder.relative_to(self.data).as_posix() for folder, runs, owned in records
                if owned or runs.intersection(owned_runs)]

    @knowledge_mutation
    def purge(self, trash_id: str):
        directory = self._entry_directory(trash_id)
        manifest_path = directory / "trash.json"
        manifest = read_json(manifest_path, {})
        if manifest.get("origin") == BENCHMARK_EVENT_ORIGIN:
            raise ValueError("This preparation belongs to a Market Evaluation.")
        if manifest.get("status") not in {"deleted", "purging"}:
            raise FileNotFoundError("This analysis is no longer in Trash. It may have been restored or permanently deleted.")
        # Retain a retry journal until every file has been removed. A partially
        # deleted entry stays visible, but can never be restored as complete.
        case_id = manifest["case_id"]
        if str(uuid.UUID(case_id)) != case_id:
            raise ValueError("Invalid case identifier")
        candidates = (set(map(tuple, manifest.get("source_snapshots", [])))
                      if manifest["status"] == "purging" else self._source_snapshots(directory))
        log_paths = set(manifest.get("log_paths", [])) | set(self._related_logs(case_id, manifest.get("experiment_ids", [])))
        logs = []
        for relative in log_paths:
            path = self.data / relative
            if (len(path.relative_to(self.data).parts) != 2 or path.relative_to(self.data).parts[0] not in self._LOG_CATEGORIES
                    or self.data.resolve() not in path.resolve().parents or path.is_symlink()):
                raise ValueError("Invalid analysis log path")
            logs.append(path)
        knowledge = KnowledgeStore(self.events)
        protected = self._source_snapshots(self.events.root) | self._source_snapshots(knowledge.root / "cases")
        for other in self.root.iterdir():
            if other != directory and other.is_dir():
                other_manifest = read_json(other / "trash.json", {})
                protected |= (set(map(tuple, other_manifest.get("source_snapshots", [])))
                              if other_manifest.get("status") == "purging" else self._source_snapshots(other))
        unreferenced = candidates - protected
        targets = []
        for source, snapshot in unreferenced:
            if not re.fullmatch(r"src_[0-9a-f]{20}", source) or not re.fullmatch(r"[0-9a-f]{64}", snapshot):
                raise ValueError("Invalid source snapshot")
            for folder, suffix in (("references", ".txt"), ("sources", ".md")):
                path = knowledge.root / folder / source / f"{snapshot}{suffix}"
                if knowledge.root.resolve() not in path.resolve().parents:
                    raise ValueError("Invalid source snapshot path")
                targets.append(path)
        manifest.update(status="purging", source_snapshots=sorted(candidates), log_paths=sorted(log_paths))
        write_json(manifest_path, manifest)
        # Soft deletion already removed search rows. Clear any stale rows too,
        # without touching an independently recreated active case.
        if not (self.events.root / case_id).exists() and not (knowledge.root / "cases" / case_id).exists():
            with knowledge._connection() as db:
                rows = db.execute("SELECT rowid FROM docs WHERE case_id=?", (case_id,)).fetchall()
                for row in rows:
                    db.execute("DELETE FROM search_fts WHERE rowid=?", (row[0],))
                db.execute("DELETE FROM docs WHERE case_id=?", (case_id,))
        for path in targets:
            path.unlink(missing_ok=True)
            if path.parent.exists() and not any(path.parent.iterdir()):
                path.parent.rmdir()
        for child in directory.iterdir():
            if child == manifest_path:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for path in logs:
            if path.exists():
                shutil.rmtree(path)
        self._refresh_root(knowledge)
        manifest_path.unlink()
        directory.rmdir()
        return {"trash_id": trash_id, "case_id": case_id, "status": "purged"}

    @staticmethod
    def _refresh_root(knowledge):
        (knowledge.root / "index.md").write_text('---\nokf_version: "0.2"\n---\n\n# Knowledge\n\n' + "\n".join(
            f"- [{p.parent.name}](cases/{p.parent.name}/index.md)"
            for p in sorted((knowledge.root / "cases").glob("*/current.json"))), encoding="utf-8")
