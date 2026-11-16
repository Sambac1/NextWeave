from __future__ import annotations

import json
import hashlib
import re
import uuid
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import get_settings
from app.storage.run_logger import TraceRecorder


_UNSAFE = re.compile(r'[^\w\-.]+', re.UNICODE)
BENCHMARK_EVENT_ORIGIN = "market_benchmark"


def safe_name(value: str, default_name: str = "item") -> str:
    cleaned = _UNSAFE.sub("_", value.strip()).strip("._")
    return cleaned[:120] or default_name


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Stored JSON is unreadable: {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


class EventRepository:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.root = (data_dir or get_settings().data_dir) / "events"
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, title: str, urls: list[str], additional_context: str | None, *, origin: str | None = None) -> dict[str, Any]:
        event_id = str(uuid.uuid4())
        event_dir = self.root / event_id
        event_dir.mkdir(parents=True, exist_ok=False)
        event = {
            "event_id": event_id,
            "title": title,
            "urls": urls,
            "additional_context": additional_context or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if origin is not None:
            event["origin"] = origin
        write_json(event_dir / "event.json", event)
        write_json(event_dir / "graph.json", {"nodes": [], "edges": []})
        return event

    def directory(self, event_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", event_id):
            raise ValueError("Invalid event_id")
        path = self.root / event_id
        if not path.exists():
            raise FileNotFoundError(event_id)
        return path

    def save_artifact(self, event_id: str, name: str, value: Any) -> Path:
        path = self.directory(event_id) / f"{safe_name(name)}.json"
        write_json(path, value)
        return path

    def load_artifact(self, event_id: str, name: str, default: Any) -> Any:
        return read_json(self.directory(event_id) / f"{safe_name(name)}.json", default)

    def save_sources(self, event_id: str, sources: list[dict[str, Any]]) -> None:
        """Store one canonical body file per source and a compact JSON index."""
        event_dir = self.directory(event_id)
        records: list[dict[str, Any]] = []
        for number, source in enumerate(sources, 1):
            source_id = f"source_{number:02d}"
            relative_dir = Path("sources") / f"{number:02d}_source"
            source_dir = event_dir / relative_dir
            body = str(source.get("content") or "")
            if body:
                _write_text(source_dir / "content.txt", body)
            else:
                legacy_body = source_dir / "content.txt"
                if legacy_body.is_file():
                    body = legacy_body.read_text(encoding="utf-8")
            body_bytes = body.encode("utf-8")
            record = {
                key: deepcopy(value)
                for key, value in source.items()
                if key not in {"content", "sections"}
            }
            record.update({
                "source_id": source_id,
                "body_file": (relative_dir / "content.txt").as_posix() if body else None,
                "body_sha256": hashlib.sha256(body_bytes).hexdigest() if body else None,
                "body_char_count": len(body),
            })
            # Fallback sections quote passages from separate URLs that may not
            # occur in the synthesized page body; retain those cited excerpts.
            if source.get("sdk_fallback") and source.get("sections"):
                write_json(source_dir / "sections.json", source["sections"])
            else:
                sections_file = source_dir / "sections.json"
                if sections_file.exists():
                    sections_file.unlink()
            records.append(record)
        write_json(event_dir / "sources.json", {"schema_version": 2, "sources": records})

    def load_sources(self, event_id: str) -> list[dict[str, Any]]:
        """Load the source corpus and migrate stored inline JSON on first read."""
        event_dir = self.directory(event_id)
        stored = read_json(event_dir / "sources.json", None)
        if isinstance(stored, list):
            self.save_sources(event_id, stored)
            stored = read_json(event_dir / "sources.json", {})
        elif (
            isinstance(stored, dict)
            and isinstance(stored.get("sources"), list)
            and any(isinstance(item, dict) and "content" in item for item in stored["sources"])
        ):
            # Also normalize transitional object-wrapped indexes that still
            # embed bodies, not just the original top-level list format.
            self.save_sources(event_id, stored["sources"])
            stored = read_json(event_dir / "sources.json", {})
        if isinstance(stored, dict) and isinstance(stored.get("sources"), list):
            records = stored["sources"]
        else:
            # Recover bodies saved before an interrupted source-index write.
            records = []
            source_root = event_dir / "sources"
            if source_root.is_dir():
                for number, folder in enumerate(sorted(source_root.iterdir()), 1):
                    body_file = folder / "content.txt"
                    if not folder.is_dir() or not body_file.is_file():
                        continue
                    body = body_file.read_text(encoding="utf-8")
                    records.append({
                        "source_id": f"source_{number:02d}",
                        "body_file": body_file.relative_to(event_dir).as_posix(),
                        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                        "body_char_count": len(body),
                        "title": "", "url": "", "summary": "", "domain": "",
                    })
        sources: list[dict[str, Any]] = []
        for record in records:
            source = deepcopy(record)
            body_file = source.pop("body_file", None)
            expected_hash = source.pop("body_sha256", None)
            source.pop("body_char_count", None)
            source_id = str(source.get("source_id") or "")
            if body_file:
                candidate = (event_dir / body_file).resolve()
                if event_dir.resolve() not in candidate.parents:
                    raise ValueError("Source body path escapes its event directory")
                if not candidate.is_file():
                    raise ValueError(f"Stored source body is missing: {source_id}")
                body = candidate.read_text(encoding="utf-8")
                if expected_hash and hashlib.sha256(body.encode("utf-8")).hexdigest() != expected_hash:
                    raise ValueError(f"Stored source body failed integrity check: {source_id}")
                source["content"] = body
            if source.get("sdk_fallback"):
                number = source_id.rsplit("_", 1)[-1]
                folder_name = f"{int(number):02d}_source" if number.isdigit() else ""
                sections_path = event_dir / "sources" / folder_name / "sections.json"
                if sections_path.is_file():
                    source["sections"] = read_json(sections_path, [])
            sources.append(source)
        return sources

    def find_by_title(self, title: str) -> dict[str, Any] | None:
        normalized = title.strip().casefold()
        matches: list[tuple[float, dict[str, Any]]] = []
        for event_file in self.root.glob("*/event.json"):
            event = read_json(event_file, {})
            if str(event.get("title", "")).strip().casefold() == normalized:
                matches.append((event_file.stat().st_mtime, event))
        return max(matches, default=(0, None), key=lambda item: item[0])[1]


class GraphRepository:
    def __init__(self, events: EventRepository | None = None) -> None:
        self.events = events or EventRepository()

    def _path(self, event_id: str) -> Path:
        return self.events.directory(event_id) / "graph.json"

    def get(self, event_id: str) -> dict[str, list[dict[str, Any]]]:
        return read_json(self._path(event_id), {"nodes": [], "edges": []})

    def add_node(self, event_id: str, node: dict[str, Any]) -> dict[str, Any]:
        graph = self.get(event_id)
        payload = {"data": node}
        existing = next((item for item in graph["nodes"] if item.get("data", {}).get("id") == node["id"]), None)
        if existing:
            existing["data"].update(node)
            payload = existing
        else:
            graph["nodes"].append(payload)
        write_json(self._path(event_id), graph)
        return payload

    def add_edge(self, event_id: str, edge: dict[str, Any]) -> dict[str, Any]:
        graph = self.get(event_id)
        source, target = edge["source"], edge["target"]
        label = edge.get("label", "")
        edge_type = edge.get("edge_type", "relationship")
        existing = next(
            (
                item
                for item in graph["edges"]
                if item.get("data", {}).get("source") == source
                and item.get("data", {}).get("target") == target
                and item.get("data", {}).get("label") == label
                and item.get("data", {}).get("edge_type", "relationship") == edge_type
            ),
            None,
        )
        if existing:
            existing["data"].update(edge)
            payload = existing
        else:
            edge.setdefault("id", str(uuid.uuid4()))
            payload = {"data": edge}
            graph["edges"].append(payload)
        write_json(self._path(event_id), graph)
        return payload

    def remove_unregistered_agents(self, event_id: str, registered_agent_ids: set[str]) -> list[str]:
        """Remove Agent nodes rejected by a new Registry result and their incident edges."""
        graph = self.get(event_id)
        removed_ids = {
            str(item.get("data", {}).get("id") or "")
            for item in graph["nodes"]
            if item.get("data", {}).get("type") == "agent"
            and str(item.get("data", {}).get("id") or "") not in registered_agent_ids
        }
        if not removed_ids:
            return []
        graph["nodes"] = [
            item for item in graph["nodes"]
            if str(item.get("data", {}).get("id") or "") not in removed_ids
        ]
        graph["edges"] = [
            item for item in graph["edges"]
            if str(item.get("data", {}).get("source") or "") not in removed_ids
            and str(item.get("data", {}).get("target") or "") not in removed_ids
        ]
        write_json(self._path(event_id), graph)
        return sorted(removed_ids)


class SnapshotRepository:
    """Committed single-path state. Branch candidates are never written here."""

    def __init__(self, events: EventRepository | None = None) -> None:
        self.events = events or EventRepository()

    def get(self, event_id: str) -> dict[str, Any] | None:
        return self.events.load_artifact(event_id, "latest_snapshot", None)

    _commit_lock = threading.RLock()

    def commit(self, event_id: str, snapshot: dict[str, Any], *, expected_version: int) -> None:
        """Atomically reject a commit when another forecast advanced the snapshot."""
        with self._commit_lock:
            current = self.get(event_id)
            current_version = int(current.get("snapshot_version") or 0) if isinstance(current, dict) else 0
            if current_version != expected_version:
                raise RuntimeError(
                    f"Simulation snapshot changed before commit: expected {expected_version}, current {current_version}."
                )
            self.events.save_artifact(event_id, "latest_snapshot", snapshot)

    def clear(self, event_id: str) -> None:
        with self._commit_lock:
            path = self.events.directory(event_id) / "latest_snapshot.json"
            if path.exists():
                path.unlink()
