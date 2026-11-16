"""Shared helpers for persisted LLM event artifacts.

Artifacts live in the framework's event store (``EventRepository``, keyed by
``event_id``) rather than the old per-title data directory. The ported modules
address events by title, so this module resolves the title to a repository
handle and keeps the path-shaped call sites working.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.layers.support.llm_normalizers import _clean_text
from app.storage.json_store import read_json


def _clip_text(value: object, limit: int = 700) -> str:
    text = _clean_text(value)
    return text[:limit] + "..." if len(text) > limit else text


def _copy_dict_list(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    copied: list[dict] = []
    for item in value:
        if isinstance(item, dict):
            copied.append(dict(item))
    return copied


def _artifact_folder_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _events_repository():
    """The event store. Imported lazily so this module stays dependency-light."""
    from app.storage.json_store import EventRepository

    return EventRepository()


def _title_key(value: object) -> str:
    """Comparison key for event titles.

    Titles reach the pipeline from several places (the create request, the
    frontend, an artifact) and their punctuation drifts -- an en dash in the
    Wikipedia title against a hyphen in a hand-typed one, spaces against
    underscores, letter case. Folding all of that away lets a title still
    resolve to its event; matching on the raw string would quietly return
    nothing and the caller would treat "not found" as "no material".
    """
    text = _clean_text(value).casefold()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _event_id_for_title(event_title: str) -> str | None:
    """Resolve an event title to its ``event_id``, or None when unknown.

    Tries the store's exact match first, then a punctuation-insensitive one.
    """
    title = _clean_text(event_title)
    if not title:
        return None
    repository = _events_repository()
    try:
        event = repository.find_by_title(title)
    except Exception:
        event = None
    if event:
        return str(event.get("event_id"))

    target = _title_key(title)
    if not target:
        return None
    try:
        candidates = list(repository.root.glob("*/event.json"))
    except Exception:
        return None
    for path in candidates:
        record = read_json(path, {})
        if _title_key(record.get("title")) == target:
            return str(record.get("event_id"))
    return None


def _event_artifact_dir(event_title: str) -> Path | None:
    """Directory holding one event's artifacts, or None when it has none."""
    event_id = _event_id_for_title(event_title)
    if not event_id:
        return None
    try:
        return _events_repository().directory(event_id)
    except (FileNotFoundError, ValueError):
        return None


def _load_event_artifact_list(
    event_title: str,
    filename: str,
    key: str | None = None,
) -> list[dict]:
    """Read a list-shaped artifact, tolerating the old wrapped payloads."""
    event_id = _event_id_for_title(event_title)
    if not event_id:
        return []
    stem = filename[:-5] if filename.endswith(".json") else filename
    try:
        payload = _events_repository().load_artifact(event_id, stem, None)
    except (FileNotFoundError, ValueError):
        return []
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = (
            payload.get(key or "")
            or payload.get("events")
            or payload.get("timeline")
            or payload.get("relationships")
            or []
        )
    return _copy_dict_list(payload)


def _save_event_artifact(event_title: str, filename: str, value) -> Path | None:
    """Persist an artifact through the repository. Returns its path, if known."""
    event_id = _event_id_for_title(event_title)
    if not event_id:
        return None
    stem = filename[:-5] if filename.endswith(".json") else filename
    try:
        return _events_repository().save_artifact(event_id, stem, value)
    except (FileNotFoundError, ValueError):
        return None


def _json_text(payload) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _load_artifact_from_dir(event_dir: Path, name: str, default=None):
    """Read an artifact given the event's directory.

    The framework stores events at ``data/events/<event_id>/``, so the directory
    name is the id the repository keys on. Retrieval works from a directory (it
    is handed one), which makes this the direct route.
    """
    if event_dir is None:
        return default
    stem = name[:-5] if name.endswith(".json") else name
    try:
        return _events_repository().load_artifact(Path(event_dir).name, stem, default)
    except (FileNotFoundError, ValueError):
        return default


def _load_timeline_from_dir(event_dir: Path) -> list[dict]:
    """The event's timeline, as stored by the extraction stage."""
    payload = _load_artifact_from_dir(event_dir, "timeline", [])
    if isinstance(payload, dict):
        payload = payload.get("timeline") or payload.get("events") or []
    return _copy_dict_list(payload)


def timeline_actor_names(ev: dict) -> list[str]:
    """The actor names to match and display a stored timeline entry against.

    ``retrieval_actors`` (written by ``seams.refresh_stored_timeline``) holds
    the layered stage's scored actor names. On a timeline this pipeline
    produced, ``actors`` holds forecast agent ids instead of names, so matching
    a figure's name against it silently finds nothing, and the retrieval agent
    is shown opaque ids. Fall back to ``actors`` for timelines that predate
    that field.
    """
    names: list[str] = []
    for item in ev.get("retrieval_actors") or []:
        name = _clean_text(item.get("name")) if isinstance(item, dict) else _clean_text(item)
        if name:
            names.append(name)
    if names:
        return names
    return [_clean_text(a) for a in (ev.get("actors") or []) if _clean_text(a)]


def timeline_actor_strength(ev: dict) -> float:
    """How central this event's actors are, 0-1.

    The stage scores every actor in an event (``core_actor_score``); the
    highest of them says how load-bearing the event is. It rides along in
    ``retrieval_actors`` and is used as a fraction of a point, so it breaks
    ties between events with the same keyword count without reordering events
    that genuinely match more terms.
    """
    best = 0.0
    for item in ev.get("retrieval_actors") or []:
        if not isinstance(item, dict):
            continue
        try:
            best = max(best, float(item.get("core_actor_score") or 0.0))
        except (TypeError, ValueError):
            continue
    return best


def _load_background_summaries_from_dir(event_dir: Path) -> list[dict]:
    """Per-section summaries as background retrieval sources.

    The framework keeps section summaries in the ``section_summaries`` artifact
    rather than as ``summary.txt`` files on disk, so the ``source`` marker is the
    section's path, matching what the file-scanning version produced.
    """
    summaries = _load_artifact_from_dir(event_dir, "section_summaries", [])
    items: list[dict] = []
    for item in summaries if isinstance(summaries, list) else []:
        if not isinstance(item, dict):
            continue
        text = _clean_text(item.get("summary"))
        if not text:
            continue
        source = _clean_text(item.get("path")) or _clean_text(item.get("section_id"))
        items.append({"source": source, "text": text})
    return items
