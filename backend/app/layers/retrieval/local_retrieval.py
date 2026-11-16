"""Local evidence retrieval for actor profile generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.layers.support.llm_normalizers import _clean_text


def _clip(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _block(source: str, body: str) -> dict:
    return {"source": source, "text": body}


def _score_by_query(text: str, query: str, base: float = 0.0) -> float:
    terms = set(re.findall(r"[a-z0-9一-鿿]+", (query or "").lower()))
    if not terms:
        return base
    lowered = text.lower()
    return sum(1 for t in terms if t in lowered) + base


def _load_timeline(event_dir: Path) -> list[dict]:
    from app.layers.support.llm_artifacts import _load_timeline_from_dir

    full = Path(event_dir) / "timeline_full.json"
    if full.is_file():
        items = json.loads(full.read_text(encoding="utf-8"))
        return [e for e in items if e.get("origin", "historical") == "historical"]
    return _load_timeline_from_dir(event_dir)


def _timeline_actor_names(ev: dict) -> list[str]:
    from app.layers.support.llm_artifacts import timeline_actor_names

    return timeline_actor_names(ev)


def _actor_strength(ev: dict) -> float:
    from app.layers.support.llm_artifacts import timeline_actor_strength

    return timeline_actor_strength(ev)


def _load_background_summaries(event_dir: Path) -> list[dict]:
    """Per-section summaries as background sources, from the artifact store."""
    from app.layers.support.llm_artifacts import _load_background_summaries_from_dir

    items = _load_background_summaries_from_dir(event_dir)
    return [
        {"source": item["source"], "text": _clip(item["text"], 700)}
        for item in items
        if item.get("text")
    ]


def _timeline_to_text(ev: dict) -> str:
    ts = _clean_text(ev.get("timestamp")) or _clean_text(ev.get("date_original"))
    desc = _clip(ev.get("description"), 400)
    actors = _timeline_actor_names(ev)
    lines = [f"[{ts or '?'}] {desc}"]
    if actors:
        lines.append(f"  Actors: {', '.join(actors)}")
    return "\n".join(lines)


def _retrieve_from_timeline(event_dir: Path, query: str, top_k: int) -> list[dict]:
    blocks: list[dict] = []
    if not query.strip():
        return blocks
    scored = []
    for ev in _load_timeline(event_dir):
        text = (_clean_text(ev.get("description")) + " "
                + " ".join(_timeline_actor_names(ev)))
        base = float(ev.get("importance_score") or 0.0) * 0.5
        score = _score_by_query(text, query, base) + 0.5 * _actor_strength(ev)
        if score > 0:
            scored.append((score, ev))
    scored.sort(key=lambda x: -x[0])
    for _, ev in scored[:top_k]:
        tid = ev.get("timeline_id") or ev.get("timestamp") or "?"
        blocks.append(_block(f"timeline.json#{tid}", _timeline_to_text(ev)))
    return blocks


def _retrieve_from_background(event_dir: Path, query: str, top_k: int) -> list[dict]:
    blocks: list[dict] = []
    if not query.strip():
        return blocks
    scored = []
    for item in _load_background_summaries(event_dir):
        score = _score_by_query(item["text"], query)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: -x[0])
    for _, item in scored[:top_k]:
        blocks.append(_block(item["source"], item["text"]))
    return blocks


def retrieve_actor_context(
    event_dir: Path,
    actor_name: str,
    aliases: list[str] | None = None,
    top_k: int = 6,
    kind: str = "all",
) -> list[dict]:
    """Retrieve source-tagged material about one actor from the event's local
    artifacts. Query = the actor's name + aliases (e.g. "Russia" +
    "Russian Federation"), so any timeline event or chapter mentioning the
    actor surfaces. Returns [{source, text}] ordered by keyword score.
    """
    if event_dir is None or not Path(event_dir).is_dir():
        return []
    query = " ".join(
        _clean_text(n) for n in [actor_name] + (aliases or []) if _clean_text(n)
    )
    blocks: list[dict] = []
    if kind in ("timeline", "all"):
        blocks.extend(_retrieve_from_timeline(event_dir, query, top_k))
    if kind in ("background", "all"):
        blocks.extend(_retrieve_from_background(event_dir, query, top_k))
    # Cap total; dedupe identical sources.
    seen: set[str] = set()
    result: list[dict] = []
    for b in blocks:
        key = (b["source"], b["text"][:80])
        if key in seen:
            continue
        seen.add(key)
        result.append(b)
        if len(result) >= top_k:
            break
    return result
