"""Event creation and source collection use case."""

from __future__ import annotations

import logging
import shutil
import time
from typing import Any

from app.domain import EventCreateRequest
from app.knowledge.scraper import SourceCollector, _is_usable
from app.storage.json_store import EventRepository, GraphRepository
from app.storage.run_logger import RunLogger

logger = logging.getLogger(__name__)


async def _close_quietly(collector: SourceCollector | None) -> None:
    """Close the HTTP client without letting it replace the real error.

    ``SourceCollector.close`` awaits an httpx close, which can raise. In the
    failure path that exception would surface instead of the collection error
    the caller needs to see, so the close is logged and swallowed there. A
    ``None`` collector means construction itself failed, so there is nothing
    open to close.
    """
    if collector is None:
        return
    try:
        await collector.close()
    except Exception:  # noqa: BLE001
        logger.warning("Could not close the source collector", exc_info=True)


class SourceCollectionError(ValueError):
    """No requested source produced usable content."""


def _discard_event(events: EventRepository, event_id: str) -> None:
    """Remove an event directory whose collection never produced content."""
    try:
        directory = events.directory(event_id)
    except (FileNotFoundError, ValueError):
        return
    try:
        shutil.rmtree(directory)
    except OSError:
        # Preserve the collection error if directory cleanup also fails.
        logger.warning("Could not discard the empty event directory %s", directory, exc_info=True)


class EventIngestionService:
    """Create an event, collect its sources, and establish the root graph node."""

    def __init__(self, events: EventRepository, graphs: GraphRepository) -> None:
        self.events = events
        self.graphs = graphs

    async def create(self, request: EventCreateRequest) -> dict[str, Any]:
        event = self.events.create(request.title, request.urls, request.additional_context)
        event_id = event["event_id"]
        logger = RunLogger("event_logs", operation="event-source-collection", event_id=event_id)
        # Constructed inside the guard: the collector validates its proxy URL at
        # construction, so a malformed HTTPS_PROXY raises here.
        collector: SourceCollector | None = None
        collection_started = time.perf_counter()
        try:
            collector = SourceCollector()
            sources = await collector.fetch_all(
                request.urls,
                self.events.directory(event_id) / "sources",
                llm_options=None,  # Source acquisition never silently starts an unmetered SDK process.
            )
        except BaseException:
            if collector is not None:
                await _close_quietly(collector)
            _discard_event(self.events, event_id)
            raise
        await _close_quietly(collector)
        self.events.save_sources(event_id, sources)
        collection_elapsed = time.perf_counter() - collection_started
        self.events.save_artifact(event_id, "ingestion_manifest", {
            "source_count": len(sources),
            "successful_sources": sum(1 for item in sources if _is_usable(item)),
            "collection_seconds": round(collection_elapsed, 3),
        })
        # One definition of "usable", the one the scraper itself applied when it decided
        # to return this result.
        successful = [item for item in sources if _is_usable(item)]
        for index, source in enumerate(sources, 1):
            succeeded = _is_usable(source)
            logger.record(
                source="source-collector",
                round_id="collection",
                agent_id=f"source-{index:02d}",
                content_type="source-index" if succeeded else "source-retrieval-error",
                status="success" if succeeded else "failed",
                data={
                    "source_id": f"source_{index:02d}",
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "domain": source.get("domain", ""),
                    "summary": source.get("summary", ""),
                    "body_char_count": len(str(source.get("content") or "")),
                    "sdk_fallback": bool(source.get("sdk_fallback")),
                    "error": source.get("error"),
                },
            )
        if not successful:
            errors = [str(item.get("error") or "unknown retrieval error") for item in sources]
            logger.finish(success=False, data={"source_count": len(sources)}, error="; ".join(errors))
            _discard_event(self.events, event_id)
            raise SourceCollectionError(
                f"No content could be retrieved from any provided URL. Causes: {'; '.join(errors)}"
            )

        self.graphs.add_node(
            event_id,
            {
                "id": event_id,
                "label": request.title,
                "type": "root_event",
                "properties": {"additional_context": request.additional_context or ""},
            },
        )
        graph = self.graphs.get(event_id)
        logger.record(
            source="event-creation",
            round_id="collection",
            agent_id="root-event",
            content_type="initial-persisted-graph",
            status="success",
            data=graph,
        )
        logger.finish(
            success=True,
            data={
                "source_count": len(sources),
                "successful_sources": len(successful),
                "collection_seconds": round(collection_elapsed, 3),
            },
        )
        public_sources = [
            {key: value for key, value in source.items() if key not in {"content", "sections"}}
            for source in sources
        ]
        return {
            **event,
            "scraped_articles": public_sources,
            "additional_context": request.additional_context or "",
        }
