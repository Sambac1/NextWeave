"""Functional-Agent boundaries for evidence extraction and profile maintenance."""

from __future__ import annotations

import json
from typing import Any

from app.knowledge.timeline import (
    extract_section_timeline,
    merge_timelines,
    select_timeline_sections,
)
from app.knowledge.timeline_agent import (
    TimelineAgentOutcome,
    build_agent_timeline,
)
from app.llm.client import LLMClient


class TimelineExtractionAgent:
    """Select evidence sections and turn them into one Timeline."""

    def __init__(self, runtime: LLMClient, utility_client: LLMClient | None = None) -> None:
        self.runtime = runtime
        # Selection and deduplication are bounded transformation calls rather
        # than autonomous Agent work. Keep them on the lightweight client.
        self.utility_client = utility_client or runtime

    async def select_sections(self, event_title: str, sections: list[dict[str, Any]]) -> list[str]:
        return await select_timeline_sections(event_title, sections, self.utility_client)

    async def extract_section(self, section: dict[str, Any]) -> list[dict[str, Any]]:
        return await extract_section_timeline(section, self.runtime)

    async def merge(self, groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return await merge_timelines(groups, self.utility_client)

    async def build_timeline(
        self,
        event_title: str,
        sections: list[dict[str, Any]],
    ) -> TimelineAgentOutcome:
        """One pass producing the whole recorded timeline.

        The production runtime is the Agent SDK client, whose ``json_call``
        matches ``LLMClient``'s signature and routes ``timeline_agent_build``
        to the ``timeline_agent.md`` skill. The annotation is widened rather
        than narrowed to the two concrete clients the duck type covers.
        """
        runtime: Any = self.runtime
        return await build_agent_timeline(event_title, sections, runtime)
