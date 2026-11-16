from __future__ import annotations

from datetime import date

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.domain.common import LLMOptions


class EventCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    urls: list[str] = Field(min_length=1, max_length=20)
    additional_context: str | None = None
    llm_config: LLMOptions | None = None


class EntityRequest(LLMOptions):
    event_title: str
    background: str


class SummaryRequest(LLMOptions):
    event_id: str | None = None
    event_title: str
    article_summaries: list[dict[str, str]]
    additional_context: str | None = None


class FiguresRequest(LLMOptions):
    # A complete corpus pass plus several dossiers needs more than 100 calls.
    # Explicit caller limits still apply; resume keeps cumulative accounting.
    max_model_calls: int | None = Field(default=300, ge=1, le=50000)
    max_total_tokens: int | None = Field(default=2000000, ge=1000, le=100000000)
    resume_profiles: bool = False
    event_id: str
    event_title: str
    history_before: date | None = None
    allow_empty_timeline: bool = False
    required_decision_subject: str = Field(default="", max_length=160)
    exclude_poll_only_profiles: bool = False
    time_window: str = "all"
    recent_years: int | None = Field(default=None, ge=1, le=20)
    max_key_figures: int = Field(default=8, ge=1, le=8)
    timeline_mode: Literal["parallel", "legacy_agent"] = "parallel"
    planning_mode: Literal["auto", "rules"] = "auto"
    extraction_deadline_seconds: int | None = Field(default=900, ge=30, le=7200)
    use_timeline_agent: bool = Field(
        default=True,
        description=(
            "Build the whole recorded timeline in one Timeline Agent pass over the "
            "selected sections instead of extracting each section separately and "
            "merging. Falls back to the per-section path automatically when the agent "
            "fails."
        ),
    )
    force: bool = Field(
        default=False,
        description="Re-run extraction even when this event already has saved figures.",
    )


class RelationshipsRequest(LLMOptions):
    event_id: str | None = None
    event_title: str
    agents: list[dict[str, Any]]


class TranslationRequest(LLMOptions):
    text: str | None = None
    texts: list[str] | None = None

    @model_validator(mode="after")
    def require_content(self) -> "TranslationRequest":
        if self.text is None and not self.texts:
            raise ValueError("text or texts is required")
        return self
