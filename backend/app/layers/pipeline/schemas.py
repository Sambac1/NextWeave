"""Request model for the layered-figure stage.

Mirrors the timeline-extraction agent's ``FiguresPipelineRequest`` for the
fields this stage actually reads. The framework owns timeline extraction, so
``time_window`` / ``recent_years`` are absent: the window is already applied
before the timeline reaches this stage.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.layers.support.layer_rules import MAX_PRINCIPALS


class LayerStageRequest(BaseModel):
    """Inputs for actor extraction, assignment and profile generation."""

    event_id: str
    resume_profiles: bool = False
    checkpoint_key: str = ""
    checkpoint_context: dict = Field(default_factory=dict, exclude=True)
    event_title: str
    evidence_cutoff: str = ""
    decision_plan: dict = Field(default_factory=dict)
    required_decision_subject: str = Field(default="", max_length=160)
    exclude_poll_only_profiles: bool = False
    api_key: str
    model: str
    base_url: str | None = None
    language: str | None = None
    max_key_figures: int = Field(
        MAX_PRINCIPALS,
        ge=1,
        le=20,
        description="Maximum number of principals. Default comes from layer_rules.",
    )
    min_portrait_figures: int = Field(
        3,
        ge=1,
        le=20,
        description="Minimum figures surviving the evidence filter even below the ratio threshold.",
    )
    min_evidence_ratio: float = Field(
        0.025,
        ge=0.0,
        le=1.0,
        description=(
            "Main evidence floor, as a share of total timeline entries an actor must "
            "appear in to stay eligible. Scales with event size, unlike a fixed count. "
            "Adjusted per entity type. This is a noise filter, not a ranking signal."
        ),
    )
    min_evidence_count: int = Field(
        2,
        ge=1,
        le=20,
        description=(
            "Absolute floor under min_evidence_ratio, in raw timeline entries. Only "
            "bites on short timelines, where the ratio rounds down to 1."
        ),
    )
