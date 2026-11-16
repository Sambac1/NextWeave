"""Inputs shared by the forecast question preview and its confirmed analysis."""
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from app.domain.common import LLMOptions


class ForecastPreviewRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1500)


class ForecastConfirmation(BaseModel):
    preview_id: str
    outcome_deadline: AwareDatetime | None = None
    yes_criterion: str | None = Field(default=None, max_length=1200)
    no_criterion: str | None = Field(default=None, max_length=1200)
    seed_wiki_urls: list[str] = Field(default_factory=list, max_length=5)


class QuestionExperimentRequest(LLMOptions):
    event_id: str
    sample_count: int = Field(default=8, ge=2, le=64)
    steps: int | None = Field(default=None, ge=1, le=12, strict=True)
    max_runtime_seconds: int | None = Field(default=900, ge=30, le=86400)


class ForecastSpec(BaseModel):
    kind: Literal["question", "market"]
    question: str = Field(min_length=3, max_length=300)
    as_of: AwareDatetime
    outcome_deadline: AwareDatetime | None = None
    deadline_basis: str = ""
    yes_criterion: str = ""
    no_criterion: str = ""
    market_url: str = ""
    rules: str = Field(default="", max_length=20000)
    resolution_source: str = ""
    outcome_window_start: AwareDatetime | str | None = None
    outcome_window_basis: str = ""
    nominal_market_end: AwareDatetime | None = None
    market_rules_at: AwareDatetime | None = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Enter a forecast question.")
        return value

    def contract_case(self) -> dict:
        result = self.model_dump(mode="json", exclude_none=True)
        if self.kind == "question":
            result["rules"] = (
                f"Resolve to Yes if {self.yes_criterion}\n"
                f"Resolve to No if {self.no_criterion}\n"
                f"Decision deadline {self.outcome_deadline.isoformat()}."
            )
        return result
