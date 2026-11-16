from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

ENGLISH_OUTPUT_INSTRUCTION = (
    "Write all generated prose, summaries, descriptions, and explanations in English. "
    "Preserve identifiers, schema values, proper names, and verbatim source quotations."
)


def checkpoint_languages(language: str | None) -> tuple[str | None, ...]:
    """Also recognize the settings written by the former bilingual UI."""
    return (language, None, "zh", "zh-CN") if (language or "").lower().startswith("en") else (language,)


class LLMOptions(BaseModel):
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    language: str | None = "en-US"

    max_model_calls: int | None = Field(default=100, ge=1, le=50000)
    max_total_tokens: int | None = Field(default=500000, ge=1000, le=100000000)
    max_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    input_price_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_price_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cached_input_price_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def monetary_budget_needs_prices(self):
        if self.max_cost_usd is not None and (self.input_price_per_million is None or self.output_price_per_million is None):
            raise ValueError("A cost limit requires both input and output prices in US dollars per million tokens.")
        return self
