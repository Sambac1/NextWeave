from __future__ import annotations

from app.domain import EventCreateRequest, LLMOptions, SimulationRequest


def test_llm_language_accepts_frontend_null_value() -> None:
    options = LLMOptions(language=None)
    simulation = SimulationRequest(event_title="Event", language=None)
    event = EventCreateRequest(
        title="Event",
        urls=["https://example.com/article"],
        llm_config={"language": None},
    )

    assert options.language is None
    assert simulation.language is None
    assert event.llm_config is not None
    assert event.llm_config.language is None
