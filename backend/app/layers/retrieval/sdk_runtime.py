"""Structured-output agent runner for the layered-figure stage."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound=BaseModel)


class SDKStrictModel(BaseModel):
    """Base for agent submissions: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


def _framework_runtime():
    from app.agents.runtime import run_structured_agent

    return run_structured_agent


def _supports_extra_tools() -> bool:
    """True once the runtime accepts ``extra_tools``."""
    import inspect

    try:
        signature = inspect.signature(_framework_runtime())
    except (TypeError, ValueError):
        return False
    return "extra_tools" in signature.parameters


async def _run_structured_agent(
    *,
    api_key: str,
    model: str,
    base_url: str | None,
    name: str,
    instructions: str,
    output_type: type[T],
    prompt: str,
    extra_tools: list[tuple[str, str, dict[str, Any], Any]] | None = None,
    max_turns: int = 6,
) -> T:
    """Run one structured-output agent, with optional extra MCP tools.

    ``max_turns`` is the agent's interaction budget, not a repair-retry count:
    the retrieval agents need ten or more turns because each ``search_local``
    call costs one.
    """
    if extra_tools and not _supports_extra_tools():
        raise RuntimeError(
            "The Agent SDK runtime does not accept extra_tools, so the "
            "retrieval agent cannot be given search_local. Pass extra_tools "
            "through run_structured_agent to enable it."
        )
    return await _framework_runtime()(
        api_key=api_key,
        model=model,
        base_url=base_url or "",
        name=name,
        instructions=instructions,
        output_type=output_type,
        prompt=prompt,
        max_turns=max_turns,
        include_base_prompt=False,
        extra_tools=extra_tools,
    )
