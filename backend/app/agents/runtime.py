from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.domain import LLMOptions
from app.llm.client import LLMClient, LLMError
from app.settings import get_settings
from app.storage.json_store import TraceRecorder


SchemaT = TypeVar("SchemaT", bound=BaseModel)
SKILL_DIRECTORY = Path(__file__).resolve().parent / "skills"
BACKEND_DIRECTORY = Path(__file__).resolve().parents[2]


def _read_skill(name: str) -> str:
    return (SKILL_DIRECTORY / name).read_text(encoding="utf-8").strip()


def _sdk_imports():
    try:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ResultMessage,
            create_sdk_mcp_server,
            query,
            tool,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Claude Agent SDK is not installed. Install backend dependency 'claude-agent-sdk'."
        ) from exc
    return ClaudeAgentOptions, ResultMessage, create_sdk_mcp_server, query, tool


def _deepseek_anthropic_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3].rstrip("/")
    return value if value.endswith("/anthropic") else f"{value}/anthropic"


def _tool_name(agent_name: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", agent_name)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return f"submit_{value or 'agent'}"[:64]


def _validation_error_text(exc: ValidationError) -> str:
    lines = []
    for item in exc.errors(include_url=False):
        path = ".".join(str(part) for part in item.get("loc", ())) or "root"
        lines.append(f"{path}: {item.get('msg', 'invalid value')} [{item.get('type', 'validation_error')}]")
    return "\n".join(lines[:30])


def _preview(value: object, limit: int = 160) -> str:
    """Keep SDK diagnostics useful without copying full model output into every trace."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _sdk_message_summary(message: object, elapsed_seconds: float) -> dict[str, Any]:
    """Return a small, credential-free description of one streamed SDK message."""
    summary: dict[str, Any] = {
        "message_type": type(message).__name__,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    for attribute in (
        "subtype", "model", "error", "stop_reason", "terminal_reason",
        "is_error", "num_turns", "duration_ms", "duration_api_ms", "api_error_status",
    ):
        value = getattr(message, attribute, None)
        if value is not None:
            summary[attribute] = value

    errors = getattr(message, "errors", None)
    if errors:
        summary["errors"] = [_preview(value) for value in list(errors)[:5]]
    usage = getattr(message, "usage", None)
    if isinstance(usage, dict):
        summary["usage"] = usage

    content = getattr(message, "content", None)
    if isinstance(content, str):
        summary["content"] = {
            "type": "str",
            "characters": len(content),
            "preview": _preview(content),
        }
    elif isinstance(content, list):
        blocks = []
        for block in content:
            block_summary: dict[str, Any] = {"type": type(block).__name__}
            text = getattr(block, "text", None)
            if isinstance(text, str):
                block_summary.update({"characters": len(text), "preview": _preview(text)})
            thinking = getattr(block, "thinking", None)
            if isinstance(thinking, str):
                # Record only size: diagnostic logs should not persist hidden reasoning text.
                block_summary["thinking_characters"] = len(thinking)
            tool_name = getattr(block, "name", None)
            if tool_name:
                block_summary["tool_name"] = str(tool_name)
            tool_input = getattr(block, "input", None)
            if isinstance(tool_input, dict):
                block_summary["input_keys"] = sorted(str(key) for key in tool_input)[:30]
                block_summary["input_characters"] = len(
                    json.dumps(tool_input, ensure_ascii=False, default=str)
                )
            if hasattr(block, "is_error") and getattr(block, "is_error") is not None:
                block_summary["is_error"] = bool(getattr(block, "is_error"))
            blocks.append(block_summary)
        summary["content_blocks"] = blocks

    result = getattr(message, "result", None)
    if isinstance(result, str):
        summary["result"] = {"characters": len(result), "preview": _preview(result)}
    data = getattr(message, "data", None)
    if isinstance(data, dict):
        summary["data_keys"] = sorted(str(key) for key in data)[:30]
    return summary


async def run_structured_agent(
    *,
    api_key: str,
    model: str,
    base_url: str,
    name: str,
    instructions: str,
    output_type: type[SchemaT],
    prompt: str,
    max_turns: int = 3,
    recorder: TraceRecorder | None = None,
    extra_tools: list[tuple[str, str, dict[str, Any], Any]] | None = None,
    include_base_prompt: bool = True,
) -> SchemaT:
    """Original SDK contract: the agent must submit through a validated MCP tool."""
    from app.llm.budget import CURRENT, BudgetExceeded
    if CURRENT.get() is not None:
        raise BudgetExceeded("The SDK subprocess cannot track this task's HTTP request budget. Use the parallel structured pipeline.")
    if not api_key.strip():
        raise ValueError("DeepSeek API key is empty.")
    if not model.strip():
        raise ValueError("DeepSeek model name is empty.")
    started = time.perf_counter()
    validation_attempts: list[str] = []
    message_count = 0
    first_message_elapsed: float | None = None
    last_message: dict[str, Any] | None = None
    tool_submission_attempts = 0
    ClaudeAgentOptions, ResultMessage, create_sdk_mcp_server, query, tool = _sdk_imports()
    accepted: dict[str, SchemaT] = {}
    submit_name = _tool_name(name)

    async def submit_handler(arguments: Any) -> dict:
        nonlocal tool_submission_attempts
        tool_submission_attempts += 1
        submission_elapsed = time.perf_counter() - started
        if "value" in accepted:
            if recorder:
                recorder.write(name, "agent_sdk_tool_submission", {
                    "attempt": tool_submission_attempts,
                    "elapsed_seconds": round(submission_elapsed, 3),
                    "status": "duplicate",
                }, duration_ms=submission_elapsed * 1000)
            return {"content": [{"type": "text", "text": "A valid output was already accepted. End the task now."}]}
        try:
            validated = output_type.model_validate(arguments, strict=True, extra="forbid")
        except ValidationError as exc:
            details = _validation_error_text(exc)
            validation_attempts.append(details)
            if recorder:
                recorder.write(name, "agent_sdk_tool_validation_error", {
                    "attempt": tool_submission_attempts,
                    "elapsed_seconds": round(submission_elapsed, 3),
                    "status": "invalid",
                    "validation_error": details,
                }, duration_ms=submission_elapsed * 1000)
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        "VALIDATION_FAILED. Correct every listed field and call this same tool again. "
                        "Do not return prose instead.\n" + details
                    ),
                }],
                "is_error": True,
            }
        accepted["value"] = validated
        if recorder:
            recorder.write(name, "agent_sdk_tool_submission", {
                "attempt": tool_submission_attempts,
                "elapsed_seconds": round(submission_elapsed, 3),
                "status": "accepted",
            }, duration_ms=submission_elapsed * 1000)
        return {
            "content": [{
                "type": "text",
                "text": "VALIDATION_SUCCEEDED. The structured result is accepted. End the task now.",
            }],
        }

    submit_tool = tool(
        submit_name,
        (
            f"Mandatory final-output tool for {name}. Call it with the complete result. Its input must match the "
            "declared schema. If validation fails, correct the reported fields and call it again."
        ),
        output_type.model_json_schema(),
    )(submit_handler)
    server_name = "structured_output"
    mcp_tools = [submit_tool]
    for tool_name, tool_description, tool_schema, tool_handler in (extra_tools or []):
        mcp_tools.append(tool(tool_name, tool_description, tool_schema)(tool_handler))
    server = create_sdk_mcp_server(server_name, tools=mcp_tools)
    qualified_tool_name = f"mcp__{server_name}__{submit_name}"
    allowed_tools = [qualified_tool_name] + [
        f"mcp__{server_name}__{tool_name}" for tool_name, *_ in (extra_tools or [])
    ]
    base_prompt = f"{_read_skill('base.md')}\n\n" if include_base_prompt else ""
    system_prompt = (
        f"{base_prompt}{instructions}\n\n"
        f"OUTPUT CONTRACT: You must finish by calling `{qualified_tool_name}`. Do not provide the final answer as "
        "prose or a markdown JSON block. If validation fails, repair the payload and call the tool again."
    )
    options = ClaudeAgentOptions(
        tools=[],
        allowed_tools=allowed_tools,
        system_prompt=system_prompt,
        mcp_servers={server_name: server},
        strict_mcp_config=True,
        permission_mode="dontAsk",
        model=model,
        max_turns=max_turns,
        cwd=str(BACKEND_DIRECTORY),
        setting_sources=[],
        skills=[],
        env={
            "ANTHROPIC_BASE_URL": _deepseek_anthropic_base_url(base_url),
            "ANTHROPIC_AUTH_TOKEN": api_key.strip(),
            "ANTHROPIC_MODEL": model,
        },
    )
    if recorder:
        recorder.write(name, "agent_sdk_request", {
            "model": model,
            "base_url": _deepseek_anthropic_base_url(base_url),
            "system_prompt": system_prompt,
            "prompt": prompt,
            "output_schema": output_type.model_json_schema(),
        }, duration_ms=0.0)
    final_result = None
    streamed_shapes: set[tuple[str, str]] = set()
    progress_messages = 0
    try:
        async for message in query(prompt=prompt, options=options):
            elapsed = time.perf_counter() - started
            message_count += 1
            if first_message_elapsed is None:
                first_message_elapsed = elapsed
            last_message = _sdk_message_summary(message, elapsed)
            if recorder:
                shape = (
                    str(last_message.get("message_type") or ""),
                    str(last_message.get("subtype") or ""),
                )
                if shape == ("SystemMessage", "thinking_tokens"):
                    progress_messages += 1
                elif shape not in streamed_shapes:
                    streamed_shapes.add(shape)
                    recorder.write(
                        name,
                        "agent_sdk_stream_message",
                        {"sequence": message_count, **last_message},
                        duration_ms=elapsed * 1000,
                    )
            if isinstance(message, ResultMessage):
                final_result = message
        validated = accepted.get("value")
        if validated is None:
            result_error = ""
            if final_result is not None:
                result_error = str(final_result.result or final_result.errors or final_result.stop_reason or "")
            raise ValueError(
                f"{name} did not submit a schema-valid result through {qualified_tool_name}. "
                f"SDK result: {result_error[:1000]}. Validation errors: {'; '.join(validation_attempts[-3:])[:3000]}"
            )
        if recorder:
            recorder.write(name, "agent_sdk_validated_response", {
                "elapsed_seconds": time.perf_counter() - started,
                "first_message_elapsed_seconds": first_message_elapsed,
                "message_count": message_count,
                "suppressed_progress_messages": progress_messages,
                "tool_submission_attempts": tool_submission_attempts,
                "validation_attempts": validation_attempts,
                "result": validated.model_dump(),
            }, duration_ms=(time.perf_counter() - started) * 1000)
        return validated
    except asyncio.CancelledError:
        if recorder:
            recorder.write(name, "agent_sdk_cancelled_error", {
                "elapsed_seconds": time.perf_counter() - started,
                "first_message_elapsed_seconds": first_message_elapsed,
                "message_count": message_count,
                "suppressed_progress_messages": progress_messages,
                "tool_submission_attempts": tool_submission_attempts,
                "accepted_output": "value" in accepted,
                "validation_attempts": validation_attempts,
                "last_message": last_message,
                "message": "Agent SDK execution was cancelled by the caller deadline.",
            }, duration_ms=(time.perf_counter() - started) * 1000)
        raise
    except Exception as exc:
        if recorder:
            recorder.write(name, "agent_sdk_error", {
                "elapsed_seconds": time.perf_counter() - started,
                "first_message_elapsed_seconds": first_message_elapsed,
                "message_count": message_count,
                "suppressed_progress_messages": progress_messages,
                "tool_submission_attempts": tool_submission_attempts,
                "last_message": last_message,
                "validation_attempts": validation_attempts,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }, duration_ms=(time.perf_counter() - started) * 1000)
        raise


class AgentSDKClient(LLMClient):
    """Drop-in structured client backed by the Claude Agent SDK tool loop."""

    def __init__(self, options: LLMOptions, recorder: TraceRecorder | None = None) -> None:
        super().__init__(options, recorder)
        settings = get_settings()
        self.timeout = settings.agent_sdk_timeout_seconds
        self._semaphore = asyncio.Semaphore(settings.agent_sdk_max_concurrency)
        self.max_turns = settings.agent_sdk_max_turns

    async def json_call(
        self,
        *,
        call_name: str,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        temperature: float = 0.4,
        max_tokens: int = 6000,
    ) -> SchemaT:
        skill = ""
        if call_name.endswith("_decision") or call_name.endswith("_information_review"):
            skill = _read_skill("ordinary_bdi_agent.md")
        elif call_name.endswith("_game_master_settlement"):
            skill = _read_skill("simulation_linear.md")
        elif "relationship" in call_name:
            skill = _read_skill("relationship_agent.md")
        elif "activation_master_review" in call_name or "master_action_audit" in call_name:
            skill = _read_skill("master_agents.md")
        # Must precede the generic timeline branch below: "timeline_agent_build"
        # contains "timeline" and would otherwise take the per-section skill.
        elif "timeline_agent" in call_name:
            skill = _read_skill("timeline_agent.md")
        elif "timeline" in call_name or "chapter" in call_name:
            skill = _read_skill("timeline_extraction.md")
        instructions = "\n\n".join(value for value in (system, skill) if value)
        queued_at = time.perf_counter()
        slot_acquired_at: float | None = None
        try:
            async with self._semaphore:
                slot_acquired_at = time.perf_counter()
                if self.recorder:
                    self.recorder.write(
                        call_name,
                        "agent_sdk_slot_acquired",
                        {
                            "queue_wait_seconds": round(slot_acquired_at - queued_at, 3),
                            "timeout_seconds": self.timeout,
                            "max_turns": self.max_turns,
                        },
                        duration_ms=(slot_acquired_at - queued_at) * 1000,
                    )
                async with asyncio.timeout(self.timeout):
                    return await run_structured_agent(
                        api_key=self.api_key,
                        model=self.model,
                        base_url=self.base_url,
                        name=call_name,
                        instructions=instructions,
                        output_type=schema,
                        prompt=prompt,
                        max_turns=self.max_turns,
                        recorder=self.recorder,
                    )
        except TimeoutError as exc:
            now = time.perf_counter()
            queue_wait_seconds = (slot_acquired_at - queued_at) if slot_acquired_at is not None else None
            execution_seconds = (now - slot_acquired_at) if slot_acquired_at is not None else None
            if self.recorder:
                self.recorder.write(
                    call_name,
                    "agent_sdk_timeout_error",
                    {
                        "timeout_seconds": self.timeout,
                        "queue_wait_seconds": round(queue_wait_seconds, 3) if queue_wait_seconds is not None else None,
                        "execution_seconds": round(execution_seconds, 3) if execution_seconds is not None else None,
                        "total_elapsed_seconds": round(now - queued_at, 3),
                        "message": "Claude Agent SDK call exceeded its hard deadline and was cancelled.",
                    },
                    duration_ms=(execution_seconds or self.timeout) * 1000,
                )
            raise LLMError(
                f"Claude Agent SDK call '{call_name}' timed out after {self.timeout:g} seconds"
            ) from exc
