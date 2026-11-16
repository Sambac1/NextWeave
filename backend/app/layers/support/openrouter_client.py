"""LLM HTTP client used by the layered-figure pipeline."""

import json
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from app.domain.common import ENGLISH_OUTPUT_INSTRUCTION
from app.llm.http_pool import shared_http_client

# Load before reading LLM_BASE_URL because service modules may import this before
# main.py has loaded environment variables.
load_dotenv()

OPENROUTER_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")


def _normalize_llm_base_url(base_url: str | None) -> str:
    effective_base = (base_url or OPENROUTER_BASE_URL or "").strip().rstrip("/")
    if not effective_base:
        raise ValueError("LLM Base URL is empty. Please enter a provider URL such as https://api.deepseek.com/v1.")
    if not effective_base.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid LLM Base URL: {effective_base}. It must start with http:// or https://."
        )
    return effective_base


def _extract_provider_error(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err.get("code") or data)
            if err:
                return str(err)
            return str(data.get("message") or data)
    except Exception:
        pass
    return response.text[:1000]


def _friendly_llm_error(response: httpx.Response, base_url: str, model: str) -> str:
    provider_msg = _extract_provider_error(response)
    status = response.status_code
    if status in {401, 403}:
        hint = "LLM authentication failed. Check whether the API key matches this Base URL/provider."
    elif status == 404:
        hint = "LLM endpoint or model was not found. Check the Base URL and model name."
    elif status == 400:
        hint = "LLM request was rejected. Check the model name and whether this provider supports OpenAI-compatible chat completions."
    elif status == 429:
        hint = "LLM provider rate limit reached. Try again later or use another key/model."
    else:
        hint = f"LLM provider returned HTTP {status}."
    return f"{hint} Base URL={base_url}, model={model}. Provider response: {provider_msg}"


async def _call_openrouter_with_meta(
    api_key: str,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    base_url: str | None = None,
    language: str | None = None,
    json_mode: bool = False,
) -> tuple[str, dict]:
    """Same call as _call_openrouter, but also returns provider response metadata.

    The metadata lets callers tell apart an empty completion caused by a
    truncated response (finish_reason="length") from one where the model
    deliberately returned nothing.
    """
    api_key = str(api_key or "").strip()
    model = str(model or "").strip()
    effective_base = _normalize_llm_base_url(base_url)

    if not api_key:
        raise ValueError("LLM API key is empty. Please enter the API key for the selected provider.")
    if not model:
        raise ValueError("LLM model name is empty. Please enter a model name supported by the selected provider.")

    if (language or "").lower().startswith("zh"):
        prefix = "Please respond in Simplified Chinese.\n\n"
        system_prompt = prefix + system_prompt if system_prompt else prefix.strip()
    elif (language or "en-US").lower().startswith("en"):
        system_prompt = "\n".join(part for part in (system_prompt, ENGLISH_OUTPUT_INSTRUCTION) if part)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        # Structured extraction is especially prone to reasoning-only prose on models
        # that otherwise return HTTP 200.
        payload["response_format"] = {"type": "json_object"}
    # Every layered extraction call should return its answer directly.
    if urlparse(effective_base).hostname == "api.deepseek.com":
        payload["thinking"] = {"type": "disabled"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if "openrouter.ai" in effective_base:
        headers["HTTP-Referer"] = "https://everything-is-possible.app"
        headers["X-Title"] = "Everything is Possible"

    try:
        trust_env = "openrouter.ai" in effective_base
        async with shared_http_client(timeout=120.0, trust_env=trust_env) as client:
            response = await client.post(
                f"{effective_base}/chat/completions",
                json=payload,
                headers=headers,
            )
            if json_mode and response.status_code == 400:
                # A few compatible gateways reject response_format even though
                # they support chat completions. Fall back only when the error
                # specifically names the JSON option; don't mask other 400s.
                error_text = response.text.lower()
                if "response_format" in error_text or "json_object" in error_text or "json mode" in error_text:
                    payload.pop("response_format", None)
                    response = await client.post(
                        f"{effective_base}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
    except httpx.InvalidURL as exc:
        raise ValueError(
            f"Invalid LLM Base URL: {effective_base}. Example: https://api.deepseek.com/v1."
        ) from exc
    except httpx.RequestError as exc:
        raise ValueError(
            f"Could not connect to LLM Base URL {effective_base}. Check the URL and network. Details: {exc}"
        ) from exc

    if not response.is_success:
        raise ValueError(_friendly_llm_error(response, effective_base, model))

    data = response.json()
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    # Reasoning is not the assistant's final answer and must never be parsed as
    # structured output. In particular, feeding reasoning_content to a JSON
    # parser turns a successful HTTP response into a misleading parse failure.
    usage = data.get("usage") or {}
    meta = {
        "finish_reason": choice.get("finish_reason") or "",
        "content_length": len(content),
        "has_choices": bool(choices),
        "has_reasoning": bool(message.get("reasoning") or message.get("reasoning_content")),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }
    return content, meta


async def _call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    base_url: str | None = None,
    language: str | None = None,
) -> str:
    content, _ = await _call_openrouter_with_meta(
        api_key,
        model,
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
        base_url=base_url,
        language=language,
    )
    return content


def _json_closing_suffix(text: str) -> str:
    """Return the closing brackets needed for a truncated JSON fragment."""
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    return "".join(reversed(stack))


def _parse_json(text: str) -> Any:
    """Parse JSON from LLM output, with multiple recovery strategies."""
    cleaned = text.strip()

    # Strip markdown fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt: find JSON array [...]
    arr_start, arr_end = cleaned.find("["), cleaned.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        try:
            return json.loads(cleaned[arr_start:arr_end + 1])
        except json.JSONDecodeError:
            partial = cleaned[arr_start:arr_end + 1]
            last_comma = partial.rfind(",")
            if last_comma != -1:
                try:
                    return json.loads(partial[:last_comma] + "]")
                except json.JSONDecodeError:
                    pass

    # Attempt: find JSON object {...} with closing brace
    obj_start, obj_end = cleaned.find("{"), cleaned.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        try:
            return json.loads(cleaned[obj_start:obj_end + 1])
        except json.JSONDecodeError:
            partial = cleaned[obj_start:obj_end + 1]
            last_comma = partial.rfind(",")
            if last_comma != -1:
                try:
                    return json.loads(partial[:last_comma] + "}")
                except json.JSONDecodeError:
                    pass

    # Attempt: recover truncated JSON (no closing brace/bracket — LLM hit max_tokens)
    # Strategy: close any open strings, drop the last incomplete element, then close
    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")
    # Pick whichever comes first (could be {"events": [ or just [)
    if obj_start != -1 or arr_start != -1:
        if obj_start == -1:
            truncated = cleaned[arr_start:]
        elif arr_start == -1:
            truncated = cleaned[obj_start:]
        elif obj_start < arr_start:
            truncated = cleaned[obj_start:]
        else:
            truncated = cleaned[arr_start:]

        # Close any unterminated string by appending a quote
        # Count unescaped quotes to detect open strings
        in_string = False
        escape = False
        for ch in truncated:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
        if in_string:
            truncated += '"'

        # Now try dropping the last incomplete element and closing
        # Find last comma outside of strings and brackets
        depth = 0
        in_string = False
        escape = False
        last_comma_pos = -1
        for i, ch in enumerate(truncated):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '[{':
                depth += 1
            elif ch in ']}':
                depth -= 1
            elif ch == ',' and depth <= 1:
                last_comma_pos = i

        # Try: drop last incomplete element + close
        if last_comma_pos > 0:
            closes = _json_closing_suffix(truncated[:last_comma_pos])
            candidate = truncated[:last_comma_pos] + closes
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Try: just close everything without dropping
        closes = _json_closing_suffix(truncated)
        candidate = truncated + closes
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON. Preview: {cleaned[:200]}")

async def _call_json_with_retries(
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    attempts: int = 3,
    base_url: str | None = None,
) -> Any:
    """Call the model and retry when it returns malformed JSON."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        retry_note = ""
        if attempt > 1:
            retry_note = (
                "\n\nYour previous response was not valid JSON. "
                "Return the same task result again as compact, valid JSON only. "
                "No markdown, no prose outside JSON, no trailing commas. "
                "Close all arrays and objects."
            )

        raw = await _call_openrouter(
            api_key,
            model,
            None,
            prompt + retry_note,
            max_tokens=max_tokens,
            temperature=max(0.2, temperature - (attempt - 1) * 0.2),
            base_url=base_url,
        )
        try:
            return _parse_json(raw)
        except ValueError as exc:
            last_error = exc

    raise ValueError(f"Failed to parse JSON after {attempts} attempts: {last_error}")


