from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from contextvars import ContextVar, Token
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import get_settings


_UNSAFE = re.compile(r"[^\w\-.]+", re.UNICODE)
_CURRENT_REQUEST_RUN_ID: ContextVar[str | None] = ContextVar(
    "current_request_run_id", default=None,
)
_SENSITIVE_KEYS = {
    "api_key", "apikey", "authorization", "auth_token", "access_token",
    "refresh_token", "password",
}


def _safe_part(value: object, default: str, limit: int) -> str:
    cleaned = _UNSAFE.sub("-", str(value or "").strip()).strip("-._").lower()
    return (cleaned or default)[:limit]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: Any) -> Any:
    """Return a JSON-safe copy while removing credentials from persisted logs."""
    if isinstance(value, dict):
        return {
            str(key): "***REDACTED***" if str(key).casefold() in _SENSITIVE_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact(item) for item in value]
    if hasattr(value, "model_dump"):
        return _redact(value.model_dump())
    try:
        json.dumps(value)
        return deepcopy(value)
    except (TypeError, ValueError):
        return str(value)


def set_request_run_id(run_id: str) -> Token:
    return _CURRENT_REQUEST_RUN_ID.set(run_id)


def reset_request_run_id(token: Token) -> None:
    _CURRENT_REQUEST_RUN_ID.reset(token)


def current_request_run_id() -> str | None:
    return _CURRENT_REQUEST_RUN_ID.get()


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if os.name == "nt":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\?\\"):
            target = Path("\\\\?\\" + resolved)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _call_context(call_name: str, phase: str) -> tuple[str, str, str, str]:
    """Infer readable source/round/agent/content labels from an LLM call name."""
    lowered = call_name.casefold()
    round_match = re.search(r"(?:round|step)_(\d+)", lowered)
    round_label = round_match.group(1).zfill(2) if round_match else "na"
    agent_match = re.search(
        r"(?:^|_)agent_(agent_[a-z0-9-]+)_(?:self_activation|information_review|decision|profile)",
        lowered,
    )
    agent_id = agent_match.group(1) if agent_match else "all"
    is_prompt = "request" in phase or phase == "configuration"
    is_error = "error" in phase

    if "master_action_audit" in lowered:
        source, kind = "master-agent", "master-action-audit"
    elif "activation_master_review" in lowered:
        source, kind = "master-agent", "master-activation-review"
    elif "self_activation" in lowered:
        source, kind = "ordinary-agent", "agent-self-activation"
    elif "information_review" in lowered:
        source, kind = "ordinary-agent", "agent-information-processing"
    elif lowered.endswith("_decision"):
        source, kind = "ordinary-agent", "agent-self-prediction"
    elif "game_master_settlement" in lowered:
        source, kind = "game-master", "game-master-settlement"
    elif "relationship_update" in lowered:
        source, kind = "relationship-agent", "relationship-update"
    elif "branch" in lowered and "pruning" in lowered:
        source, kind = "branch-master", "branch-pruning"
    elif "timeline" in lowered:
        source, kind = "timeline-extraction", "timeline"
    elif "registry" in lowered:
        source, kind = "agent-registry", "registry"
    elif lowered.endswith("_profile"):
        source, kind = "profile-extraction", "agent-profile"
    else:
        source, kind = "llm", _safe_part(call_name, "llm-call", 34)

    if is_error:
        content = f"{kind}-error"
    elif is_prompt:
        content = f"{kind}-prompt"
    elif "provider_response" in phase:
        content = f"{kind}-raw-response"
    else:
        content = f"{kind}-result"
    return source, round_label, agent_id, content


class RunLogger:
    """Append-only, human-readable audit files for one API or domain operation."""

    def __init__(
        self,
        category: str,
        *,
        run_id: str | None = None,
        operation: str = "operation",
        event_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        inherited_run_id = current_request_run_id()
        self.run_id = run_id or inherited_run_id or str(uuid.uuid4())
        self.operation = operation
        self.event_id = event_id
        folder = (
            f"{now.strftime('%Y%m%dT%H%M%S_%fZ')}_"
            f"{_safe_part(self.run_id, 'run', 12)}_"
            f"{_safe_part(operation, 'operation', 36)}"
        )
        self.directory = get_settings().data_dir / _safe_part(category, "process_logs", 40) / folder
        self.directory.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = threading.RLock()
        self._started_at = _utc_now()
        self._started_monotonic = time.perf_counter()
        self.record(
            source="runtime",
            round_id="na",
            agent_id="system",
            content_type="run-started",
            status="started",
            data={"category": category, "operation": operation, "event_id": event_id},
        )

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._started_monotonic) * 1000, 3)

    def record(
        self,
        *,
        source: str,
        content_type: str,
        data: Any,
        round_id: int | str | None = None,
        agent_id: str | None = None,
        status: str = "info",
        phase: str = "",
        duration_ms: float | None = None,
    ) -> Path:
        round_label = "na" if round_id in (None, "") else str(round_id).zfill(2)
        agent_label = agent_id or "all"
        with self._lock:
            self._counter += 1
            filename = (
                f"{self._counter:04d}_"
                f"source-{_safe_part(source, 'unknown', 24)}_"
                f"round-{_safe_part(round_label, 'na', 8)}_"
                f"agent-{_safe_part(agent_label, 'all', 30)}_"
                f"content-{_safe_part(content_type, 'data', 38)}_"
                f"status-{_safe_part(status, 'info', 10)}.json"
            )
            path = self.directory / filename
            written_path = _write_json(path, {
                "metadata": {
                    "timestamp_utc": _utc_now(),
                    "run_started_at_utc": self._started_at,
                    "run_elapsed_ms": self.elapsed_ms,
                    "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
                    "run_id": self.run_id,
                    "operation": self.operation,
                    "event_id": self.event_id,
                    "source": source,
                    "round_id": round_id,
                    "agent_id": agent_id,
                    "content_type": content_type,
                    "phase": phase,
                    "status": status,
                },
                "data": _redact(data),
            })
            return written_path

    def finish(self, *, success: bool, data: Any = None, error: Exception | str | None = None) -> Path:
        return self.record(
            source="runtime",
            round_id="na",
            agent_id="system",
            content_type="run-completed" if success else "run-failed",
            status="success" if success else "failed",
            duration_ms=self.elapsed_ms,
            data={
                "success": success,
                "summary": data,
                "error_type": type(error).__name__ if isinstance(error, Exception) else "",
                "error": str(error or ""),
            },
        )


class TraceRecorder(RunLogger):
    """Compatibility recorder that turns every LLM phase into readable audit files."""

    def __init__(self, category: str, run_id: str | None = None) -> None:
        super().__init__(category, run_id=run_id, operation=category)

    def write(
        self,
        call_name: str,
        phase: str,
        value: Any,
        *,
        duration_ms: float | None = None,
    ) -> Path:
        source, round_id, agent_id, content_type = _call_context(call_name, phase)
        if "error" in phase:
            status = "failed"
        elif "request" in phase:
            status = "started"
        elif "validated_response" in phase:
            status = "success"
        elif "provider_response" in phase:
            status = "received"
        else:
            status = "info"
        return self.record(
            source=source,
            round_id=round_id,
            agent_id=agent_id,
            content_type=content_type,
            status=status,
            phase=f"{call_name}:{phase}",
            duration_ms=duration_ms,
            data=value,
        )


def audit_record(recorder: Any, **kwargs: Any) -> Path | None:
    """Record when a real audit recorder is available; remain friendly to test doubles."""
    method = getattr(recorder, "record", None)
    return method(**kwargs) if callable(method) else None


def audit_finish(recorder: Any, **kwargs: Any) -> Path | None:
    method = getattr(recorder, "finish", None)
    return method(**kwargs) if callable(method) else None
