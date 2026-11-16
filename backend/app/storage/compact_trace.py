"""Small append-only model audit; full states already live in checkpoints."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class CompactTrace:
    def __init__(self, directory: Path, api_key: str = ""):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "model_calls.jsonl"
        self.api_key = api_key
        self.tokens = {"prompt_tokens": 0, "completion_tokens": 0}

    def write(self, call_name, phase, value, *, duration_ms=None):
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        item = {"at": datetime.now(timezone.utc).isoformat(), "call": call_name,
                "phase": phase, "duration_ms": duration_ms, "char_count": len(serialized),
                "sha256": hashlib.sha256(serialized.encode()).hexdigest()}
        if isinstance(value, dict):
            usage = value.get("usage") or {}
            if usage:
                item["usage"] = usage
                for key in self.tokens:
                    self.tokens[key] += int(usage.get(key) or 0)
            if "error" in phase:
                message = str(value.get("message", ""))[:1000]
                item["error"] = message.replace(self.api_key, "[redacted]") if self.api_key else message
                if phase == 'settlement_semantic_error':
                    item['diagnostic'] = {key: value.get(key, []) for key in ('known_agent_ids', 'returned_agent_ids', 'event_ids')}
                    # Keep the exact failed binding locally, without full
                    # requests, private inboxes, or authentication headers.
                    detail = {key: value.get(key, []) for key in (
                        'known_agent_ids', 'returned_agent_ids', 'event_ids', 'execution_records', 'state_changes')}
                    detail['message'] = item['error']
                    body = json.dumps(detail, ensure_ascii=False)
                    if self.api_key:
                        body = body.replace(self.api_key, '[redacted]')
                    folder = self.path.parent / 'settlement_diagnostics'
                    folder.mkdir(exist_ok=True)
                    name = hashlib.sha256(call_name.encode()).hexdigest() + '.json'
                    (folder / name).write_text(body, encoding='utf-8')
                    item['diagnostic_artifact'] = f'settlement_diagnostics/{name}'
                if phase == 'outlook_validation_error':
                    body = json.dumps({'issue': item['error'], 'public_readout': value.get('public_readout')}, ensure_ascii=False)
                    if self.api_key:
                        body = body.replace(self.api_key, '[redacted]')
                    folder = self.path.parent / 'outlook_diagnostics'
                    folder.mkdir(exist_ok=True)
                    name = hashlib.sha256((call_name + body).encode()).hexdigest() + '.json'
                    (folder / name).write_text(body, encoding='utf-8')
                    item['diagnostic_artifact'] = f'outlook_diagnostics/{name}'
            if phase == 'settlement_annotation_diagnostics':
                item['diagnostic_counts'] = {}
                for row in value.get('diagnostics', []):
                    code = row.get('code', 'unknown')
                    item['diagnostic_counts'][code] = item['diagnostic_counts'].get(code, 0) + 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def record(self, **kwargs):
        # Round snapshots, actions and public/private state are checkpointed by
        # the ensemble runner. Do not duplicate them for each audit hook.
        return None
