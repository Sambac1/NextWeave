"""Checkpoint storage independent of the interactive world's latest_snapshot."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from app.settings import get_settings
from app.storage.json_store import read_json, write_json


class EnsembleStore:
    def __init__(self, root: Path | None = None):
        self.root = root or get_settings().data_dir / "ensemble_runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id: str) -> Path:
        try:
            if str(uuid.UUID(run_id)) != run_id:
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError("Invalid experiment ID") from None
        return self.root / run_id

    def load(self, run_id: str, name: str, default=None):
        return read_json(self.directory(run_id) / f"{name}.json", default)

    def save(self, run_id: str, name: str, value):
        return write_json(self.directory(run_id) / f"{name}.json", value)

    def sample(self, run_id: str, sample_id: int):
        return self.load(run_id, f"samples/{sample_id:04d}", {})

    def save_sample(self, run_id: str, sample_id: int, value):
        return self.save(run_id, f"samples/{sample_id:04d}", value)

    def acquire(self, run_id: str):
        """Also excludes duplicate work across API workers; stale PID locks recover."""
        directory = self.directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / ".running"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                os.kill(int(lock.read_text()), 0)
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True)
                return self.acquire(run_id)
            raise ValueError("This experiment is already running") from None
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))

    def release(self, run_id: str):
        (self.directory(run_id) / ".running").unlink(missing_ok=True)
