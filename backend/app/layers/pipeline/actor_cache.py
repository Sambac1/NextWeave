"""Reusable actor/action batches; technical failures never masquerade as empties."""
from pathlib import Path

from app.knowledge.okf import digest
from app.layers.prompts.llm_prompts import EXTRACT_FIGURES_BATCH_PROMPT, EXTRACT_FIGURES_FROM_ENTRY_PROMPT
from app.storage.json_store import read_json, write_json


class ActorBatchCache:
    def __init__(self, directory: Path, model: str, base_url: str | None):
        base = directory.path() if callable(getattr(directory, "path", None)) else Path(directory)
        self.root = base / "actor_chunks"
        self.settings = {"version": 1, "model": model, "base_url": base_url,
                         "prompt": digest(EXTRACT_FIGURES_BATCH_PROMPT + EXTRACT_FIGURES_FROM_ENTRY_PROMPT)}

    def path(self, entries):
        return self.root / (digest({"settings": self.settings, "entries": entries}) + ".json")

    def load(self, entries):
        data = read_json(self.path(entries), None)
        if not data or len(data) != len(entries):
            return None
        return [(entry, row["figures"], row["actions"], row["diagnostics"])
                for entry, row in zip(entries, data)]

    def save(self, entries, results):
        if len(entries) == len(results) and all(diag.get("status") == "ok" for _, _, _, diag in results):
            write_json(self.path(entries), [{"figures": figures, "actions": actions, "diagnostics": diag}
                                          for _, figures, actions, diag in results])
