"""Local intermediate checkpoints, isolated by run/round/input and never public."""
import asyncio
import hashlib
import json
from pathlib import Path
from app.storage.json_store import read_json, write_json

VERSION = 'round-stages-1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def stable_inputs(value):
    if isinstance(value, list):
        return [stable_inputs(v) for v in value]
    if isinstance(value, dict):
        result = {k: stable_inputs(v) for k, v in value.items()}
        if 'agent_id' in result and isinstance(result.get('retrieval_context'), dict):
            # CPU timing does not change what evidence an actor received.
            result['retrieval_context'].pop('duration_ms', None)
        return result
    return value


class RoundStages:
    def __init__(self, root: Path | None, scope, *, locks=None):
        self.root = root
        self.scope = digest({'version': VERSION, 'scope': stable_inputs(scope)})
        self.locks = locks

    def _location(self, stage, inputs):
        signature = digest({'scope': self.scope, 'stage': stage, 'inputs': stable_inputs(inputs)})
        path = self.root / f'{signature}.json' if self.root is not None else None
        return signature, path

    def read(self, stage, inputs, decode):
        signature, path = self._location(stage, inputs)
        if path is not None:
            try:
                cached = read_json(path, {})
                if cached.get('key') == signature and cached.get('payload_sha256') == digest(cached.get('payload')):
                    return decode(cached['payload'])
            except (ValueError, KeyError, TypeError):
                pass  # Never reuse incomplete or mismatched local state.
        return None

    async def run(self, stage, inputs, compute, decode):
        if self.locks is not None:
            signature, _ = self._location(stage, inputs)
            async with self.locks.setdefault(signature, asyncio.Lock()):
                return await self._run(stage, inputs, compute, decode)
        return await self._run(stage, inputs, compute, decode)

    async def _run(self, stage, inputs, compute, decode):
        cached = self.read(stage, inputs, decode)
        if cached is not None:
            return cached
        signature, path = self._location(stage, inputs)
        value = await compute()
        decoded = decode(value)
        if path is not None:
            write_json(path, {'key': signature, 'payload_sha256': digest(value), 'payload': value})
        return decoded

    async def model(self, stage, inputs, compute, model):
        async def generate():
            return (await compute()).model_dump()
        return await self.run(stage, inputs, generate, model.model_validate)
