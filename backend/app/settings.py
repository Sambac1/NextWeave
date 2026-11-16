from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Request-level LLM values override these defaults."""

    llm_api_key: str
    llm_model: str
    llm_base_url: str
    llm_timeout_seconds: float
    agent_sdk_timeout_seconds: float
    agent_sdk_max_concurrency: int
    agent_sdk_max_turns: int
    actor_extraction_max_concurrency: int
    frontend_origins: str
    data_dir: Path

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.frontend_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings(
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        agent_sdk_timeout_seconds=max(
            1.0,
            float(os.getenv("AGENT_SDK_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "120"))),
        ),
        agent_sdk_max_concurrency=max(1, int(os.getenv("AGENT_SDK_MAX_CONCURRENCY", "2"))),
        agent_sdk_max_turns=max(1, int(os.getenv("AGENT_SDK_MAX_TURNS", "3"))),
        actor_extraction_max_concurrency=min(
            12, max(1, int(os.getenv("ACTOR_EXTRACTION_MAX_CONCURRENCY", "8")))
        ),
        frontend_origins=os.getenv("FRONTEND_ORIGINS", "http://localhost:3000"),
        data_dir=Path(os.getenv("DATA_DIR", str(BACKEND_DIR / "data"))).resolve(),
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
