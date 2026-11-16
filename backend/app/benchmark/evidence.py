"""Compatibility exports for benchmark callers."""
from app.knowledge.wikipedia import discover_wikipedia, fetch_revision, utc_timestamp

__all__ = ["discover_wikipedia", "fetch_revision", "utc_timestamp"]
