from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from app.storage.json_store import EventRepository


ExtractionState = Literal["idle", "running", "completed", "partial", "failed"]


class ExtractionStatusRepository:
    """Persist the latest observable extraction stage for one event."""

    def __init__(self, events: EventRepository) -> None:
        self.events = events

    def get(self, event_id: str) -> dict[str, Any]:
        value = self.events.load_artifact(event_id, "extraction_status", {
            "event_id": event_id,
            "run_id": "",
            "status": "idle",
            "stage": "idle",
            "progress": 0,
            "message": "Extraction has not started.",
            "completed_items": 0,
            "total_items": 0,
            "started_at": None,
            "updated_at": None,
            "error": "",
        })
        quality = self.events.load_artifact(event_id, 'layer_quality', {})
        agents = self.events.load_artifact(event_id, 'agents', [])
        # Validate stored completion flags without rewriting artifacts.
        if value.get('status') == 'completed' and not agents and quality.get('key_figures_selected', 0):
            value = dict(value, status='partial', stage='generating_profiles', progress=90,
                         error=quality.get('profile_error') or '; '.join(quality.get('warnings', [])),
                         message='The timeline and actors were extracted. Resume from the saved progress to finish the profiles.')
        if value.get('status') == 'completed':
            from app.knowledge.decision_seats import actor_coverage
            plan = self.events.load_artifact(event_id, 'decision_seat_plan', {})
            if plan.get('seats'):
                context = self.events.load_artifact(event_id, 'decision_context', {})
                coverage = actor_coverage(plan, agents, context.get('as_of', ''), require_bdi=True,
                                          rules=context.get('rules', ''))
                if coverage.get('missing_required') or coverage.get('missing_essential'):
                    value = dict(value, status='partial', stage='profiles_pending', progress=95,
                                 error='Required actor coverage or an evidence-linked decision state is incomplete.',
                                 message='The timeline is saved. Resume preparation to review decision actors and complete their profiles.')
        budget = self.events.load_artifact(event_id, 'extraction_budget', {})
        value = dict(value, budget={k:v for k,v in budget.items() if k != 'requests'},
                     resume_available=value.get('status') in {'partial','failed'} and self._has_saved_timeline(event_id))
        return value

    def _has_saved_timeline(self, event_id: str) -> bool:
        cache = self.events.load_artifact(event_id, 'timeline_cache', {})
        return bool(cache.get('timeline') or (cache.get('allow_empty_timeline') and cache.get('cache_key')))

    def update(
        self,
        event_id: str,
        *,
        run_id: str,
        stage: str,
        progress: int,
        message: str,
        status: ExtractionState = "running",
        completed_items: int = 0,
        total_items: int = 0,
        error: str = "",
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        previous = self.get(event_id)
        same_run = previous.get("run_id") == run_id
        value = {
            "event_id": event_id,
            "run_id": run_id,
            "status": status,
            "stage": stage,
            "progress": max(0, min(100, int(progress))),
            "message": message,
            "completed_items": max(0, int(completed_items)),
            "total_items": max(0, int(total_items)),
            "started_at": previous.get("started_at") if same_run else now,
            "updated_at": now,
            "error": error,
        }
        self.events.save_artifact(event_id, "extraction_status", value)
        return self.get(event_id)

    def fail(self, event_id: str, error: Exception | str) -> dict[str, Any]:
        previous = self.get(event_id)
        return self.update(
            event_id,
            run_id=str(previous.get("run_id") or ""),
            stage=str(previous.get("stage") or "failed"),
            progress=int(previous.get("progress") or 0),
            message="Extraction failed.",
            status="failed",
            completed_items=int(previous.get("completed_items") or 0),
            total_items=int(previous.get("total_items") or 0),
            error=str(error),
        )
