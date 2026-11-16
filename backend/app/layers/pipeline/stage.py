"""Layered-figure stage: actor extraction, assignment and profiles."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domain.common import checkpoint_languages
from app.layers.pipeline.schemas import LayerStageRequest
from app.settings import get_settings

logger = logging.getLogger(__name__)

MAX_CONCURRENT_LLM = get_settings().actor_extraction_max_concurrency


class ArtifactDir:
    """Path-shaped adapter over the framework's EventRepository.

    The ported code writes artifacts as ``event_dir / "name.json"`` and scans
    the event directory for retrieval. The framework stores artifacts through
    ``EventRepository`` keyed by ``event_id``, so this adapter keeps those call
    sites working without rewriting them.
    """

    def __init__(self, events: Any, event_id: str) -> None:
        self.events = events
        self.event_id = event_id

    def __truediv__(self, name: str) -> "ArtifactFile":
        return ArtifactFile(self.events, self.event_id, str(name))

    def path(self) -> Path:
        return self.events.directory(self.event_id)

    def exists(self) -> bool:
        try:
            return self.path().exists()
        except (FileNotFoundError, ValueError):
            return False

    def rglob(self, pattern: str):
        return self.path().rglob(pattern)


class ArtifactFile:
    """One ``event_dir / "<name>"`` target, persisted through the repository."""

    def __init__(self, events: Any, event_id: str, filename: str) -> None:
        self.events = events
        self.event_id = event_id
        self.filename = filename

    def write_text(self, payload: str, encoding: str = "utf-8") -> None:
        stem = self.filename[:-5] if self.filename.endswith(".json") else self.filename
        try:
            parsed: Any = json.loads(payload)
        except ValueError:
            parsed = payload
        self.events.save_artifact(self.event_id, stem, parsed)

    def exists(self) -> bool:
        return (self.events.directory(self.event_id) / self.filename).exists()

    def __fspath__(self) -> str:
        return str(self.events.directory(self.event_id) / self.filename)


class TraceShim:
    """Progress recorder placeholder.

    The framework exposes extraction progress through its own status repository;
    wiring that in is separate work, so the ported stage logs and reports an
    empty public state.
    """

    def record_stage(self, **kwargs: Any) -> None:
        logger.info(
            "layer stage %s: %s",
            kwargs.get("stage", "?"),
            kwargs.get("status", "?"),
        )

    def public_info(self) -> dict[str, Any]:
        return {}


async def run_layer_stage(
    *,
    request: LayerStageRequest,
    timeline: list[Any],
    events: Any,
) -> dict[str, Any]:
    """Run actor extraction, assignment and profiles on a finished timeline.

    ``timeline`` is the framework's ``TimelineEvent`` list. It is converted to
    the plain dicts the step body reads, run, and converted back: the body
    writes ``figures`` / ``actions`` onto its entries, and the forecast needs
    those on the events themselves.

    Returns the stage's result plus the converted forecast agents.
    """
    from app.layers.pipeline import seams, steps
    from app.knowledge.okf import digest
    from app.knowledge.actor_identity import VERSION as IDENTITY_POLICY
    from app.knowledge.actor_evidence import VERSION as AUTHORITY_POLICY
    from app.knowledge.decision_seats import prepare_seat_plan, should_replan_required_seat, VERSION as SEAT_POLICY
    from app.llm.budget import BudgetExceeded

    plan_warning = ''
    saved_plan = events.load_artifact(request.event_id, 'decision_seat_plan', {})
    restart_actor_stage = request.resume_profiles and should_replan_required_seat(saved_plan, request.required_decision_subject)
    profile_checkpoint = (events.load_artifact(request.event_id, 'profile_inputs', {}).get('checkpoint_key')
                          if request.resume_profiles and not restart_actor_stage else None)
    try:
        # A profile-only retry must retain the original actor plan and version.
        # The checkpoint digest below still verifies the complete frozen inputs.
        plan = saved_plan if profile_checkpoint and saved_plan.get('version') else await prepare_seat_plan(request, events)
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning('Decision-seat planning unavailable: %s', exc)
        plan = {'version': SEAT_POLICY, 'status': 'unavailable', 'seats': [],
                'gaps': ['Decision-seat coverage has not been verified.']}
        events.save_artifact(request.event_id, 'decision_seat_plan', plan)
        plan_warning = 'Decision-seat coverage has not been verified; historical candidate selection is provisional.'

    checkpoint_input = {
        'version': 2, 'identity_policy': IDENTITY_POLICY, 'model': request.model, 'base_url': request.base_url,
        'authority_policy': AUTHORITY_POLICY, 'sources': events.load_artifact(request.event_id, 'sources', {}),
        'evidence_cutoff': request.evidence_cutoff,
        'seat_policy': plan.get('version', SEAT_POLICY), 'seat_plan': plan,
        'title': request.event_title, 'language': request.language,
        'max_key_figures': request.max_key_figures,
        'timeline': [t.model_dump() for t in timeline],
    }
    checkpoint_key = digest(checkpoint_input)
    if profile_checkpoint and profile_checkpoint != checkpoint_key and any(
        profile_checkpoint == digest(checkpoint_input | {'language': language})
        for language in checkpoint_languages(request.language)[1:]
    ):
        checkpoint_key = profile_checkpoint
    req = request.model_copy(update={'decision_plan': plan, 'resume_profiles': request.resume_profiles and not restart_actor_stage,
                                    'checkpoint_key': checkpoint_key, 'checkpoint_context':checkpoint_input})
    merged = seams.to_stage_timeline(timeline)
    if plan.get('status') == 'observation_only':
        return {'agents':[], 'profiles':[], 'figures':[], 'timeline':merged, 'name_mapping':{},
                'quality':{'key_figures_selected':0,'decision_seat_plan':plan,
                           'warnings':['This question requires an observation mechanism, not independent actor decisions.']}}
    event_dir = ArtifactDir(events, request.event_id)
    warnings: list[str] = [plan_warning] if plan_warning else []
    extract_trace = TraceShim()
    active_stage = "figure"
    stage_started_at = datetime.now(timezone.utc).isoformat()
    stage_started_perf = time.perf_counter()
    sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    # Bookkeeping the original pipeline filled from its timeline stage.
    merged_all = merged
    older: list[dict[str, Any]] = []
    raw_timeline_events = merged
    cutoff = ""
    timeline_quality: dict[str, Any] = {}

    result = await steps.run(
        req=req,
        merged=merged,
        event_dir=event_dir,
        sem=sem,
        warnings=warnings,
        extract_trace=extract_trace,
        active_stage=active_stage,
        stage_started_at=stage_started_at,
        stage_started_perf=stage_started_perf,
        merged_all=merged_all,
        older=older,
        raw_timeline_events=raw_timeline_events,
        cutoff=cutoff,
        timeline_quality=timeline_quality,
    )

    # The actor ids go on the timeline before the profiles are converted, so the
    # conversion can map the stage's own ids onto the forecast's.
    seams.apply_stage_extraction(
        merged,
        timeline,
        event_id=request.event_id,
        event_title=request.event_title,
        profiles=result.get("profiles") or [],
    )
    # Retrieval metadata first (it lands on the events), then the artifact rewrite
    # (which reads the events back out).
    seams.export_stage_retrieval_context(merged, timeline)
    seams.refresh_stored_timeline(merged, timeline, events, request.event_id)
    result["agents"] = seams.to_agent_profiles(
        result.get("profiles") or [],
        event_id=request.event_id,
        event_title=request.event_title,
    )
    from app.knowledge.decision_seats import actor_coverage
    plan = req.decision_plan
    context = events.load_artifact(request.event_id, 'decision_context', {})
    coverage = actor_coverage(plan, [agent.model_dump() for agent in result['agents']],
                              request.evidence_cutoff, require_bdi=True, rules=context.get('rules', ''))
    quality = result.setdefault('quality', {})
    quality['actor_coverage'] = coverage
    if coverage.get('missing_required') or coverage.get('missing_essential') or coverage.get('unreviewed_question_actors'):
        quality['profile_status'] = 'limited'
        missing = coverage.get('missing_essential') or coverage.get('missing_required') or coverage.get('unreviewed_question_actors', [])
        quality['profile_error'] = 'Actor preparation needs review: ' + '; '.join(missing[:3])
        quality.setdefault('warnings', []).append(quality['profile_error'])
    return result
