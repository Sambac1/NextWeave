"""Step body for the layered-figure stage (actor extraction through profiles).

This module is the original ``run_figures_pipeline`` Step 4a-6 slice, kept
verbatim apart from de-indentation and the ``openrouter_service`` re-exports
being replaced by their real modules. Keeping the body unchanged is what makes
the ported behaviour match the 31-event baseline.

``run`` lists every value the body read from its enclosing scope in the
original function, so nothing is guessed at call time.
"""

from __future__ import annotations

import asyncio
from app.llm.concurrency import gather_complete
import json
import logging
import re
import time
import uuid
from app.domain.agency import is_decision_subject, non_agent_kind
from app.domain.identity import contract_subject_key, matching_subjects
from app.llm.budget import BudgetExceeded
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.layers.support.layer_rules import (
    CATEGORY_REPRESENTATIVE_LIMITS,
    COMPACTABLE_CATEGORIES,
    MAX_MEMBERS_PER_PRINCIPAL,
    evidence_threshold_for,
    find_container,
    infer_entity_type,
    is_leader,
    name_key,
)
from app.layers.pipeline.helpers import (
    _actor_id,
    _build_figure_context,
    _compact_key_figure_names,
    _compose_figure_profile,
    _fill_missing_stances,
    _filter_by_min_evidence,
    _merge_canonical_figures,
    _principal_pick_rank,
    _resolve_mapping,
    _score_figures,
    _write_json,
)
from app.layers.pipeline.llm_figure_service import (
    EXTRACT_BATCH_SIZE,
    assign_subordinates,
    audit_layer_assignment,
    break_alias_cycles,
    check_principal_overlaps,
    extract_figures_from_batch,
    merge_across_batches,
    merge_figures,
    resolve_member_ownership,
    select_principals,
)
from app.layers.support.llm_normalizers import (
    _clean_text,
    _dedupe_texts,
    _normalize_stance,
    _stance_rank,
)
from app.layers.pipeline.profile_service import (
    build_profiles_view,
    generate_profiles,
    write_profiles_artifact,
)
from app.layers.pipeline.actor_cache import ActorBatchCache

logger = logging.getLogger(__name__)

CROSS_BATCH_MERGE_ENABLED = False

_SCOPE = (
    "req merged event_dir sem warnings extract_trace active_stage "
    "stage_started_at stage_started_perf merged_all older raw_timeline_events "
    "cutoff timeline_quality"
)


async def run(*, req, merged, event_dir, sem, warnings, extract_trace,
              active_stage, stage_started_at, stage_started_perf, merged_all,
              older, raw_timeline_events, cutoff, timeline_quality,
              fallback_warning=""):
    """Run the ported step body. See ``stage.run_layer_stage`` for the contract."""
    return await _run_body(
        req=req, merged=merged, event_dir=event_dir, sem=sem, warnings=warnings,
        extract_trace=extract_trace, active_stage=active_stage,
        stage_started_at=stage_started_at, stage_started_perf=stage_started_perf,
        merged_all=merged_all, older=older,
        raw_timeline_events=raw_timeline_events, cutoff=cutoff,
        timeline_quality=timeline_quality, fallback_warning=fallback_warning,
    )


async def _run_body(*, req, merged, event_dir, sem, warnings, extract_trace,
                    active_stage, stage_started_at, stage_started_perf,
                    merged_all, older, raw_timeline_events, cutoff,
                    timeline_quality, fallback_warning=""):
    if req.resume_profiles:
        from app.storage.json_store import read_json
        directory = event_dir.path() if callable(getattr(event_dir, 'path', None)) else Path(event_dir)
        saved = read_json(directory / 'profile_inputs.json', None)
        if saved:
            if saved.get('checkpoint_key') != req.checkpoint_key:
                raise ValueError('The profile checkpoint differs from the current input. Restore the original model, actor limit, and date range to resume.')
            merged[:] = saved['timeline']
            return await _finish_profiles(req=req, merged=merged, event_dir=event_dir,
                figure_details=saved['figures'], principal_members=saved['members'],
                name_mapping=saved['name_mapping'], quality=saved['quality'],
                timeline_quality=saved.get('timeline_quality', {}), extract_trace=extract_trace,
                fallback_warning=saved.get('fallback_warning', ''))
    stage_timings: dict[str, float] = {}
    logger.info("Step 4a: Extracting figures from each timeline entry")

    _BATCH_SIZE = EXTRACT_BATCH_SIZE
    actor_cache = ActorBatchCache(event_dir, req.model, req.base_url)

    async def _extract_batched(batch: list[dict]) -> list[tuple[dict, list, list, dict]]:
        saved = actor_cache.load(batch)
        if saved is not None:
            return saved
        async with sem:
            try:
                result = await extract_figures_from_batch(
                    req.api_key, req.model, req.event_title, batch,
                    base_url=req.base_url,
                )
                actor_cache.save(batch, result)
                return result
            except Exception as exc:
                if isinstance(exc, BudgetExceeded):
                    raise
                logger.warning(
                    "Step 4a: 批抽取失败（%s），该批记为空，不展开逐条重试",
                    exc, exc_info=True,
                )
                return [
                    (entry, [], [], {"status": "batch_error", "detail": str(exc)[:200]})
                    for entry in batch
                ]

    if not merged:
        entry_results = []
    else:
        batches = [merged[i:i + _BATCH_SIZE] for i in range(0, len(merged), _BATCH_SIZE)]
        logger.info(
            "Step 4a: %d entries → %d batches of %d",
            len(merged), len(batches), _BATCH_SIZE,
        )
        actor_extraction_started = time.perf_counter()
        batched = await gather_complete(*[_extract_batched(b) for b in batches])
        stage_timings["actor_action_extraction"] = round(time.perf_counter() - actor_extraction_started, 3)
        entry_results = [r for group in batched for r in group]

    # Action evidence inherits its already-validated parent event's time.
    for entry, _, actions, _ in entry_results:
        for action in actions:
            if entry.get("timestamp"):
                action["proposed_normalized_timestamp"] = action.get("normalized_timestamp")
                action["normalized_timestamp"] = entry["timestamp"]
                action["timestamp_precision"] = entry.get("date_precision", "month")
                action["time_role"] = entry.get("time_role", "occurrence")
                action["source_timeline_id"] = entry.get("timeline_id")

    figures_found = sum(len(figures) for _, figures, _, _ in entry_results)
    actions_found = sum(len(actions) for _, _, actions, _ in entry_results)
    extraction_status_counts: dict[str, int] = {}
    for _, _, _, diag in entry_results:
        status = (diag or {}).get("status") or "unknown"
        extraction_status_counts[status] = extraction_status_counts.get(status, 0) + 1
    logger.info(
        "Step 4a result: %d/%d entries had figures extracted (%d figures, %d actions total)",
        sum(1 for _, f, _, _ in entry_results if f),
        len(entry_results),
        figures_found,
        actions_found,
    )
    logger.info("Step 4a extraction status breakdown: %s", extraction_status_counts)

    # Entries that yielded nothing are not all failures: the prompt tells the
    # model to return empty for pure statistics/price lines. Surface the split
    # so a genuine extraction regression is not mistaken for correct silence.
    unproductive = {k: v for k, v in extraction_status_counts.items() if k != "ok"}
    unproductive_total = sum(unproductive.values())
    if unproductive_total and merged:
        unproductive_ratio = unproductive_total / len(merged)
        parse_failures = extraction_status_counts.get("parse_failed", 0)
        if parse_failures:
            extraction_warning = (
                f"The model returned non-JSON output for {parse_failures} actor/action batches. "
                "Check that the selected model and Base URL return a final chat completion as JSON; "
                "reasoning-only output cannot be used for actor evidence."
            )
            warnings.append(extraction_warning)
        extraction_warning = (
            f"Figure extraction produced no output for {unproductive_total}/{len(merged)} "
            f"timeline entries ({unproductive_ratio:.0%}): {unproductive}"
        )
        logger.warning(extraction_warning)
        if unproductive_ratio >= 0.2:
            warnings.append(extraction_warning)

    technical_failures = sum(
        extraction_status_counts.get(status, 0)
        for status in (
            "parse_failed", "entry_missing", "empty_content", "reasoning_only",
            "http_error", "batch_error",
        )
    )
    if technical_failures:
        summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(extraction_status_counts.items())
            if count
        )
        raise ValueError(
            "Actor/action extraction is incomplete because some model "
            f"responses could not be processed ({summary}). Check the selected "
            "model and Base URL support OpenAI-compatible JSON chat completions, "
            "then retry the extraction. The Timeline is preserved."
        )

    for entry, figures, actions, diag in entry_results:
        entry["figures"] = figures
        entry["actions"] = actions

    figures_per_entry = [fr for _, fr, _, _ in entry_results]
    actions_per_entry = [ar for _, _, ar, _ in entry_results]

    logger.info("Step 4c: Merging and deduplicating figures")
    actor_merge_started = time.perf_counter()
    merged_figures_result = await merge_figures(
        req.api_key, req.model, req.event_title,
        figures_per_entry, actions_per_entry,
        base_url=req.base_url,
        timeline_entries=merged,
    )
    stage_timings["actor_action_merge"] = round(time.perf_counter() - actor_merge_started, 3)

    if isinstance(merged_figures_result, dict):
        all_figures = merged_figures_result.get("figures", [])
        all_actions = merged_figures_result.get("actions", [])
        name_mapping = merged_figures_result.get("name_mapping", {})
    else:
        logger.warning("Step 4c: merge_figures returned unexpected type: %s", type(merged_figures_result))
        all_figures = []
        all_actions = []
        name_mapping = {}

    if not all_figures:
        raw_figures = [
            f for flist in figures_per_entry
            for f in flist
            if isinstance(f, dict) and f.get("name")
        ]
        if raw_figures:
            logger.warning("Step 4c: merge_figures empty, using %d raw figures", len(raw_figures))
            all_figures = raw_figures
            name_mapping = {}

    if CROSS_BATCH_MERGE_ENABLED:
        try:
            _type_by_name: dict[str, str] = {}
            for flist in figures_per_entry:
                for f in flist:
                    if not isinstance(f, dict):
                        continue
                    _n = _clean_text(f.get("name"))
                    if _n and _n not in _type_by_name:
                        _type_by_name[_n] = f.get("entity_type", "")
            cross = await merge_across_batches(
                req.api_key, req.model, req.event_title,
                sorted(_type_by_name),
                types=_type_by_name,
                base_url=req.base_url,
            )
        except Exception as exc:
            logger.warning("Step 4c: 跨批归并失败（%s），跳过", exc)
            cross = {}
        if cross:
            _added = 0
            for alias, target in cross.items():
                if alias not in name_mapping:
                    name_mapping[alias] = target
                    _added += 1
            logger.info("Step 4c: 跨批归并并入 %d 条别名", _added)
    else:
        logger.info("Step 4c: 跨批归并已关闭（CROSS_BATCH_MERGE_ENABLED=False）")

    from app.knowledge.actor_identity import guard_alias_mapping, discovered_person_aliases
    name_mapping.update(discovered_person_aliases(all_figures))
    name_mapping, alias_review = guard_alias_mapping(name_mapping, all_figures, all_actions, merged)
    previous_alias_review = (merged_figures_result or {}).get('alias_review', {}) if isinstance(merged_figures_result, dict) else {}
    alias_review['rejected'] = [*previous_alias_review.get('rejected', []), *alias_review['rejected']]
    _write_json(event_dir / 'alias_review.json', alias_review)
    name_mapping = _resolve_mapping(name_mapping)
    _broken = break_alias_cycles(name_mapping)
    name_mapping = _resolve_mapping(name_mapping)
    if _broken:
        logger.info("Step 4c: 打破 %d 条环形别名映射", _broken)

    canonical_figures = _merge_canonical_figures(all_figures, name_mapping)
    figure_occurrences, figure_actions, figure_actions_received = _build_figure_context(
        merged, name_mapping,
    )
    scored_figures = _score_figures(
        canonical_figures,
        figure_occurrences,
        figure_actions,
        figure_actions_received,
        req.event_title,
    )
    # After scoring, not before: _score_figures adds records for actors that
    # only ever appear as an action participant, and those would otherwise
    # keep a blank stance.
    _fill_missing_stances(scored_figures, figure_actions, figure_actions_received)
    from app.knowledge.decision_seats import seat_figures, needs_roster_review, prepare_seat_plan
    context = event_dir.events.load_artifact(req.event_id, 'decision_context', {})
    if req.decision_plan and needs_roster_review(req.decision_plan, scored_figures, context.get('question') or req.event_title):
        reviewed = await prepare_seat_plan(req, event_dir.events, candidates=scored_figures, force_review=True)
        req.decision_plan = reviewed
        from app.knowledge.okf import digest
        req.checkpoint_context.update(seat_plan=reviewed, seat_policy=reviewed.get('version'))
        req.checkpoint_key = digest(req.checkpoint_context)
    seat_figures(req.decision_plan, scored_figures)
    if req.decision_plan:
        from app.knowledge.okf import digest
        req.checkpoint_context.update(seat_plan=req.decision_plan, seat_policy=req.decision_plan.get('version'))
        req.checkpoint_key = digest(req.checkpoint_context)
        event_dir.events.save_artifact(req.event_id, 'decision_seat_plan', req.decision_plan)

    participating_objects = [f | {"entity_type": non_agent_kind(f)} for f in scored_figures if not is_decision_subject(f)]
    _write_json(event_dir / "participating_objects.json", participating_objects)
    scored_figures = [f for f in scored_figures if is_decision_subject(f)]
    if participating_objects:
        warnings.append(f"{len(participating_objects)} assets/locations retained as evidence entities, excluded from decision profiles")

    pre_filter_count = len(scored_figures)
    from app.knowledge.actor_evidence import VERSION as AUTHORITY_POLICY, attach_authority_evidence
    source_records = event_dir.events.load_sources(req.event_id)
    evidence_cutoff = req.evidence_cutoff or max(
        (str(source.get('revision_at') or source.get('retrieved_at') or '') for source in source_records),
        default='',
    )
    attach_authority_evidence(scored_figures, sources=source_records,
                              cutoff=evidence_cutoff, event_title=req.event_title)
    scored_figures, filtered_out, relax_applied = _filter_by_min_evidence(
        scored_figures,
        req.min_evidence_ratio,
        req.min_evidence_count,
        len(merged),
        req.min_portrait_figures,
    )
    _write_json(event_dir / 'actor_evidence_review.json', {
        'version': AUTHORITY_POLICY, 'cutoff': evidence_cutoff,
        'retained_authority_candidates': [
            {'name': row['canonical_name'], 'timeline_evidence_count': row.get('evidence_count', 0),
             'source_spans': row['source_authority_evidence']}
            for row in scored_figures if row.get('source_authority_evidence')],
        'filtered_names': [row['canonical_name'] for row in filtered_out],
    })
    if relax_applied:
        logger.warning(
            "Evidence thresholds relaxed by %d: too few figures cleared the "
            "configured floor, need at least %d for key figure selection. "
            "%d figures remain.",
            relax_applied, req.min_portrait_figures, len(scored_figures),
        )
    if filtered_out:
        logger.info(
            "Evidence filter: removed %d/%d figures below %.1f%% of %d timeline "
            "entries (per-entity-type multipliers applied). "
            "%d figures remain for key figure selection.",
            len(filtered_out), pre_filter_count, req.min_evidence_ratio * 100,
            len(merged), len(scored_figures),
        )
        removed_names = [
            f.get("canonical_name", "?") for f in filtered_out[:15]
        ]
        logger.debug("Filtered out (first 15): %s", removed_names)


    relationship_action_evidence = {
        "event_title": req.event_title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "timeline_action_evidence",
        "actions": [
            {
                "source": a.get("source", ""),
                "target": a.get("target", ""),
                "action": a.get("action", ""),
                "result": a.get("result", ""),
                "confidence": a.get("confidence", 0.7),
                "cameo_root_code": a.get("cameo_root_code"),
                "cameo_code": a.get("cameo_code"),
                "action_label": a.get("action_label"),
                "quad_class": a.get("quad_class"),
                "action_confidence": a.get("action_confidence"),
                "timestamp_text": a.get("timestamp_text"),
                "normalized_timestamp": a.get("normalized_timestamp"),
                "timestamp_precision": a.get("timestamp_precision"),
            }
            for a in all_actions
            if isinstance(a, dict)
        ],
    }
    _write_json(event_dir / "relationship_action_evidence.json", relationship_action_evidence)

    _write_json(
        event_dir / "figures_merged.json",
        {
            "event_title": req.event_title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "figures": scored_figures,
            "actions": all_actions,
            "name_mapping": name_mapping,
        },
    )

    logger.info(
        "Step 4c: Merged to %d unique actors, %d actions, %d name mappings",
        len(scored_figures), len(all_actions), len(name_mapping),
    )

    if not scored_figures:
        warning = "No figures extracted; skipping figures.json generation"
        logger.warning(warning)
        warnings.append(warning)
        _write_json(
            event_dir / "figures_merged.json",
            {
                "event_title": req.event_title,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "figures": [],
                "actions": all_actions,
                "name_mapping": name_mapping,
            },
        )
        quality = {
            "timeline_events_total": len(merged_all),
            "timeline_events_used": len(merged),
            "timeline_events_older": len(older),
            "raw_figures_found": figures_found,
            "raw_actions_found": actions_found,
            "extraction_status_counts": extraction_status_counts,
            "merged_figures_found": 0,
            "key_figures_selected": 0,
            # No time window here: the framework applies it before this stage
            # sees the timeline, so this stage has no window of its own.
            "timeline_quality": timeline_quality,
            "warnings": warnings,
        }
        extract_trace.record_stage(
            stage="figure",
            started_at=stage_started_at,
            elapsed_seconds=time.perf_counter() - stage_started_perf,
            status="success",
            details={
                "timeline_entries_processed": len(merged),
                "raw_figures_found": figures_found,
                "raw_actions_found": actions_found,
                "extraction_status_counts": extraction_status_counts,
                "merged_figures_found": 0,
                "key_figures_selected": 0,
                "warnings": warnings,
            },
        )
        return {
            "timeline": merged,
            "figures": [],
            "name_mapping": name_mapping,
            "quality": quality,
            "timeline_quality": timeline_quality,
            "fallback_warning": fallback_warning,
            "extract_trace": extract_trace.public_info(),
        }

    principal_selection_started = time.perf_counter()
    planned_names = [row['canonical_name'] for row in scored_figures if row.get('decision_seat')]
    has_plan = bool(req.decision_plan.get('seats'))
    key_figure_names = planned_names if has_plan else await select_principals(
        req.api_key, req.model, req.event_title, scored_figures,
        max_figures=req.max_key_figures, base_url=req.base_url,
    )
    by_name = {f["canonical_name"]: f for f in scored_figures}
    key_figure_names = [name for name in key_figure_names if name in by_name]
    if not has_plan:
        key_figure_names = _compact_key_figure_names(
            key_figure_names, scored_figures, req.max_key_figures,
        )
    if not key_figure_names and not has_plan:
        key_figure_names = _compact_key_figure_names(
            [], scored_figures, req.max_key_figures,
        )
    logger.info("Step 5a: Selected %d principal actors: %s", len(key_figure_names), key_figure_names)

    # Task 3: audit the principal list for fully-overlapping decision spaces (a state vs
    # its ruling party / government / head of state in a party-state event).
    overlapped = [] if has_plan else await check_principal_overlaps(
        req.api_key, req.model, req.event_title,
        key_figure_names, scored_figures,
        base_url=req.base_url,
    )
    if overlapped:
        logger.info(
            "Step 5a: Overlap check demoted principals: %s", overlapped,
        )
        key_figure_names = [name for name in key_figure_names if name not in overlapped]
        if not key_figure_names:
            key_figure_names = _compact_key_figure_names(
                [], scored_figures, req.max_key_figures,
            )
    stage_timings["principal_selection_and_overlap"] = round(
        time.perf_counter() - principal_selection_started, 3
    )

    _write_json(
        event_dir / "figures_merged.json",
        {
            "event_title": req.event_title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "figures": scored_figures,
            "actions": all_actions,
            "name_mapping": name_mapping,
        },
    )

    # Task 3: for each principal, LLM-assign its internal members (subordinates) from
    # the remaining scored candidates.
    async def _assign_one(principal: str) -> tuple[str, list[dict]]:
        base = by_name.get(principal, {})
        from app.layers.pipeline.llm_figure_service import _country_of

        principal_country = _country_of(base)
        context = (
            f"type={base.get('entity_type','?')}; role_in_event="
            f"{base.get('role_in_event') or base.get('role','?')}; "
            f"title={base.get('title','')}; affiliation={base.get('affiliation','')}; "
            f"country={principal_country or '?'}"
        )
        async with sem:
            members = await assign_subordinates(
                req.api_key, req.model, req.event_title,
                principal, context, scored_figures,
                max_members=MAX_MEMBERS_PER_PRINCIPAL,
                base_url=req.base_url,
                cutoff=cutoff,
                event_timeline=merged,
                excluded_names=key_figure_names,
            )
        return principal, members

    member_assignment_started = time.perf_counter()
    assigned = await gather_complete(*[_assign_one(name) for name in key_figure_names])
    assigned.sort(
        key=lambda pair: _principal_pick_rank(pair[0], by_name),
        reverse=True,
    )
    principal_members: dict[str, list[dict]] = {}
    claimed_members: set[str] = set()
    member_claims: dict[str, list[tuple[str, dict]]] = {}
    for principal, members in assigned:
        enriched = []
        for m in members:
            member_name = m.get("name")
            key = name_key(member_name)
            base = by_name.get(member_name) or {}
            entry = {
                "name": member_name,
                "role_in_entity": m.get("role_in_entity"),
                "id": base.get("id") or _actor_id(m.get("name")),
                "entity_type": base.get("entity_type"),
                "title": base.get("title"),
                "affiliation": base.get("affiliation"),
                "stance": base.get("stance"),
            }
            if key:
                claims = member_claims.setdefault(key, [])
                claims.append((principal, entry))
                if len(claims) == 1:
                    enriched.append(entry)
        principal_members[principal] = enriched

    # Ownership decisions for different contested members do not depend on
    # each other. Resolve them concurrently, then apply them in stable claim
    # order so the fallback owner remains deterministic.
    async def _resolve_claim(key: str, claims: list[tuple[str, dict]]):
        claimants = list(dict.fromkeys(principal for principal, _ in claims))
        if len(claimants) < 2:
            return None
        winner = claimants[0]
        winner_entry = next(e for p, e in claims if p == winner)
        contexts = {}
        for principal in claimants:
            base = by_name.get(principal) or {}
            contexts[principal] = (
                f"type={base.get('entity_type','?')}; role_in_event="
                f"{base.get('role_in_event') or base.get('role','?')}; "
                f"title={base.get('title','')}; affiliation={base.get('affiliation','')}"
            )
        try:
            async with sem:
                owner = await resolve_member_ownership(
                    req.api_key, req.model, req.event_title,
                    winner_entry.get("name") or "",
                    claimants, contexts,
                    event_timeline=merged,
                    base_url=req.base_url,
                )
        except Exception:
            owner = ""
        return key, winner, winner_entry, claimants, owner

    ownership_results = await gather_complete(*(
        _resolve_claim(key, claims) for key, claims in member_claims.items()
    ))
    for result in ownership_results:
        if result is None:
            continue
        key, winner, winner_entry, claimants, owner = result
        if owner and owner != winner:
            bucket = principal_members.get(winner)
            if bucket is not None:
                principal_members[winner] = [
                    e for e in bucket if name_key(e.get("name")) != key
                ]
            principal_members.setdefault(owner, []).append(winner_entry)
            logger.info(
                "Step 5a: Member ownership resolved %s -> %s (contested by %s)",
                winner_entry.get("name"), owner, claimants,
            )
        else:
            logger.info(
                "Step 5a: Member %s kept by %s (contested by %s, LLM verdict: %s)",
                winner_entry.get("name"), winner,
                [p for p, _ in claims[1:]], owner or "none",
            )

    for bucket in principal_members.values():
        for entry in bucket:
            entry_key = name_key(entry.get("name"))
            if entry_key:
                claimed_members.add(entry_key)
    if overlapped:
        claimed_lower = claimed_members
        restored = []
        for name in overlapped:
            if name in key_figure_names:
                continue
            if name_key(name) not in claimed_lower:
                key_figure_names.append(name)
                principal_members.setdefault(name, [])
                restored.append(name)
        if restored:
            logger.info(
                "Step 5a: Restored demoted principals with no member slot: %s",
                restored,
            )

    audit = await audit_layer_assignment(
        req.api_key, req.model, req.event_title,
        key_figure_names, principal_members, scored_figures,
        base_url=req.base_url,
    )
    for name in audit.get("missing_demote") or []:
        if name in planned_names:
            continue  # Hierarchy cannot erase a registered independent decision.
        # The audited actor's decisions are its container's in this event (US + Trump).
        if name not in key_figure_names:
            continue
        container = find_container(name, key_figure_names, scored_figures)
        if not container:
            continue
        key_figure_names.remove(name)
        principal_members.pop(name, None)
        base = by_name.get(name) or {}
        member_id = base.get("id") or _actor_id(name)
        demoted_key = name_key(name)
        if demoted_key:
            claimed_members.add(demoted_key)
        principal_members.setdefault(container, []).append({
            "name": name,
            "role_in_entity": _clean_text(base.get("role_in_event")) or "sub-unit",
            "id": member_id,
            "entity_type": base.get("entity_type"),
            "title": base.get("title"),
            "affiliation": base.get("affiliation"),
            "stance": base.get("stance"),
        })
        logger.info(
            "Step 5a: Layer audit demoted %s -> member of %s", name, container,
        )
    # Two audit verdicts share one remedy: the named member is not an organ the
    # principal acts through, so it comes off the list.
    for verdict, label in (("conflict_member", "conflicting"), ("foreign_member", "foreign")):
        for fix in audit.get(verdict) or []:
            principal = fix.get("principal")
            member = fix.get("member")
            members = principal_members.get(principal) or []
            before = len(members)
            drop_key = name_key(member)
            principal_members[principal] = [
                m for m in members if name_key(m.get("name")) != drop_key
            ]
            if len(principal_members[principal]) < before:
                logger.info(
                    "Step 5a: Layer audit dropped %s member %s from %s (%s)",
                    label, member, principal, fix.get("reason", ""),
                )
    total_members = sum(len(v) for v in principal_members.values())
    logger.info(
        "Step 5a: Assigned %d subordinates across %d principals",
        total_members, len(key_figure_names),
    )
    if total_members:
        for p, ms in principal_members.items():
            logger.info("  %s -> %s", p, ", ".join(m["name"] for m in ms))
    stage_timings["member_assignment_and_audit"] = round(
        time.perf_counter() - member_assignment_started, 3
    )

    async def _generate_detail(name: str) -> dict:
        base = by_name.get(name, {"canonical_name": name, "figure": name, "id": _actor_id(name)})
        initiated = figure_actions.get(name, [])
        received = figure_actions_received.get(name, [])
        # Build a source-backed factual summary locally.
        detail = {
            "figure": name,
            "history": figure_occurrences.get(name, []),
        }
        return _compose_figure_profile(
            {**base, "aliases": list(base.get("aliases") or [])},
            detail,
            figure_occurrences.get(name, []),
            initiated,
            received,
        )

    figure_details = await gather_complete(*[
        _generate_detail(name) for name in key_figure_names
    ])

    quality = {
        "timeline_events_total": len(merged_all),
        "timeline_events_used": len(merged),
        "timeline_events_older": len(older),
        "raw_timeline_events": len(raw_timeline_events),
        "raw_figures_found": figures_found,
        "raw_actions_found": actions_found,
        "extraction_status_counts": extraction_status_counts,
        "merged_figures_found": len(scored_figures),
        "key_figures_selected": len(figure_details),
        # See the note on the empty-result branch: the time window belongs to
        # the framework's timeline stage, not here.
        "cutoff": cutoff,
        "timeline_quality": timeline_quality,
        "stage_timings_seconds": stage_timings,
        "warnings": warnings,
    }

    logger.info("Step 5: Retired figures.json artifact")

    extract_trace.record_stage(
        stage="figure",
        started_at=stage_started_at,
        elapsed_seconds=time.perf_counter() - stage_started_perf,
        status="success",
        details={
            "timeline_entries_processed": len(merged),
            "raw_figures_found": figures_found,
            "raw_actions_found": actions_found,
            "extraction_status_counts": extraction_status_counts,
            "merged_figures_found": len(scored_figures),
            "key_figures_selected": len(figure_details),
            "warnings": warnings,
        },
    )
    active_stage = "profile"
    stage_started_at = datetime.now(timezone.utc).isoformat()
    stage_started_perf = time.perf_counter()

    return await _finish_profiles(req=req, merged=merged, event_dir=event_dir,
        figure_details=figure_details, principal_members=principal_members,
        name_mapping=name_mapping, quality=quality, timeline_quality=timeline_quality,
        extract_trace=extract_trace, fallback_warning=fallback_warning)


def _figure_subject_keys(figure):
    return {contract_subject_key(name) for name in [
        figure.get("canonical_name") or figure.get("name") or figure.get("figure"),
        *(figure.get("aliases") or []),
    ]} - {""}


def _retain_required_subject(subject, candidates, timeline, name_mapping, figures, members, cap):
    """Keep a contract's observed decision actor eligible for dossier review."""
    matches = matching_subjects(subject, [row for row in candidates if is_decision_subject(row)],
                                lambda row: [row.get("canonical_name") or row.get("name") or row.get("figure"),
                                             *(row.get("aliases") or [])],
                                lambda row: row.get("entity_type") == "person")
    if len(matches) != 1:
        return figures, members, "ambiguous" if matches else "not_extracted"
    candidate = matches[0]
    name = candidate["canonical_name"]
    keys = _figure_subject_keys(candidate)
    supported = [row for row in timeline if row.get("date_evidence", {}).get("status") == "supported"
                 and row.get("evidence_text")]
    occurrences, initiated, received = _build_figure_context(
        [dict(row) for row in supported], name_mapping,
    )
    history = [entry for actor, entries in occurrences.items() if contract_subject_key(actor) in keys
               for entry in entries]
    actions = [entry for actor, entries in initiated.items() if contract_subject_key(actor) in keys
               for entry in entries]
    if not actions:
        return figures, members, "no_supported_action"
    if not any(keys & _figure_subject_keys(row) for row in figures):
        # Names and positions on the latest supported action take precedence
        # over an older generic title from the entity merge.
        action_ids = {action["timeline_id"] for action in actions}
        matching_figures = [
            figure for row in sorted(supported, key=lambda item: item.get("timestamp", ""), reverse=True)
            if row.get("timeline_id") in action_ids
            for figure in row.get("figures", [])
            if keys & _figure_subject_keys(figure) and figure.get("title")
        ]
        if matching_figures:
            candidate = {**candidate, "title": matching_figures[0]["title"]}
        dossier = _compose_figure_profile(candidate, {"figure": name}, history, actions,
                                          received.get(name, []))
        figures = [dossier, *figures][:cap]
    retained_names = {row["canonical_name"] for row in figures}
    members = {principal: [row for row in group if not keys & _figure_subject_keys(row)]
               for principal, group in members.items() if principal in retained_names}
    return figures, members, "retained"


_POLL = re.compile(r"\b(?:poll(?:ing|s|ster|sters)?|survey(?:s|ed)?|fieldwork)\b|民调|民意调查", re.I)
_POLL_ONLY_ROLE = re.compile(r"\bpolling (?:firm|company|organization|agency|institute)\b|\bpollster\b|\bbloc\b.*\bpoll\b|民调机构|民调中.*阵营", re.I)


def _poll_only_figure(figure):
    if not _POLL_ONLY_ROLE.search(figure.get("role_in_event") or ""):
        return False
    descriptions = [row.get("description") for row in figure.get("history", [])]
    descriptions.extend(row.get("action") or row.get("action_label")
                        for row in figure.get("actions_initiated", []))
    recorded = [text for text in descriptions if text]
    return bool(recorded) and all(_POLL.search(text) for text in recorded)


async def _finish_profiles(*, req, merged, event_dir, figure_details, principal_members,
                           name_mapping, quality, timeline_quality, extract_trace, fallback_warning):
    directory = event_dir.path() if callable(getattr(event_dir, "path", None)) else Path(event_dir)
    from app.storage.json_store import read_json
    old_inputs = read_json(directory / 'profile_inputs.json', {})
    prior_figures = quality.pop("profile_checkpoint_figures", old_inputs.get('figures') or figure_details)
    if req.required_decision_subject and not req.decision_plan.get('seats'):
        from app.storage.json_store import read_json

        scored = read_json(directory / "figures_merged.json", {}).get("figures", [])
        figure_details, principal_members, reason = _retain_required_subject(
            req.required_decision_subject, scored, merged, name_mapping, figure_details,
            principal_members, req.max_key_figures,
        )
        quality["required_subject_review"] = {"subject": req.required_decision_subject, "status": reason}
        quality["key_figures_selected"] = len(figure_details)
    elif req.required_decision_subject:
        quality['required_subject_review'] = {'subject':req.required_decision_subject,
                                               'status':'reviewed_in_decision_plan'}
    if req.exclude_poll_only_profiles:
        kept = [row for row in figure_details if not _poll_only_figure(row)]
        excluded = len(figure_details) - len(kept)
        if excluded:
            retained_names = {row["canonical_name"] for row in kept}
            principal_members = {name: rows for name, rows in principal_members.items() if name in retained_names}
            figure_details = kept
            quality["poll_only_profiles_excluded"] = excluded
            quality["key_figures_selected"] = len(figure_details)
    # A roster refinement can reuse unchanged per-actor inputs from the prior
    # batch. Preserve that list in the private checkpoint across interruptions.
    checkpoint_quality = (quality | {"profile_checkpoint_figures": prior_figures}
                          if prior_figures != figure_details else quality)
    # Save all expensive actor selection and ownership work BEFORE spending on
    # dossiers. A failed profile call must never require repeating those steps.
    _write_json(event_dir / 'profile_inputs.json', {
        'version': 1, 'checkpoint_key': req.checkpoint_key, 'figures': figure_details,
        'members': principal_members, 'timeline': merged, 'name_mapping': name_mapping,
        'quality': checkpoint_quality, 'timeline_quality': timeline_quality, 'fallback_warning': fallback_warning,
    })
    warnings = quality.setdefault('warnings', [])
    quality['decision_seat_plan'] = {
        'status': req.decision_plan.get('status', 'unavailable'),
        'requirements': req.decision_plan.get('requirements', []),
        'gaps': req.decision_plan.get('gaps', []),
    }
    stage_timings = quality.setdefault('stage_timings_seconds', {})
    stage_started_at = datetime.now(timezone.utc).isoformat()
    stage_started_perf = time.perf_counter()

    profiles: list[dict] = []
    profile_path = ""
    profile_warning = ""
    profile_started = time.perf_counter()
    try:
        profiles, power_dimensions = await generate_profiles(
            api_key=req.api_key,
            model=req.model,
            event_title=req.event_title,
            figures=figure_details,
            timeline=merged,
            base_url=req.base_url,
            members_by_principal=principal_members,
            case_id=req.event_id,
            resume_profiles=req.resume_profiles,
            checkpoint_dir=directory / "profile_checkpoints",
            previous_figures=prior_figures if prior_figures != figure_details else None,
            evidence_cutoff=req.evidence_cutoff,
        )
        if any(p.get('member_traits_status') == 'not_generated_budget_limit' for p in profiles):
            warnings.append('Actor profiles are complete. Member decision styles remain unknown under the current budget.')
        profile_path = write_profiles_artifact(
            req.event_title, profiles, power_dimensions, event_dir=event_dir,
        )
        logger.info("Step 6: Saved profiles.json with %d profiles (%s)", len(profiles), profile_path)
        extract_trace.record_stage(
            stage="profile",
            started_at=stage_started_at,
            elapsed_seconds=time.perf_counter() - stage_started_perf,
            status="success",
            details={
                "profile_count": len(profiles),
            },
            complete_run=True,
        )
    except Exception as exc:
        profile_warning = f"Some actor profiles are incomplete. The timeline, actor selection, and profile checkpoint were saved. {exc}"
        logger.warning(profile_warning, exc_info=True)
        warnings.append(profile_warning)
        extract_trace.record_stage(
            stage="profile",
            started_at=stage_started_at,
            elapsed_seconds=time.perf_counter() - stage_started_perf,
            status="failed",
            error=str(exc),
        )
        quality.update(profile_status='incomplete', profile_error=str(exc))
        _write_json(event_dir / 'layer_quality.json', quality)
        raise
    stage_timings["profile_generation"] = round(time.perf_counter() - profile_started, 3)
    quality.update(profile_status='completed', profile_error='')

    profiles_view: dict = {}
    if profiles:
        try:
            profiles_view = build_profiles_view(profiles)
        except Exception:
            logger.warning("build_profiles_view failed; omitting profiles_view", exc_info=True)

    return {
        "timeline": merged,
        "figures": figure_details,
        "profiles": profiles,
        "profiles_view": profiles_view,
        "profile_path": profile_path,
        "name_mapping": name_mapping,
        "quality": quality,
        "timeline_quality": timeline_quality,
        "fallback_warning": fallback_warning,
        "profile_warning": profile_warning,
        "extract_trace": extract_trace.public_info(),
    }
