"""One canonical path, built from repeated independent next-step forecasts."""

from __future__ import annotations

import asyncio
import logging

from app.application.forecasting import ForecastRoundOrchestrator
from app.domain import (
    AgentKnowledgeUpdate, ForecastOutcome, ForecastRound,
    GameMasterSettlement, PowerImpact, RelationshipProjection, SelectedAction, WorldSnapshot,
)
from app.domain.identity import identity_key
from app.functional_agents.event_consensus import EventConsensusAgent
from app.functional_agents.relationships import RelationshipAgent
from app.llm.client import LLMClient
from app.simulation.bdi_updates import process_agent_information
from app.simulation.registry import maintain_registry
from app.simulation.transitions import apply_round_update
from app.storage.run_logger import audit_record

logger = logging.getLogger(__name__)

# How many times one sample is re-run before the round gives up on it.
SAMPLE_ATTEMPTS = 3


def _consensus_power_impacts(
    samples: list[ForecastRound],
    required: int,
) -> list[PowerImpact]:
    """Fold the samples' strength verdicts into one set, the way events are voted."""
    tally: dict[tuple[str, str], dict[str, list[PowerImpact]]] = {}
    for sample in samples:
        # One sample may state the same dimension twice; the first wins, which
        # matches the one-move-per-dimension-per-round rule the settlement runs
        # under and keeps a single sample from voting twice.
        seen: set[tuple[str, str]] = set()
        for impact in sample.power_impacts or []:
            status = str(impact.status or "").strip().lower()
            if not status.startswith(("strengthen", "weaken")):
                continue
            key = (identity_key(impact.agent), identity_key(impact.dimension))
            if not all(key) or key in seen:
                continue
            seen.add(key)
            tally.setdefault(key, {}).setdefault(status, []).append(impact)

    agreed: list[PowerImpact] = []
    for (actor_key, dimension_key), by_status in tally.items():
        status, impacts = max(by_status.items(), key=lambda item: len(item[1]))
        if len(impacts) < required:
            continue
        # Keep the wording of the last sample that agreed, so the actor name
        # and dimension spelling that reach the strength settlement are ones a
        # model actually wrote -- the normalised keys are for tallying only.
        agreed.append(impacts[-1].model_copy(update={"status": status}))
    return agreed


async def forecast_consensus_step(
    snapshot: WorldSnapshot, selected_actions: list[SelectedAction], client: LLMClient,
    call_prefix: str, prediction_repeats: int, event_threshold_percent: int,
) -> tuple[WorldSnapshot, ForecastRound]:
    # Only already committed information may change canonical BDI before sampling.
    frozen = await process_agent_information(maintain_registry(snapshot), client, call_prefix)

    async def sample(number: int) -> ForecastRound:
        _, result = await ForecastRoundOrchestrator(client).run(
            frozen.model_copy(deep=True),
            [a.model_copy(deep=True) for a in selected_actions],
            f"{call_prefix}_sample_{number:02d}",
        )
        return result

    async def sample_with_retries(number: int) -> ForecastRound:
        """Run one sample, retrying it in place before giving up on it."""
        last_error: Exception | None = None
        for attempt in range(1, SAMPLE_ATTEMPTS + 1):
            try:
                return await sample(number)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "%s sample %d failed on attempt %d/%d: %s",
                    call_prefix, number, attempt, SAMPLE_ATTEMPTS, exc,
                )
        assert last_error is not None
        raise last_error

    settled = await asyncio.gather(
        *(sample_with_retries(i) for i in range(1, prediction_repeats + 1)),
        return_exceptions=True,
    )
    failed = [item for item in settled if isinstance(item, BaseException)]
    if failed:
        raise failed[0]
    samples = list(settled)
    matcher = EventConsensusAgent(client)
    consensus = await matcher.match(frozen, samples, event_threshold_percent, call_prefix)
    retained = [vote.event for vote in consensus.events if vote.retained]
    if retained:
        projection = await matcher.project(frozen, retained, call_prefix)
        settlement = GameMasterSettlement(
            **projection.model_dump(), public_events=retained,
            outcomes=[ForecastOutcome(
                description=v.event.description,
                probability=v.occurrence_count / prediction_repeats,
                affected_agent_ids=v.event.actors,
            ) for v in consensus.events if v.retained],
            agent_knowledge_updates=[AgentKnowledgeUpdate(agent_id=a.agent_id) for a in frozen.agents],
            # Profiles stay fixed, but public standing still moves: the capability
            # levels step when the samples agree on a verdict.
            power_impacts=_consensus_power_impacts(samples, consensus.required_occurrences),
        )
        relationship_projection = await RelationshipAgent(client).update(
            frozen, settlement, call_prefix
        )
    else:
        settlement = GameMasterSettlement(
            state_summary=frozen.state_summary, next_world_state=frozen.world_state,
            analysis="No events met the occurrence threshold; the previous world state is preserved.",
        )
        relationship_projection = RelationshipProjection()
    # Never import candidate actions, BDI, private inboxes, or world snapshots.
    updated = apply_round_update(frozen, [], settlement, relationship_projection)
    result = ForecastRound(
        step=updated.round_id, input_snapshot_version=snapshot.snapshot_version,
        output_snapshot_version=updated.snapshot_version,
        state_summary=settlement.state_summary, analysis=settlement.analysis,
        outcomes=settlement.outcomes, timeline=retained,
        new_relationships=relationship_projection.relationship_updates,
        agent_updates=[],
        next_watchpoints=settlement.next_watchpoints, event_consensus=consensus,
    )
    audit_record(
        getattr(client, "recorder", None), source="event-consensus", round_id=updated.round_id,
        agent_id="all", content_type="event-votes-and-canonical-state", status="success",
        data={"round": result.model_dump(), "snapshot": updated.model_dump()},
    )
    return updated, result
