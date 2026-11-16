"""Forecast entry points; one-round policy lives in the application orchestrator."""

from __future__ import annotations

import logging
from typing import Any

from app.agents import AgentSDKClient
from app.application.forecasting import ForecastRoundOrchestrator
from app.domain import (
    AgentProfile,
    ForecastRound,
    Relationship,
    SelectedAction,
    SimulationRequest,
    SimulationResponse,
    TimelineEvent,
    Trajectory,
    WorldSnapshot,
)
from app.llm.client import LLMClient
from app.simulation.registry import eligible_simulation_inputs, maintain_registry, profile_from_input
from app.simulation.consensus import forecast_consensus_step
from app.storage.json_store import SnapshotRepository, TraceRecorder
from app.storage.run_logger import audit_finish

logger = logging.getLogger(__name__)

# How many times one round is re-run before the forecast gives up on it.
ROUND_ATTEMPTS = 3


def _recorded_only(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the unsupported ``origin="projected"`` rows from an incoming timeline."""
    return [item for item in items if item.get("origin") != "projected"]


def build_snapshot(
    *,
    event_id: str | None,
    event_title: str,
    background: str,
    raw_agents: list[dict[str, Any]],
    recent_timeline: list[dict[str, Any]],
    raw_relationships: list[dict[str, Any]],
    snapshots: SnapshotRepository,
) -> WorldSnapshot:
    """Load the canonical snapshot or construct the initial simulation boundary."""
    if event_id:
        committed = snapshots.get(event_id)
        if committed:
            return WorldSnapshot.model_validate(committed)
    agents = [profile_from_input(item, event_id, event_title) for item in eligible_simulation_inputs(raw_agents)]
    snapshot = WorldSnapshot(
        event_id=event_id,
        event_title=event_title,
        background=background,
        state_summary=background,
        world_state={"forecast_status": "initialized"},
        strength_dimensions=_strength_dimensions(agents),
        agents=agents,
        timeline=[TimelineEvent.model_validate(item) for item in _recorded_only(recent_timeline)],
        relationships=[Relationship.model_validate(item) for item in raw_relationships],
    )
    return maintain_registry(snapshot)


def _strength_dimensions(agents: list[AgentProfile]) -> list[str]:
    """The event's strength dimensions, in the order the profiles list them."""
    dimensions: list[str] = []
    for agent in agents:
        for capability in agent.capabilities:
            if capability.name and capability.name not in dimensions:
                dimensions.append(capability.name)
    return dimensions


async def forecast_one_step(
    snapshot: WorldSnapshot,
    selected_actions: list[SelectedAction],
    client: LLMClient,
    call_prefix: str,
) -> tuple[WorldSnapshot, ForecastRound]:
    return await ForecastRoundOrchestrator(client).run(snapshot, selected_actions, call_prefix)


async def run_trajectory(
    snapshot: WorldSnapshot,
    steps: int,
    client: LLMClient,
    candidate_id: str,
    selected_actions: list[SelectedAction],
    prediction_repeats: int = 4,
    event_threshold_percent: int = 50,
) -> Trajectory:
    current = snapshot.model_copy(deep=True)
    rounds: list[ForecastRound] = []
    for local_step in range(1, steps + 1):
        started_from = current
        last_error: Exception | None = None
        for attempt in range(1, ROUND_ATTEMPTS + 1):
            try:
                current, round_result = await forecast_consensus_step(
                    # Retrying a round must start from the state it began with.
                    # ``forecast_consensus_step`` returns a new snapshot, so a failed
                    # attempt would otherwise leave the loop carrying a half-applied
                    # state into the retry.
                    started_from.model_copy(deep=True),
                    selected_actions if local_step == 1 else [],
                    client,
                    f"{candidate_id}_round_{local_step:02d}",
                    prediction_repeats,
                    event_threshold_percent,
                )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "%s round %d failed on attempt %d/%d: %s",
                    candidate_id, local_step, attempt, ROUND_ATTEMPTS, exc,
                )
        if last_error is not None:
            raise last_error
        rounds.append(round_result)
    return Trajectory(candidate_id=candidate_id, rounds=rounds, final_snapshot=current)


async def run_linear_forecast(request: SimulationRequest, steps: int) -> SimulationResponse:
    recorder = TraceRecorder("forecast_runs")
    recorder.write(
        "linear_forecast_run",
        "configuration",
        request.model_dump(exclude={"api_key"}) | {"resolved_steps": steps},
    )
    client = AgentSDKClient(request, recorder)
    snapshots = SnapshotRepository()
    # Both fields carry the same agent list under different names: the single-step
    # endpoint sends ``all_agents``, the multi-step one sends ``agents``.
    raw_agents = request.agents or request.all_agents
    if not raw_agents and (request.agents != request.all_agents):
        raise ValueError(
            "The forecast received an empty agent list in one field and a "
            "populated one in the other. Send the agents in exactly one of "
            "'agents' or 'all_agents'."
        )
    base = build_snapshot(
        event_id=request.event_id,
        event_title=request.event_title,
        background=request.background,
        raw_agents=raw_agents,
        recent_timeline=request.recent_timeline,
        raw_relationships=request.relationships,
        snapshots=snapshots,
    )
    expected_version = base.snapshot_version
    trajectory = await run_trajectory(
        base, steps, client, "linear", request.selected_actions,
        request.prediction_repeats, request.event_threshold_percent,
    )
    if request.event_id:
        snapshots.commit(
            request.event_id,
            trajectory.final_snapshot.model_dump(),
            expected_version=expected_version,
        )
    rounds = trajectory.rounds
    response = SimulationResponse(
        mode="single_step" if steps == 1 else "multi_step",
        prediction_repeats=request.prediction_repeats,
        event_threshold_percent=request.event_threshold_percent,
        probability_note="Event frequencies count independent predictions, not calibrated real-world probabilities.",
        analysis="\n\n".join(item.analysis for item in rounds if item.analysis),
        state_summary=trajectory.final_snapshot.state_summary,
        final_state=trajectory.final_snapshot.state_summary,
        steps=rounds,
        agent_actions=[item for round_item in rounds for item in round_item.agent_actions],
        outcomes=[item for round_item in rounds for item in round_item.outcomes],
        timeline=[item for round_item in rounds for item in round_item.timeline],
        new_relationships=[item for round_item in rounds for item in round_item.new_relationships],
        # Always empty: profiles are fixed at initialisation and no round rewrites them,
        # so no round produces a public agent update.
        agent_updates=[],
        updated_context=trajectory.final_snapshot,
        run_id=recorder.run_id,
    )
    audit_finish(
        recorder,
        success=True,
        data={
            "mode": response.mode,
            "rounds_completed": len(rounds),
            "final_snapshot_version": trajectory.final_snapshot.snapshot_version,
        },
    )
    return response
