import asyncio

import pytest

from app.application.forecasting import ForecastRoundOrchestrator
from app.domain import GameMasterSettlement, RelationshipProjection, TimelineEvent, WorldSnapshot
from app.domain.forecast import SequentialGameMasterSettlement, WorldStateChange
from app.functional_agents.simulation import GameMasterAgent
from app.simulation.transitions import apply_round_update, project_world_state


def test_sparse_ordinary_settlement_preserves_nested_commitment():
    original = WorldSnapshot(event_title="Supply", world_state={
        "commitment": {"recipient": "Board", "status": "proposed", "terms": ["inspection"]},
        "confirmed": True,
        "measurement_paths": ["commitment.status"],
    })
    settlement = GameMasterSettlement(state_summary="Accepted", analysis="",
                                      next_world_state={"commitment": {"status": "accepted"},
                                                        "measurement_paths": None, "state_invariants": None})
    updated = apply_round_update(original, [], settlement, RelationshipProjection())
    assert updated.world_state == {
        "commitment": {"recipient": "Board", "status": "accepted", "terms": ["inspection"]},
        "confirmed": True,
        "measurement_paths": ["commitment.status"],
    }
    assert original.world_state["commitment"]["status"] == "proposed"


def _snapshot():
    return WorldSnapshot(event_title="Supply", world_state={
        "permit": {"status": "filed", "recipient": "Board"},
        "offer": {"expires_on": "2025-01-08", "terms": ["inspection"]},
        "benchmark_contract": {"question": "Will the permit be issued?"},
        "simulation_clock": {"interval_start": "2025-01-01", "interval_end": "2025-01-08"},
    })


def _permit_change(**overrides):
    return WorldStateChange.model_validate({
        "path": ["permit", "status"], "operation": "replace", "before": "filed", "after": "approved",
        "reason": "The approval took effect.", "event_ref": "sim_1_permit",
        "evidence_quote": "The board approved the permit", **overrides,
    })


def _sequential(*changes):
    return SequentialGameMasterSettlement(
        state_summary="The board approved the permit.", analysis="", state_changes=list(changes),
        public_events=[TimelineEvent(timeline_id="sim_1_permit", timestamp="2025-01-07",
                                     description="The board approved the permit.", confidence=1, importance=1)],
    )


def test_sequential_delta_applies_validated_evidence_and_expiry():
    original = _snapshot()
    change = WorldStateChange(path=["offer"], operation="expire", before=original.world_state["offer"],
                              after=None, reason="The declared expiry date was reached.")
    settlement = _sequential(_permit_change(), change)
    updated = apply_round_update(original, [], settlement, RelationshipProjection())
    assert updated.world_state["permit"] == {"status": "approved", "recipient": "Board"}
    assert "offer" not in updated.world_state and "offer" in original.world_state
    assert updated.world_state["benchmark_contract"] == original.world_state["benchmark_contract"]


@pytest.mark.parametrize("overrides,error", [
    ({"before": "withdrawn"}, "before-value"),
    ({"event_ref": "unretained"}, "retained public event"),
    ({"path": ["benchmark_contract", "question"]}, "program-owned"),
    ({"path": ["permit"]}, "program-owned"),
])
def test_sequential_delta_rejects_stale_proofs_and_program_overwrites(overrides, error):
    original = _snapshot()
    with pytest.raises(ValueError, match=error):
        project_world_state(original, _sequential(_permit_change(**overrides)), [],
                            {"permit.recipient": "Board"})
    assert original.world_state["permit"] == {"status": "filed", "recipient": "Board"}


def test_sequential_game_master_repairs_only_invalid_delta_then_commits():
    original = _snapshot()
    calls = []

    class Runtime:
        async def json_call(self, **call):
            calls.append(call)
            assert call["schema"] is SequentialGameMasterSettlement
            return _sequential(_permit_change(before="withdrawn" if len(calls) == 1 else "filed"))

    gm = GameMasterAgent(Runtime())
    gm.sequential_outcome = True
    settlement = asyncio.run(gm.settle(original, [], [], "repair"))
    assert len(calls) == 2 and "before-value" in calls[-1]["prompt"]
    assert project_world_state(original, settlement, [])["permit"]["status"] == "approved"


def test_resume_after_downstream_failure_reuses_validated_settlement(tmp_path):
    original = _snapshot()
    orchestrator = ForecastRoundOrchestrator(object(), checkpoint_dir=tmp_path,
                                               bdi_updates=False, sequential_outcome=True)
    calls = []

    async def select(snapshot, *_):
        calls.append("select")
        return snapshot, [], []

    async def produce(*_):
        calls.append("produce")
        return []

    async def settle(*_):
        calls.append("settle")
        return _sequential(_permit_change())

    async def update(*_):
        calls.append("update")
        if calls.count("update") == 1:
            raise RuntimeError("Downstream step interrupted")
        return RelationshipProjection()

    orchestrator.activation_master.select = select
    orchestrator.action_auditor.produce = produce
    orchestrator.game_master.settle = settle
    orchestrator.relationship_agent.update = update

    async def run():
        with pytest.raises(RuntimeError, match="interrupted"):
            await orchestrator.run(original, [], "round_one")
        return await orchestrator.run(original, [], "round_one")

    updated, frame = asyncio.run(run())
    assert calls == ["select", "produce", "settle", "update", "update"]
    assert original.world_state["permit"]["status"] == "filed"
    assert updated.world_state["permit"]["status"] == "approved"
    assert frame.state_changes[0].event_ref == "sim_1_permit"
