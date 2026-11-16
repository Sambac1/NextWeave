import asyncio
import json

import httpx
import pytest

from app.domain import LLMOptions, WorldSnapshot, TimelineEvent, PrivateObservation
from app.domain.agency import non_agent_kind
from app.domain.ensemble import EnsembleRequest, OutcomePlan, OutcomeDefinition
from app.knowledge.brief import EvidenceBrief, BriefExcerpt, render_brief
from app.knowledge.hierarchy import public_profiles
from app.layers.support.layer_rules import infer_entity_type
from app.llm.budget import (
    ModelBudget,
    BudgetExceeded,
    budget_scope,
    reserve_http_request,
    settle_http_response,
)
from app.llm.http_pool import _ClientLease
from app.simulation.bdi_updates import process_agent_information
from app.simulation.context import actor_context, public_context, compact_json
from app.simulation.ensemble import prepare_experiment, aggregate
from app.simulation.registry import profile_from_input
from app.storage.json_store import EventRepository
from app.storage.ensemble_store import EnsembleStore
from tests.test_agent_protocol import agent


@pytest.mark.parametrize(
    "description,kind",
    [
        ("container ship", "asset"),
        ("container ship owned by a Japanese company", "asset"),
        ("ship operated by its crew", "asset"),
        ("Italian tug assisting refloating operations", "asset"),
        ("ship captain", None),
        ("captain of a container ship", None),
        ("shipping company", None),
        ("canal authority", None),
        ("canal", "location"),
    ],
)
def test_asset_head_is_distinct_from_responsible_person_or_organization(
    description, kind
):
    assert non_agent_kind({"agent_type": "person", "description": description}) == kind


def test_asset_cannot_enter_new_or_legacy_simulator_but_stays_discoverable():
    assert infer_entity_type("Named Vessel", "person", "container ship") == "asset"
    raw = agent("ship", "Named Vessel").model_dump() | {"title": "container ship"}
    with pytest.raises(ValueError, match="is an asset, place, or event"):
        profile_from_input(raw, None, "A shipping incident")
    records, simulation_ids = public_profiles([raw], "case", "Shipping")
    assert records[0]["agent_type"] == "asset"
    assert records[0]["decision_role"] == "mentioned"
    assert "behavioral_model" not in records[0] and "ship" not in simulation_ids
    assert (
        raw["agent_type"] == "person"
    )  # generated projection never rewrites original data


def test_transport_failure_releases_waiters_but_keeps_unknown_charge():
    async def run():
        ledger = ModelBudget(LLMOptions(max_total_tokens=1000))

        async def transport(request):
            await asyncio.sleep(0.01)
            raise httpx.ConnectError("offline", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport),
            event_hooks={
                "request": [reserve_http_request],
                "response": [settle_http_response],
            },
        ) as client:
            async with _ClientLease(client) as lease:
                with budget_scope(ledger):
                    results = await asyncio.wait_for(
                        asyncio.gather(
                            *[
                                lease.post(
                                    "https://test.example/chat/completions",
                                    json={"messages": [], "max_tokens": 700},
                                )
                                for _ in range(2)
                            ],
                            return_exceptions=True,
                        ),
                        1,
                    )
        assert not ledger.inflight and ledger.calls == 1
        assert ledger.report()["unknown_usage_requests"] == 1
        assert any(isinstance(r, BudgetExceeded) for r in results)
        assert any(isinstance(r, httpx.ConnectError) for r in results)
        resumed = ModelBudget(LLMOptions(max_total_tokens=1000), ledger.report())
        with pytest.raises(BudgetExceeded):
            await resumed.reserve_when_available(10, 700)

    asyncio.run(run())


def test_cancelled_http_keeps_reservation_without_deadlocking_budget():
    async def run():
        ledger = ModelBudget(LLMOptions(max_total_tokens=1000))
        entered = asyncio.Event()

        async def transport(request):
            entered.set()
            await asyncio.Future()

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport),
            event_hooks={"request": [reserve_http_request]},
        ) as client:
            async with _ClientLease(client) as lease:
                with budget_scope(ledger):
                    task = asyncio.create_task(
                        lease.post(
                            "https://test.example/chat/completions",
                            json={"messages": [], "max_tokens": 700},
                        )
                    )
                    await entered.wait()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
        assert not ledger.inflight and ledger.report()["unknown_usage_requests"] == 1

    asyncio.run(run())


def test_invalid_bdi_patch_does_not_promote_assertion_or_desire_and_marks_degradation():
    a = agent("a", "Research team", "UNPROCESSED_PRIVATE")
    old = a.beliefs.model_dump()

    class Client:
        async def json_call(self, **kw):
            assert "sim_1_consensus" not in kw["prompt"]
            return kw["schema"].model_validate(
                {
                    "agent_id": "a",
                    "round_id": 1,
                    "snapshot_version": 0,
                    "processed_observation_ids": ["obs_a"],
                    "assessment": "Updated",
                    "bdi_patch": {
                        "beliefs": {
                            "situation_views": [
                                {
                                    "topic": "new",
                                    "assessment": "UNSUPPORTED",
                                    "confidence": 1,
                                    "timeline_refs": ["invented"],
                                }
                            ]
                        },
                        "desires": [
                            {
                                "desire_id": "new",
                                "description": "UNSUPPORTED_GOAL",
                                "priority": 1,
                            }
                        ],
                    },
                }
            )

    updated = asyncio.run(
        process_agent_information(
            WorldSnapshot(event_title="Case", agents=[a]), Client(), "test"
        )
    )
    assert (
        updated.agents[0].beliefs.model_dump() == old and not updated.agents[0].desires
    )
    assert not updated.agents[0].private_observations[0].processed
    assert updated.validation_issues[0]["code"] == "bdi_patch_rejected"
    assert "UNSUPPORTED" not in json.dumps(updated.model_dump())
    definition = OutcomeDefinition(id="yes", label="Yes", criterion="Criterion")
    plan = OutcomePlan(
        categories=[definition, definition.model_copy(update={"id": "no"})]
    )
    result = aggregate(
        "run",
        {
            "request": {"sample_count": 1, "steps": 1, "step_unit": "day"},
            "knowledge_version": None,
            "baseline_sha256": "hash",
        },
        [
            {
                "sample_id": 1,
                "status": "completed",
                "rounds": [],
                "snapshot": updated.model_dump(),
                "classification": {"category_id": "yes"},
            }
        ],
        plan,
        0,
    )
    assert result["reliability"]["degraded_samples"] == 1
    assert result["reliability"]["bdi_updates_rejected"] == 1


def test_context_deduplicates_evidence_without_removing_private_facts_or_constraints():
    a = agent("a", "Research team")
    a.constraints = ["Cannot spend more than the approved resource cap"]
    a.private_observations = [
        PrivateObservation(
            observation_id="p", description="OWN_PRIVATE_FACT", received_round=0
        )
    ]
    a.evidence_context = [
        {
            "timeline_id": f"h{i}",
            "description": "Observed public fact",
            "evidence": [{"quote": "x" * 2000}],
        }
        for i in range(8)
    ]
    history = [
        TimelineEvent(
            timeline_id=f"h{i}",
            timestamp="2021-03",
            description="Observed public fact",
            evidence_text="SOURCE_QUOTE",
            confidence=1,
            importance=1,
        )
        for i in range(8)
    ]
    s = WorldSnapshot(event_title="Case", agents=[a], timeline=history)
    private = compact_json(actor_context(a, observations=True))
    public = compact_json(public_context(s, "decision", a))
    assert "OWN_PRIVATE_FACT" in private and "OWN_PRIVATE_FACT" not in public
    assert a.constraints[0] in private and "SOURCE_QUOTE" in public
    assert all(f"h{i}" in private for i in range(8))
    assert len(private) + len(public) < 12000


def test_preflight_includes_private_information_before_any_model_call(tmp_path):
    a = agent("a", "Research team")
    request = EnsembleRequest(
        event_title="Case",
        agents=[a.model_dump()],
        context_max_chars=8000,
        initial_private_information={"a": ["PRIVATE_INPUT" * 2000]},
    )
    store = EnsembleStore(tmp_path / "runs")
    with pytest.raises(ValueError, match="No model calls were made"):
        prepare_experiment(
            request, events=EventRepository(tmp_path / "events"), store=store
        )
    assert not list(store.root.glob("*/config.json"))


def test_background_never_rewrites_ambiguous_subsidiary_relation():
    source = "The Canal Authority hired Royal Engineering through its subsidiary Rescue Team to manage salvage operations."
    bad = "The Canal Authority owns Rescue Team as its subsidiary and hired Royal Engineering."
    text = render_brief(
        EvidenceBrief(excerpts=[BriefExcerpt(source_index=1, quote=bad)]),
        [{"title": "Source", "summary": source}],
    )
    assert bad not in text and source in text
    assert "ambiguous pronouns" in text and "Independent corroboration" in text


def test_filtered_asset_metadata_survives_old_actor_cache_projection(tmp_path):
    from tests.test_okf import corpus
    from app.knowledge.okf import KnowledgeStore

    events, case_id = corpus(tmp_path)
    history = events.load_artifact(case_id, "timeline", [])[:1]
    history[0]["retrieval_actors"] = [
        {"name": "Named Vessel", "entity_type": "person", "title": "container ship"}
    ]
    events.save_artifact(case_id, "timeline", history)
    events.save_artifact(
        case_id,
        "participating_objects",
        [
            {
                "canonical_name": "Named Vessel",
                "entity_type": "asset",
                "title": "container ship",
            }
        ],
    )
    store = KnowledgeStore(events)
    store.publish(case_id)
    hit = next(
        h for h in store.search(case_id, kind="actor") if h["title"] == "Named Vessel"
    )
    assert hit["metadata"]["profile"]["agent_type"] == "asset"
    assert not hit["metadata"]["simulation_agent"]


def test_preflight_rejects_oversized_user_intervention_before_dispatch(tmp_path):
    from app.domain import SelectedAction

    a = agent("a", "Research team")
    request = EnsembleRequest(
        event_title="Case",
        agents=[a.model_dump()],
        context_max_chars=8000,
        selected_actions=[
            SelectedAction(agent_id="a", action="long intervention " * 1000)
        ],
    )
    with pytest.raises(ValueError, match="No model calls were made"):
        prepare_experiment(
            request,
            events=EventRepository(tmp_path / "events"),
            store=EnsembleStore(tmp_path / "runs"),
        )


def test_failed_parallel_stage_joins_successful_siblings_before_terminal_state():
    from app.llm.concurrency import gather_complete

    async def run():
        saved = []

        async def success():
            await asyncio.sleep(0.02)
            saved.append("cache-and-billing-settled")

        async def failure():
            raise BudgetExceeded("hard cap")

        with pytest.raises(BudgetExceeded):
            await gather_complete(failure(), success())
        assert saved == ["cache-and-billing-settled"]
        assert not [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]

    asyncio.run(run())
