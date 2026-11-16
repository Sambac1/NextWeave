from __future__ import annotations

import asyncio

from app.domain import (
    ActionProposal,
    AgentProfile,
    Beliefs,
    DecisionPolicy,
    GameMasterSettlement,
    PrivateObservation,
    RelationshipProjection,
    SelectedAction,
    TimelineEvent,
    WorldSnapshot,
)
from app.simulation.action_audit import produce_audited_actions
from app.simulation.activation import choose_active_agents
from app.simulation.bdi_updates import process_agent_information
from app.functional_agents.simulation import GameMasterAgent
from app.simulation.registry import maintain_registry
from app.simulation.transitions import apply_round_update


def agent(agent_id: str, name: str, secret: str = "") -> AgentProfile:
    observations = []
    if secret:
        observations.append(PrivateObservation(
            observation_id=f"obs_{agent_id}", description=secret, received_round=1,
        ))
    return AgentProfile(
        agent_id=agent_id,
        name=name,
        aliases=[],
        agent_type="person",
        decision_subject=f"{name} decision center",
        role="neutral",
        description=f"Profile for {name}",
        private_observations=observations,
        beliefs=Beliefs(),
        decision_policy=DecisionPolicy(risk_tolerance=0.4, compromise_tendency=0.5, response_speed=0.6),
    )


def test_registry_uses_one_agent_id_and_merges_alias_duplicates() -> None:
    first = agent("agent_a", "Alice")
    second = agent("agent_duplicate", "A. Alice")
    second.aliases = ["Alice"]
    result = maintain_registry(WorldSnapshot(event_title="Event", agents=[first, second]))
    assert len(result.agents) == 1
    payload = result.agents[0].model_dump()
    assert payload["agent_id"] == "agent_a"
    assert "id" not in payload
    assert "decision_center_id" not in payload


def test_activation_keeps_private_self_reviews_isolated_and_honors_intervention() -> None:
    prompts: dict[str, str] = {}

    class Client:
        async def json_call(self, **kwargs):
            prompts[kwargs["call_name"]] = kwargs["prompt"]
            schema = kwargs["schema"]
            if schema.__name__ == "MasterActivationBatch":
                ids = ("agent_b",) if kwargs["call_name"].endswith("_repair") else ("agent_a", "agent_b")
                return schema.model_validate({
                    "round_id": 1,
                    "snapshot_version": 0,
                    "reviews": [
                        {"agent_id": value, "master_score": 0.1, "decision_point_now": False, "inaction_consistency": "stable", "feasible_action_exists": False, "decision_window": "closed", "cross_agent_dependency": False, "reasons": ["low"]}
                        for value in ids
                    ],
                })
            agent_id = "agent_a" if "agent_agent_a" in kwargs["call_name"] else "agent_b"
            return schema.model_validate({
                "agent_id": agent_id, "round_id": 1, "snapshot_version": 0,
                "self_score": 0.1, "decision_required": False, "feasible_action_exists": False,
                "preferred_mode": "wait", "reasons": ["low"],
            })

    snapshot = WorldSnapshot(event_title="Event", agents=[agent("agent_a", "Alice", "ALICE_ONLY"), agent("agent_b", "Bob", "BOB_ONLY")])
    # Bob has already deliberated; only Alice's explicit intervention is due.
    snapshot.agents[1].activation.last_deliberation_round = 0
    updated, active, decisions = asyncio.run(choose_active_agents(
        snapshot,
        [SelectedAction(agent_id="agent_a", agent_name="Alice", action="Respond")],
        Client(),
        "test",
    ))
    assert [item.agent_id for item in active] == ["agent_a"]
    assert "test_agent_agent_a_self_activation" not in prompts
    bob_prompt = prompts["test_agent_agent_b_self_activation"]
    assert "BOB_ONLY" in bob_prompt and "ALICE_ONLY" not in bob_prompt
    assert 'ALICE_ONLY' not in prompts['test_activation_master_review']
    assert 'STRUCTURE REPAIR' in prompts['test_activation_master_review_repair']
    assert next(item for item in decisions if item.agent_id == "agent_a").mandatory is True
    assert updated.agents[0].activation.status == "active"


def test_only_rejected_agent_is_retried() -> None:
    decision_calls = {"agent_a": 0, "agent_b": 0}

    class Client:
        async def json_call(self, **kwargs):
            schema = kwargs["schema"]
            call_name = kwargs["call_name"]
            if schema.__name__ == "ActionProposal":
                agent_id = "agent_a" if "agent_agent_a" in call_name else "agent_b"
                decision_calls[agent_id] += 1
                bad_target = agent_id == "agent_a" and decision_calls[agent_id] == 1
                return schema.model_validate({
                    "agent_id": agent_id,
                    "agent_name": "Alice" if agent_id == "agent_a" else "Bob",
                    "round_id": 1,
                    "snapshot_version": 0,
                    "action_id": f"action_1_{agent_id}",
                    "decision": "act",
                    "action_type": "statement",
                    "action": "Issue a statement",
                    "target": "unknown" if bad_target else "",
                    "visibility": "public",
                    "reasoning": "Within current authority",
                    "confidence": 0.7,
                })
            proposal_text = kwargs["prompt"].split("PROGRAM-VALID PROPOSALS", 1)[1]
            proposal_ids = [value for value in ("agent_a", "agent_b") if f'"agent_id": "{value}"' in proposal_text]
            return schema.model_validate({
                "round_id": 1, "snapshot_version": 0,
                "reviews": [{"agent_id": value, "accepted": True} for value in proposal_ids],
            })

    active = [agent("agent_a", "Alice"), agent("agent_b", "Bob")]
    proposals = asyncio.run(produce_audited_actions(WorldSnapshot(event_title="Event", agents=active), active, [], Client(), "test"))
    assert len(proposals) == 2
    assert decision_calls == {"agent_a": 2, "agent_b": 1}


def test_private_observation_is_delivered_only_to_recipient() -> None:
    snapshot = WorldSnapshot(event_title="Event", agents=[agent("agent_a", "Alice"), agent("agent_b", "Bob")])
    action = ActionProposal(
        agent_id="agent_a", agent_name="Alice", round_id=1, snapshot_version=0,
        action_id="action_1_agent_a", decision="act", action_type="message", action="Send message",
        target="agent_b", visibility="direct", reasoning="test", confidence=0.7,
    )
    settlement = GameMasterSettlement(
        state_summary="After",
        next_world_state={},
        agent_knowledge_updates=[
            {"agent_id": "agent_a", "information": []},
            {"agent_id": "agent_b", "information": ["Private message contents"]},
        ],
        analysis="settled",
    )
    updated = apply_round_update(
        snapshot,
        [action],
        settlement,
        RelationshipProjection(),
    )
    assert updated.agents[0].private_observations == []
    assert updated.agents[1].private_observations[0].description == "Private message contents"


def test_each_ordinary_agent_updates_its_own_bdi_from_public_and_private_information() -> None:
    prompts: dict[str, str] = {}

    class Client:
        async def json_call(self, **kwargs):
            call_name = kwargs["call_name"]
            prompts[call_name] = kwargs["prompt"]
            agent_id = "agent_a" if "agent_agent_a" in call_name else "agent_b"
            observation_id = f"obs_{agent_id}"
            return kwargs["schema"].model_validate({
                "agent_id": agent_id,
                "round_id": 2,
                "snapshot_version": 1,
                "processed_observation_ids": [observation_id],
                "bdi_patch": {
                    "beliefs": {
                        "situation_views": [{
                            "topic": f"view_{agent_id}",
                            "assessment": "Updated from information",
                            "confidence": 0.8,
                            "timeline_refs": ["sim_1_public"],
                        }],
                        "actor_models": [],
                    },
                },
                "assessment": "Information processed",
            })

    snapshot = WorldSnapshot(
        event_title="Event",
        round_id=1,
        snapshot_version=1,
        agents=[agent("agent_a", "Alice", "ALICE_ONLY"), agent("agent_b", "Bob", "BOB_ONLY")],
        timeline=[TimelineEvent(
            timeline_id="sim_1_public",
            timestamp="2026-01-01",
            description="A public event",
            confidence=0.9,
            importance=0.7,
            origin="simulated",
            round_id=1,
        )],
    )
    updated = asyncio.run(process_agent_information(snapshot, Client(), "test"))

    alice_prompt = prompts["test_agent_agent_a_information_review"]
    bob_prompt = prompts["test_agent_agent_b_information_review"]
    assert "A public event" in alice_prompt and "A public event" in bob_prompt
    assert "ALICE_ONLY" in alice_prompt and "BOB_ONLY" not in alice_prompt
    assert "BOB_ONLY" in bob_prompt and "ALICE_ONLY" not in bob_prompt
    assert updated.agents[0].beliefs.situation_views[0].topic == "view_agent_a"
    assert updated.agents[1].beliefs.situation_views[0].topic == "view_agent_b"
    assert updated.agents[0].private_observations[0].processed is True
    assert updated.agents[1].private_observations[0].processed is True


def test_game_master_receives_public_profiles_but_not_private_bdi() -> None:
    prompts: list[str] = []

    class Client:
        async def json_call(self, **kwargs):
            prompts.append(kwargs["prompt"])
            return kwargs["schema"].model_validate({
                "state_summary": "After",
                "next_world_state": {"status": "settled"},
                "agent_knowledge_updates": [
                    {"agent_id": "agent_a", "information": ["A_ONLY_NEW"]},
                    {"agent_id": "agent_b", "information": []},
                ],
                "analysis": "settled",
            })

    alice = agent("agent_a", "Alice", "ALICE_OLD_SECRET")
    bob = agent("agent_b", "Bob", "BOB_OLD_SECRET")
    result = asyncio.run(GameMasterAgent(Client()).settle(
        WorldSnapshot(event_title="Event", agents=[alice, bob]),
        [],
        [],
        "test",
    ))
    assert "ALICE_OLD_SECRET" not in prompts[0]
    assert "BOB_OLD_SECRET" not in prompts[0]
    assert result.agent_knowledge_updates[0].information == ["A_ONLY_NEW"]
