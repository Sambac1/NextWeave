"""Regression cases for shared context, causal settlement, and public replay."""
import pytest

from app.domain import ActionProposal, WorldSnapshot, ForecastRound
from app.domain.forecast import SequentialGameMasterSettlement
from app.simulation.activation import _seed_candidates
from app.simulation.context import public_context
from app.simulation.joint_settlement import VERSION, validate_joint_settlement
from app.simulation.outlook import evidence_catalog
from app.simulation.transitions import apply_round_update
from app.domain import RelationshipProjection
from app.simulation.ensemble import validate_step
from app.benchmark.runner import public_replay
from test_agent_protocol import agent as protocol_agent


def actor(identifier):
    return protocol_agent(identifier, identifier)


def snapshot():
    return WorldSnapshot(event_title="Supplier approval", agents=[actor("A"), actor("B")], world_state={
        "simulation_clock": {"interaction_protocol": VERSION, "sequential_harness": "fixture", "interval_start": "2026-01-01", "interval_end": "2026-01-02"},
        "question_conditions": [{"id": "approval", "description": "Has the supplier approved the offer?"}],
        "question_states": {}, "contract_outlook": {"p_yes": .9}, "contract_progress": "INTERNAL FORECAST",
        "production": {"capacity": 5},
    })


def action(identifier="send", owner="A", **kwargs):
    return ActionProposal(action_id=identifier, agent_id=owner, agent_name=owner, round_id=1, snapshot_version=0,
                          decision=kwargs.pop("decision", "act"), action_type="communicate", action="Send an offer to B",
                          reasoning="Seek approval", confidence=.7, **kwargs)


def settlement():
    text = "A sent B an offer; B's approval is pending."
    return SequentialGameMasterSettlement(state_summary=text, analysis="An offer was sent", state_changes=[],
        public_events=[{"timeline_id": "sim_1_offer", "timestamp": "2026-01-02", "description": text,
                        "actors": ["A", "B"], "cause_action_ids": ["send"],
                        "executed_decisions": [{"actor_id": "A", "action_id": "send", "quote": text}],
                        "confidence": 1, "importance": 1}],
        action_results=[{"action_id": "send", "status": "executed", "summary": "Offer sent", "event_refs": ["sim_1_offer"]}],
        question_updates=[{"condition_id": "approval", "before": "unknown", "after": "pending", "summary": "INTERNAL SUMMARY",
                           "event_refs": ["sim_1_offer"], "evidence_quote": text, "pending_actor_ids": ["B"]}],
        agent_knowledge_updates=[{"agent_id": identifier, "information": []} for identifier in ("A", "B")])


def test_public_brief_has_environment_conditions_and_no_forecast():
    state = snapshot()
    public = public_context(state, "decision", state.agents[0])
    assert public["world_state"]["production"]["capacity"] == 5
    assert "contract_outlook" not in public["world_state"] and "contract_progress" not in public["world_state"]
    assert public["round_briefing"]["question_conditions"][0]["status"] == "unknown"


def test_offer_remains_pending_activates_recipient_and_reaches_probability_readout():
    state, result, chosen = snapshot(), settlement(), [action()]
    validate_joint_settlement(state, result, chosen)
    after = apply_round_update(state, chosen, result, RelationshipProjection())
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1, state_summary=result.state_summary,
                          agent_actions=chosen, action_results=result.action_results, question_updates=result.question_updates,
                          timeline=result.public_events)
    validate_step(state, after, frame, 1)
    record = after.world_state["question_states"]["approval"]
    assert record["status"] == "pending" and record["event_refs"] == ["sim_0001_001_001"]
    assert record["summary"] != "INTERNAL SUMMARY"
    # Make ordinary periodic/initial activation inapplicable to isolate the trigger.
    for item in after.agents:
        item.activation.last_deliberation_round = 1
    assert _seed_candidates(after, [])["B"]["mandatory"]
    assert evidence_catalog(state, frame)["condition:approval"].text.startswith("approval: pending")
    replay, = public_replay({"snapshot": after.model_dump(), "rounds": [frame.model_dump()]})
    assert replay["action_results"][0]["actor_name"] == "A"
    assert replay["question_updates"][0]["label"] == "Has the supplier approved the offer?"
    assert "INTERNAL SUMMARY" not in str(replay)


@pytest.mark.parametrize("mutation", ["missing_action", "fake_action", "missing_condition", "wrong_before", "invented_event", "clear_pending_without_evidence"])
def test_invalid_settlement_does_not_modify_original_state(mutation):
    state, result = snapshot(), settlement()
    saved = state.model_dump()
    if mutation == "missing_action":
        result.action_results = []
    elif mutation == "fake_action":
        result.action_results[0].action_id = "unselected"
    elif mutation == "missing_condition":
        result.question_updates = []
    elif mutation == "wrong_before":
        result.question_updates[0].before = "met"
    elif mutation == "invented_event":
        result.action_results[0].event_refs = ["missing"]
    else:
        state.world_state["question_states"] = {"approval": {"status": "pending", "pending_actor_ids": ["B"]}}
        saved = state.model_dump()
        result.question_updates[0].before = "pending"
        result.question_updates[0].event_refs = []
        result.question_updates[0].pending_actor_ids = []
    with pytest.raises(ValueError):
        validate_joint_settlement(state, result, [action()])
    assert state.model_dump() == saved


def test_new_same_round_request_cannot_create_an_unchosen_response():
    result = settlement()
    from app.domain.forecast import ActionResult, ActionInteraction
    result.action_results.append(ActionResult(action_id="wait", status="waiting", summary="B waits"))
    result.interactions = [ActionInteraction(action_ids=["send", "wait"], kind="depends_on", precondition="B reads A's new offer",
        available_at="new_this_round", outcome="realized", effect="B accepts", event_refs=["sim_1_offer"], evidence_quote=result.public_events[0].description)]
    with pytest.raises(ValueError, match="newly learned"):
        validate_joint_settlement(snapshot(), result, [action(), action("wait", "B", decision="wait")])
    result.interactions[0].outcome = "pending"
    validate_joint_settlement(snapshot(), result, [action(), action("wait", "B", decision="wait")])


def test_private_resolution_stays_private_and_legacy_records_still_load():
    from app.domain.forecast import ActionResult
    state, result = snapshot(), settlement()
    result.public_events, result.question_updates = [], []
    state.world_state["question_conditions"] = []
    result.agent_knowledge_updates[1].information = ["PRIVATE offered terms arrived"]
    result.action_results = [ActionResult(action_id="send", status="executed", summary="PRIVATE offer delivered",
                                          private_observation_quote="PRIVATE offered terms arrived")]
    chosen = [action(visibility="direct", target="B")]
    validate_joint_settlement(state, result, chosen)
    replay, = public_replay({"rounds": [{"step": 1, "agent_actions": [chosen[0].model_dump()],
                                       "action_results": [row.model_dump() for row in result.action_results]}]})
    assert replay["action_results"] == [] and "PRIVATE" not in str(replay)
    legacy, = public_replay({"rounds": [{"step": 1, "agent_actions": [{"agent_id": "A"}]}]})
    assert legacy["action_results"] == []


def test_unverified_quotes_cannot_enter_public_replay():
    from app.domain.forecast import ActionInteraction
    state, result = snapshot(), settlement()
    result.question_updates[0].after = "unknown"
    result.question_updates[0].event_refs = []
    result.question_updates[0].evidence_quote = "SECRET unsupported statement"
    with pytest.raises(ValueError, match="public quote"):
        validate_joint_settlement(state, result, [action()])
    forged = result.model_dump()
    forged["step"], forged["agent_actions"] = 1, [action().model_dump()]
    rows, = public_replay({"rounds": [forged]})
    assert "SECRET" not in str(rows)
    result = settlement()
    result.interactions = [ActionInteraction(action_ids=["send", "other"], kind="depends_on",
        precondition="Waiting for response", available_at="round_start", outcome="pending", effect="Still pending",
        event_refs=["sim_1_offer"], evidence_quote="SECRET arbitrary explanation")]
    from app.domain.forecast import ActionResult
    result.action_results.append(ActionResult(action_id="other", status="waiting", summary="Monitoring"))
    with pytest.raises(ValueError, match="public quote"):
        validate_joint_settlement(state, result, [action(), action("other", "B", decision="monitor")])
    forged = result.model_dump() | {"step": 1, "agent_actions": [action().model_dump(), action("other", "B").model_dump()]}
    assert "SECRET" not in str(public_replay({"rounds": [forged]}))


def test_event_executor_cannot_claim_a_waiting_actor_accepted():
    from app.domain.world import ExecutedDecision
    from app.domain.forecast import ActionResult
    result = settlement()
    result.public_events[0].description = "B formally accepted A's new offer immediately."
    result.public_events[0].executed_decisions = [ExecutedDecision(actor_id="B", action_id="wait",
                                                                  quote=result.public_events[0].description)]
    result.question_updates = []
    result.action_results.append(ActionResult(action_id="wait", status="waiting", summary="Wait"))
    with pytest.raises(ValueError, match="executor"):
        validate_joint_settlement(snapshot(), result, [action(), action("wait", "B", decision="wait")])
    # Merely assigning the effect to A's trigger also fails the subject check.
    result.public_events[0].executed_decisions[0].actor_id = "A"
    result.public_events[0].executed_decisions[0].action_id = "send"
    with pytest.raises(ValueError, match="registered subject"):
        validate_joint_settlement(snapshot(), result, [action(), action("wait", "B", decision="wait")])


def test_prior_offer_can_be_accepted_by_a_currently_selected_response():
    from app.domain.world import ExecutedDecision
    state, result = snapshot(), settlement()
    state.agents[0].historical_actions = [action().model_dump()]
    state.world_state["question_states"] = {"approval": {"status": "pending", "pending_actor_ids": ["B"]}}
    chosen = action("accept", "B", target="A")
    text = "B accepted A's previously delivered offer."
    result.public_events[0].description = text
    result.public_events[0].cause_action_ids = ["send", "accept"]
    result.public_events[0].executed_decisions = [ExecutedDecision(actor_id="B", action_id="accept", quote=text)]
    result.action_results[0].action_id = "accept"
    result.question_updates[0].before, result.question_updates[0].after = "pending", "met"
    result.question_updates[0].evidence_quote = text
    result.question_updates[0].pending_actor_ids = []
    validate_joint_settlement(state, result, [chosen])
    updated = apply_round_update(state, [chosen], result, RelationshipProjection())
    assert updated.world_state["question_states"]["approval"]["status"] == "met"
    assert updated.world_state["question_states"]["approval"]["pending_actor_ids"] == []


def test_public_cross_round_dependency_and_intervention_are_visible():
    first = {"step": 1, "agent_actions": [action().model_dump()]}
    text = "B accepted the offer after a user intervention."
    second = {"step": 2, "agent_actions": [action("accept", "B").model_dump()],
              "intervention_actions": [action("intervention_2_01", "A").model_dump()],
              "timeline": [{"timeline_id": "e2", "timestamp": "2026-01-03", "description": text}],
              "action_results": [{"action_id": "intervention_2_01", "status": "executed", "event_refs": ["e2"]}],
              "interactions": [{"kind": "depends_on", "action_ids": ["send", "accept"], "outcome": "realized",
                                "event_refs": ["e2"], "evidence_quote": text}]}
    _, visible = public_replay({"rounds": [first, second]})
    assert len(visible["interactions"]) == 1 and visible["action_results"][0]["actor_name"] == "A"
    first["agent_actions"][0]["visibility"] = "private"
    _, hidden = public_replay({"rounds": [first, second]})
    assert hidden["interactions"] == []


@pytest.mark.parametrize("track_issues", [False, True])
def test_prior_private_action_can_have_a_delayed_public_effect_without_exposing_its_id(track_issues):
    import asyncio
    from app.functional_agents.simulation import GameMasterAgent
    from app.simulation.process import VERSION as PROCESS_VERSION

    state, result = snapshot(), settlement()
    state.agents[0].private_action_history = [action(visibility="private", target="B").model_dump()]
    state.agents[0].private_action_history[0]["action"] = "CONFIDENTIAL dispatch of offer to B"
    assert "CONFIDENTIAL" not in str(public_context(state, "decision", state.agents[1]))
    if track_issues:
        state.world_state["process_tracking_version"] = PROCESS_VERSION
    result.action_results = []  # No fresh decision is required for a delayed effect.

    class Runtime:
        calls = 0

        async def json_call(self, **kwargs):
            self.calls += 1
            assert "RECORDED CAUSAL ACTIONS (Game Master only)" in kwargs["prompt"]
            assert "CONFIDENTIAL dispatch" in kwargs["prompt"]
            return result.model_copy(deep=True)

    runtime = Runtime()
    master = GameMasterAgent(runtime)
    master.sequential_outcome = True
    accepted = asyncio.run(master.settle(state, [], [], "delayed_fixture"))
    assert runtime.calls == 1
    assert len(accepted.public_events) == 1
    assert accepted.public_events[0].cause_action_ids == []
    assert accepted.public_events[0].executed_decisions == []
    assert state.agents[0].private_action_history[0]["action_id"] == "send"


def test_one_actors_execution_cannot_substantiate_another_actors_executed_result():
    from app.domain.forecast import ActionResult
    result = settlement()
    result.action_results.append(ActionResult(action_id="accept", status="executed", summary="B accepted",
                                              event_refs=["sim_1_offer"]))
    with pytest.raises(ValueError, match="own execution"):
        validate_joint_settlement(snapshot(), result, [action(), action("accept", "B")])
