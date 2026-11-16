"""Offline checks for joint-settlement validation and recovery."""
import asyncio
import json

import pytest

from app.domain import ForecastRound, RelationshipProjection
from app.domain.forecast import ActionInteraction, ActionResult, PendingDecision, QuestionUpdate
from app.domain.forecast import ContractOutlook, WorldStateChange
from app.functional_agents.simulation import GameMasterAgent
from app.simulation.activation import _seed_candidates
from app.simulation.ensemble import validate_step
from app.simulation.joint_settlement import apply_question_updates
from app.simulation.settlement_repair import normalize_joint_settlement
from app.simulation.transitions import apply_round_update
from app.simulation.transitions import project_world_state
from app.simulation.outlook import PublicEvidence, validate_outlook, canonical_table_quote
from app.storage.compact_trace import CompactTrace
from test_joint_settlement import action, settlement, snapshot


def malformed_quote_result():
    state, result = snapshot(), settlement()
    state.agents[0].name = "Liberals"
    text = "The Liberals deployed a returning former leader in public campaign events."
    result.public_events[0].description = text
    result.public_events[0].executed_decisions[0].quote = "Liberals: Deploy a returning former leader in public campaign events."
    result.question_updates[0].before = result.question_updates[0].after = "unknown"
    result.question_updates[0].event_refs = []
    result.question_updates[0].evidence_quote = ""
    result.question_updates[0].pending_actor_ids = ["B"]
    return state, result


@pytest.mark.parametrize("quote,expected", [
    ("The exact  private terms arrived.", "executed"),
    ("The recipient accepted the private terms.", "unknown"),
])
def test_private_action_quote_repair_does_not_invent_a_consequence(quote, expected):
    state, result = snapshot(), settlement()
    result.public_events, result.question_updates = [], []
    state.world_state["question_conditions"] = []
    received = "The exact private terms arrived."
    result.agent_knowledge_updates[1].information = [received]
    result.action_results = [ActionResult(action_id="send", status="executed", summary="Terms sent",
                                          private_observation_quote=quote)]
    normalize_joint_settlement(state, result, [action(visibility="direct", target="B")])
    row, = result.action_results
    assert row.status == expected
    assert row.private_observation_quote == (received if expected == "executed" else "")
    assert result.agent_knowledge_updates[1].information == [received]
    assert result.public_events == []


def test_real_failure_shape_is_repaired_without_changing_facts_or_draws():
    state, result = malformed_quote_result()
    chosen = [action()]
    frozen, draws = state.model_dump(), [row.model_dump() for row in chosen]
    text = result.public_events[0].description
    normalize_joint_settlement(state, result, chosen)
    assert result.public_events[0].description == text
    assert result.public_events[0].executed_decisions[0].quote == text.removeprefix("The ")
    assert result.question_updates[0].pending_actor_ids == []
    assert [row.model_dump() for row in chosen] == draws and state.model_dump() == frozen
    assert {row["code"] for row in result._process_diagnostics} == {
        "execution_quote_span_repaired", "pending_annotation_rejected"}


def test_format_only_repair_uses_no_additional_model_call():
    state, result = malformed_quote_result()

    class Model:
        calls = 0

        async def json_call(self, **kwargs):
            self.calls += 1
            return result.model_copy(deep=True)

    model = Model()
    master = GameMasterAgent(model)
    master.sequential_outcome = master.weighted_decisions = True
    accepted = asyncio.run(master.settle(state, [action()], [], "format_fixture"))
    assert model.calls == 1
    assert len(accepted.public_events) == 1 and accepted.question_updates[0].pending_actor_ids == []


def test_legacy_semantic_repair_serializes_dated_observation_and_accepts_old_null_paths():
    from app.domain import GameMasterSettlement, WorldSnapshot

    calls = []

    class Model:
        async def json_call(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return GameMasterSettlement.model_validate({
                    "state_summary": "Pending", "analysis": "", "next_world_state": {},
                    "public_events": [{"timeline_id": "wrong_id", "timestamp": "2026-11-09",
                                       "description": "The permit is active.", "importance": 1, "confidence": 1}],
                    "state_observations": [{"dimension_id": "permit", "category_id": "active",
                                            "quote": "The permit is active.", "expires_on": "2026-11-10"}],
                })
            return GameMasterSettlement(state_summary="Pending", analysis="", next_world_state={})

    baseline = WorldSnapshot(event_title="Permit", world_state={"measurement_paths": None})
    result = asyncio.run(GameMasterAgent(Model()).settle(baseline, [], [], "legacy_fixture"))
    assert result.public_events == []
    assert len(calls) == 2
    assert '"expires_on":"2026-11-10"' in calls[-1]["prompt"]


def test_registered_actor_can_act_through_its_delegation():
    from app.domain.identity import leading_actor_span

    state = snapshot()
    state.agents[0].name = "Qatar"
    assert leading_actor_span("Qatar’s delegation conveyed the offer.", state.agents, "A") == 0
    assert leading_actor_span("Qatar's rival rejected the offer.", state.agents, "A") is None
    state.agents[1].name = "Qatar's delegation"
    assert leading_actor_span("Qatar's delegation conveyed the offer.", state.agents, "A") is None
    assert leading_actor_span("Qatar's delegation conveyed the offer.", state.agents, "B") == 0


def test_legacy_format_repair_drops_only_malformed_optional_issue_annotations():
    from app.domain import GameMasterSettlement, WorldSnapshot
    from app.llm.client import LLMError
    from pydantic import ValidationError

    valid_issue = {
        "issue_id": "offer", "title": "Offer", "after": "Offer delivered", "kind": "proposal",
        "state_code": "proposed", "event_refs": ["sim_1_offer"], "evidence_quote": "Offer delivered.",
    }
    valid_action = {"action_id": "send", "status": "unknown", "summary": "Outcome unavailable."}
    raw = {"state_summary": "Open", "analysis": "", "next_world_state": {"offer": "sent"},
           "issue_updates": [valid_issue, valid_issue | {"issue_id": "bad", "state_code": "pending"}],
           "action_results": [valid_action, valid_action | {"action_id": "bad", "summary": "x" * 301}]}

    class Model:
        async def json_call(self, **kwargs):
            try:
                result = kwargs["schema"].model_validate(raw)
            except ValidationError as exc:
                raise LLMError("invalid optional annotation") from exc
            assert kwargs["call_name"].endswith("_format_repair")
            assert [row.issue_id for row in result.issue_updates] == ["offer"]
            assert [row.action_id for row in result.action_results] == ["send"]
            return result

    state = WorldSnapshot(event_title="Permit")
    result = asyncio.run(GameMasterAgent(Model()).settle(state, [], [], "legacy_fixture"))
    assert result.next_world_state == {"offer": "sent"}


@pytest.mark.parametrize("kind", ["response", "decision_window"])
def test_supported_pending_decision_survives_round_storage_and_activates_recipient(kind):
    state, result = snapshot(), settlement()
    result.question_updates[0].pending_decisions = [PendingDecision(actor_id="B", kind=kind,
        event_ref="sim_1_offer", evidence_quote=result.public_events[0].description)]
    normalize_joint_settlement(state, result, [action()])
    after = apply_round_update(state, [action()], result, RelationshipProjection())
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1,
        state_summary=result.state_summary, timeline=result.public_events,
        question_updates=result.question_updates, action_results=result.action_results)
    validate_step(state, after, frame, 1)
    saved = after.world_state["question_states"]["approval"]
    assert saved["pending_actor_ids"] == ["B"]
    assert saved["pending_decisions"][0]["event_ref"] == "sim_0001_001_001"
    for actor in after.agents:
        actor.activation.last_deliberation_round = 1
    assert _seed_candidates(after, [])["B"]["mandatory"]


def test_optional_followup_repairs_a_literal_clause_without_forcing_a_response():
    state, result = snapshot(), settlement()
    update = result.question_updates[0]
    update.pending_actor_ids = []
    update.pending_decisions = [PendingDecision(actor_id="B", kind="opportunity",
        event_ref="sim_1_offer", evidence_quote="an offer", choice="Review the terms", review_by="2026-12-31")]
    normalize_joint_settlement(state, result, [action()])
    assert update.pending_actor_ids == []  # Normalization replaces the proposed row.
    revised = result.question_updates[0]
    assert revised.pending_actor_ids == ["B"]
    assert not revised.pending_decisions[0].review_by
    assert revised.pending_decisions[0].evidence_quote == "A sent B an offer; B's approval is pending."
    after = apply_round_update(state, [action()], result, RelationshipProjection())
    for agent in after.agents:
        agent.activation.last_deliberation_round = 1
    candidate = _seed_candidates(after, [])["B"]
    assert "public_decision_opportunity" in candidate["sources"] and not candidate["mandatory"]
    after.agents[1].activation.last_deliberation_round = 2
    assert "public_decision_opportunity" not in _seed_candidates(after, []).get("B", {}).get("sources", [])


@pytest.mark.parametrize("typed", [False, True])
def test_named_counterpart_can_receive_an_opportunity_without_becoming_an_executor(typed):
    state, result = snapshot(), settlement()
    recipient = state.agents[1]
    recipient.name, recipient.title, recipient.agent_type = "Morgan Hale", "Director of Procurement", "person"
    event = result.public_events[0]
    event.description = "A issued a public statement offering to supply Director Hale if the buyer requests a quote."
    event.actors = ["A"]
    event.executed_decisions[0].quote = event.description
    update = result.question_updates[0]
    update.evidence_quote = event.description
    if typed:
        update.pending_decisions = [PendingDecision(actor_id="B", kind="response",
            event_ref=event.timeline_id, evidence_quote=event.description, choice="Decide whether to request a quote")]
    normalize_joint_settlement(state, result, [action()])
    pending, = result.question_updates[0].pending_decisions
    assert pending.actor_id == "B" and pending.kind == "opportunity"
    assert event.actors == ["A"]


def test_invalid_optional_state_update_preserves_prior_condition_and_pending_actors():
    state, result = snapshot(), settlement()
    prior = {"status": "pending", "pending_actor_ids": ["B"], "updated_round": 0}
    state.world_state["question_states"] = {"approval": prior}
    result.question_updates[0].before, result.question_updates[0].after = "pending", "met"
    result.question_updates[0].event_refs = []
    result.question_updates[0].evidence_quote = ""
    result.question_updates[0].pending_actor_ids = []
    normalize_joint_settlement(state, result, [action()])
    assert apply_question_updates(state, result)["approval"] == prior
    assert result.public_events[0].description == settlement().public_events[0].description
    assert any(row["code"] == "question_annotation_rejected" for row in result._process_diagnostics)


def test_invalid_interaction_annotation_cannot_promote_an_unchosen_response():
    state, result = snapshot(), settlement()
    from app.domain.forecast import ActionResult
    result.action_results.append(ActionResult(action_id="wait", status="waiting", summary="Wait"))
    result.interactions = [ActionInteraction(action_ids=["send", "wait"], kind="depends_on",
        precondition="A new request arrives", available_at="new_this_round", outcome="realized", effect="B accepted",
        event_refs=["sim_1_offer"], evidence_quote=result.public_events[0].description)]
    normalize_joint_settlement(state, result, [action(), action("wait", "B", decision="wait")])
    assert result.interactions == []
    assert result.public_events[0].executed_decisions[0].actor_id == "A"


@pytest.mark.parametrize("violation", ["wrong_owner", "waiting_actor", "object_mention", "ambiguous_alias"])
def test_annotation_repairs_do_not_bypass_core_authorization(violation):
    state, result = snapshot(), settlement()
    chosen = [action()]
    execution = result.public_events[0].executed_decisions[0]
    if violation == "wrong_owner":
        execution.actor_id = "B"
    elif violation == "waiting_actor":
        chosen[0].decision = "wait"
        result.action_results[0].status = "waiting"
        result.public_events[0].cause_action_ids = []
    elif violation == "object_mention":
        result.public_events[0].description = "B sent A a different proposal."
        execution.quote = "A: Send an offer"
    else:
        for actor in state.agents:
            actor.aliases = ["Joint Office"]
        result.public_events[0].description = "The Joint Office delivered a proposal."
        execution.quote = "Joint Office: Deliver a proposal"
    with pytest.raises(ValueError):
        normalize_joint_settlement(state, result, chosen)


def test_invalid_question_annotations_do_not_discard_independent_valid_progress():
    state, result = snapshot(), settlement()
    state.world_state["question_conditions"].append({"id": "delivery", "description": "Offer delivery"})
    valid = QuestionUpdate(condition_id="delivery", before="unknown", after="met", summary="Offer delivered",
        event_refs=["sim_1_offer"], evidence_quote=result.public_events[0].description)
    result.question_updates = [result.question_updates[0], result.question_updates[0].model_copy(), valid]
    normalize_joint_settlement(state, result, [action()])
    states = apply_question_updates(state, result)
    assert "approval" not in states and states["delivery"]["status"] == "met"


def test_failed_binding_artifact_redacts_key_and_excludes_requests(tmp_path):
    recorder = CompactTrace(tmp_path, "TEST_SECRET")
    recorder.write("fixture", "settlement_semantic_error", {
        "message": "TEST_SECRET quote error", "known_agent_ids": ["A"], "event_ids": ["e1"],
        "execution_records": [{"description": "A said TEST_SECRET", "executed_decisions": []}],
        "headers": {"Authorization": "DO_NOT_PERSIST"}, "private_inbox": "DO_NOT_PERSIST",
    })
    row = json.loads((tmp_path / "model_calls.jsonl").read_text())
    artifact = (tmp_path / row["diagnostic_artifact"]).read_text()
    assert "TEST_SECRET" not in artifact and "DO_NOT_PERSIST" not in artifact
    assert "[redacted]" in artifact and "execution_records" in artifact


@pytest.mark.parametrize("quote,reference,valid", [
    ('{| class="wikitable"\n! Pollster !! Date !! Party A', 'table', False),
    ('{| class="wikitable"\n! Pollster !! Date !! Party A', '', False),
    ('| Agency One || 2026-01-01 || 23.5%', 'table', True),
])
def test_readout_requires_table_data_instead_of_formatting_headers(quote, reference, valid):
    state = snapshot()
    state.world_state.pop("contract_outlook")
    raw = '{| class="wikitable"\n! Pollster !! Date !! Party A\n|-\n| Agency One || 2026-01-01 || 23.5%\n|}'
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1, state_summary="A poll exists",
        contract_outlook=ContractOutlook(p_yes=.3, basis="The poll is historical context.", evidence_ref=reference,
                                         evidence_quote=quote))
    estimate, issue = validate_outlook(state, state, frame, event_ids={},
        catalog={"table": PublicEvidence("table", "frozen_excerpt", raw)}, terminal=True)
    assert (estimate is not None) == valid
    assert issue == (None if valid else "non_substantive_quote")


def test_prose_in_a_mixed_excerpt_is_not_mistaken_for_table_formatting():
    state = snapshot()
    state.world_state.pop("contract_outlook")
    prose = 'The company plans to start trading after it files its registration.'
    raw = '{| class="wikitable"\n! Company !! Date\n|-\n| Agency One || 2026-01-01\n|}\n' + prose
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1, state_summary="No filing observed",
        contract_outlook=ContractOutlook(p_yes=.3, basis="A filing remains pending.", evidence_ref="mixed", evidence_quote=prose))
    estimate, issue = validate_outlook(state, state, frame, event_ids={},
        catalog={"mixed": PublicEvidence("mixed", "frozen_excerpt", raw)}, terminal=True)
    assert issue is None and estimate.evidence_quote == prose


def test_result_can_recover_missing_reference_only_from_matching_causal_execution():
    state, result = snapshot(), settlement()
    result.action_results[0].event_refs = []
    normalize_joint_settlement(state, result, [action()])
    assert result.action_results[0].event_refs == ["sim_1_offer"]
    assert any(row["code"] == "action_result_refs_repaired" for row in result._process_diagnostics)
    result = settlement()
    result.action_results[0].event_refs = []
    result.public_events[0].executed_decisions = []
    with pytest.raises(ValueError, match="public consequence"):
        normalize_joint_settlement(state, result, [action()])


@pytest.mark.parametrize("parent", ["missing", "existing_object", "null", "scalar", "protected"])
def test_nested_add_creates_only_absent_containers_and_preserves_frozen_state(parent):
    state, result = snapshot(), settlement()
    if parent == "existing_object":
        state.world_state["campaign"] = {"other": {"status": "scheduled"}}
    elif parent in {"null", "scalar"}:
        state.world_state["campaign"] = None if parent == "null" else 3
    change = WorldStateChange(path=["campaign", "events", "offer"], operation="add", before=None,
        after={"status": "delivered"}, reason="The offer was publicly delivered.",
        event_ref="sim_1_offer", evidence_quote=result.public_events[0].description)
    if parent == "protected":
        change.path = ["question_states", "approval", "invented"]
    result.state_changes = [change]
    before = state.model_dump()
    if parent in {"null", "scalar", "protected"}:
        with pytest.raises(ValueError):
            project_world_state(state, result, [action()])
    else:
        after = project_world_state(state, result, [action()])
        assert after["campaign"]["events"]["offer"] == {"status": "delivered"}
        if parent == "existing_object":
            assert after["campaign"]["other"] == {"status": "scheduled"}
    assert state.model_dump() == before


@pytest.mark.parametrize("quote,valid", [
    ("Demoskop | 13–24 Aug | 2,117 | 6.8 | 30.4", True),
    ("Demoskop\\n|13–24 Aug\\n|2,117\\n|6.8\\n|30.4", True),
    ("Demoskop | 13–24 Aug | 2,117 | 9.8 | 30.4", False),
    ("Demoskop 13–24 Aug: V 6.8; S 30.4", False),
    ("Pollster | Date | V | S", False),
    ("Demoskop", False),
])
def test_inline_model_table_quote_resolves_only_identical_unambiguous_source_cells(quote, valid):
    text = '{| class="wikitable"\n!Pollster\n!Date\n!V\n!S\n|-\n|Demoskop\n|13–24 Aug\n|2,117\n|6.8\n|30.4\n|-\n|Demoskop\n|10–12 July\n|2,003\n|6.5\n|31.0\n|}'
    resolved = canonical_table_quote(text, quote)
    assert (resolved is not None) == valid
    if valid:
        assert resolved in text and "6.8\n|30.4" in resolved
