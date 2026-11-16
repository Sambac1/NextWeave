import asyncio

import pytest

from app.domain.ensemble import EventQuestion, OutcomeDimension
from app.domain.forecast import ForecastRound, RoundStateObservation
from app.domain.protocol import ScenarioProtocol
from app.simulation.ensemble import prepare_experiment, run_experiment
from app.simulation.event_metrics import resolve_answer
from app.simulation.state_observations import (
    resolve_terminal_states, save_round_observations, validate_measurement_plan,
)
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository
from test_ensemble import Classifier, request, runner


def dimension(mode="standing"):
    return OutcomeDimension(
        id="import_rule", label="Import rule", observation_mode=mode,
        observation_scope="Effective import rule for the modeled market",
        categories=[
            {"id": "active", "label": "Active", "criterion": "The rule is in force",
             "required_actor_ids": ["actor_a"]},
            {"id": "ended", "label": "Ended", "criterion": "The rule has been withdrawn"},
        ],
    )


def frame(step, end, **kwargs):
    return ForecastRound(step=step, simulation_interval={"interval_end": end}, state_summary="",
                         input_snapshot_version=step-1, output_snapshot_version=step, **kwargs)


def test_standing_status_is_recorded_without_duplicate_events_and_expires():
    text = "The published import rule remains in effect until January 20."
    first = frame(1, "2027-01-10", continuing_state_notes=[text], state_observations=[
        RoundStateObservation(dimension_id="import_rule", category_id="active",
                              actor_ids=["actor_a"], quote=text, expires_on="2027-01-20"),
    ])
    d = dimension()
    recorded = save_round_observations(first, [d], {"actor_a"}, {})
    assert recorded["import_rule"]["source"] == "settled_public_state" and not first.timeline
    start = first.model_dump() | {"measurement_states": recorded}
    for end, expected in [("2027-01-19", "active"), ("2027-01-20", "other")]:
        path = {"rounds": [start, frame(2, end).model_dump()]}
        assert resolve_terminal_states(path, [d])[0].category_id == expected


def test_boundary_observation_cannot_be_carried_or_fabricated():
    d = dimension("boundary")
    text = "The published import rule remains in effect today."
    first = frame(1, "2027-01-10", continuing_state_notes=[text], state_observations=[
        RoundStateObservation(dimension_id="import_rule", category_id="active",
                              actor_ids=["actor_a"], quote=text, until_superseded=True),
        RoundStateObservation(dimension_id="import_rule", category_id="ended", quote="Unsupported claim."),
    ])
    recorded = save_round_observations(first, [d], {"actor_a"}, {})
    assert recorded["import_rule"]["category_id"] == "active"
    path = {"rounds": [first.model_dump() | {"measurement_states": recorded},
                       frame(2, "2027-01-11").model_dump()]}
    assert resolve_terminal_states(path, [d])[0].category_id == "other"


def test_explicit_mechanism_produces_numeric_measurement():
    d = OutcomeDimension(id="volume", label="Trade volume", observation_mode="mechanism",
                         observation_scope="Configured simulated number of shipments", categories=[
        {"id": "low", "label": "Low", "criterion": "Below 10",
         "conditions": [{"path": "shipments", "operator": "lt", "value": 10}]},
        {"id": "high", "label": "High", "criterion": "At least 10",
         "conditions": [{"path": "shipments", "operator": "ge", "value": 10}]},
    ])
    with pytest.raises(ValueError, match="no configured mechanism"):
        validate_measurement_plan([d], {"shipments": 8}, ScenarioProtocol())
    protocol = ScenarioProtocol(rates=[{"path": "shipments", "delta_per_step": -1}])
    validate_measurement_plan([d], {"shipments": 8}, protocol)
    state = save_round_observations(frame(1, "2027-01-10"), [d], set(), {"shipments": 8})
    assert state["volume"]["source"] == "configured_mechanism"
    assert state["volume"]["category_id"] == "low"


def test_negative_requires_whole_window_coverage():
    question = EventQuestion(id="announcement", label="Announcement", criterion="Public agreement announcement")
    quote = "The team said this draft remains private."
    record = {"rounds": [{"timeline": [{"timeline_id": "draft", "description": quote}]}]}
    value = resolve_answer(question, {
        "question_id": question.id, "answer": "no", "evidence_kind": "explicit_negative",
        "evidence_scope": "whole_horizon", "event_refs": ["draft"], "evidence_quote": quote,
    }, record, strict_occurrence=True)
    assert value.answer == "unclear" and value.trace_answer == "unclear"


@pytest.mark.parametrize("with_observation", [False, True])
def test_preflight_gates_batch_and_classification_uses_measurements(tmp_path, with_observation):
    class NoCalls(Classifier):
        async def json_call(self, **kwargs):
            raise AssertionError("This experiment should need no post-hoc model call")

    options = request()
    options.outcome_definitions, options.event_questions = [], []
    options.outcome_dimensions = [dimension()]
    calls = []
    make_round = runner(calls)

    async def observe_round(*args):
        state, result = await make_round(*args)
        if with_observation and result.step == 1:
            note = "The published import rule remains in effect until withdrawn."
            result.continuing_state_notes = [note]
            result.state_observations = [RoundStateObservation(
                dimension_id="import_rule", category_id="active", actor_ids=["actor_a"],
                quote=note, until_superseded=True,
            )]
        return state, result

    store = EnsembleStore(tmp_path / "runs")
    run_id, options = prepare_experiment(options, events=EventRepository(tmp_path / "events"), store=store)
    result = asyncio.run(run_experiment(run_id, options, store=store, runtime=NoCalls(), round_runner=observe_round))
    assert result["measurement_preflight"]["passed"] == with_observation
    assert len(calls) == (8 if with_observation else 4)
    assert result["outcome_dimensions"][0]["assessable_samples"] == (4 if with_observation else 0)
    assert result["completed_samples"] == (4 if with_observation else 2)
    assert result["model_calls_total"] == 0
    if not with_observation:
        assert store.load(run_id, "status")["stage"] == "observation_preflight"
        assert not store.sample(run_id, 3)


def test_standing_effect_rejects_an_unimplemented_proposal():
    from app.domain.world import TimelineEvent

    text = "The authority said it would not promise to lift the import rule."
    result = frame(1, "2027-01-10", timeline=[TimelineEvent(
        timeline_id="offer", timestamp="2027-01-10", description=text, actors=["actor_a"],
        confidence=1, importance=1,
    )], state_observations=[RoundStateObservation(
        dimension_id="import_rule", category_id="active", actor_ids=["actor_a"],
        event_ref="offer", quote=text,
    )])
    assert save_round_observations(result, [dimension()], {"actor_a"}, {}) == {}


def test_linked_public_issue_supplies_boundary_state_without_duplicate_model_record():
    from app.domain.process import IssueUpdate
    from app.domain.world import TimelineEvent

    d = OutcomeDimension(id="deal", label="Deal", observation_mode="boundary",
                         observation_scope="Public contract status", issue_id="terms", categories=[
        {"id": "pending", "label": "Pending", "criterion": "A proposal was exchanged", "issue_state_codes": ["proposed"]},
        {"id": "signed", "label": "Signed", "criterion": "The proposal was formally signed", "issue_state_codes": ["completed"]},
    ])
    text = "The supplier publicly proposed revising the draft terms."
    result = frame(1, "2027-01-10", timeline=[TimelineEvent(
        timeline_id="proposal", timestamp="2027-01-10", description=text, actors=["actor_a"],
        confidence=1, importance=1,
    )], issue_updates=[IssueUpdate(
        issue_id="terms", title="Deal", after="Proposal sent", kind="proposal", state_code="proposed",
        actor_ids=["actor_a"], event_refs=["proposal"], evidence_quote=text,
    )])
    row = save_round_observations(result, [d], {"actor_a"}, {})["deal"]
    assert row["category_id"] == "pending" and row["source"] == "public_issue_update"


def test_expiry_requires_a_date_in_the_evidence():
    d = dimension()
    text = "The published import rule remains in effect."
    result = frame(1, "2027-01-10", continuing_state_notes=[text], state_observations=[
        RoundStateObservation(dimension_id="import_rule", category_id="active",
                              actor_ids=["actor_a"], quote=text, expires_on="2027-02-01"),
    ])
    row = save_round_observations(result, [d], {"actor_a"}, {})["import_rule"]
    assert row["carry"] is False and row["expires_on"] is None
