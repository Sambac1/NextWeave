import asyncio
import json
from typing import get_args

import pytest
from pydantic import ValidationError

from app.domain.activities import ActivityStage
from app.domain.ensemble import (
    ClassificationBatch, EventQuestion, OutcomeDimension, OutcomePlan, ReassessmentRequest, TerminalJudgment,
)
from app.simulation.ensemble import reassess_experiment, saved_reassessment, wilson
from app.simulation.event_metrics import resolve_answer
from app.simulation.terminal_outcomes import summarize_dimensions, validate_judgments
from test_ensemble import request, runner, Classifier
from app.simulation.ensemble import prepare_experiment, run_experiment
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository


DELIVERY = OutcomeDimension(id="delivery", label="Supply delivery", categories=[
    {"id": "delivered", "label": "Delivered", "criterion": "The shipment arrived",
     "required_actor_ids": ["actor_a"], "evidence_stages": ["completed"]},
    {"id": "cancelled", "label": "Cancelled", "criterion": "The shipment was cancelled"},
])
NEGOTIATION = OutcomeDimension(id="contract", label="Contract status", categories=[
    {"id": "proposed", "label": "Proposed", "criterion": "A proposal remains open",
     "evidence_stages": ["proposed"]},
    {"id": "signed", "label": "Signed", "criterion": "The contract was signed",
     "evidence_stages": ["accepted"]},
])


def claim(dimension, category, ref, quote):
    return TerminalJudgment(dimension_id=dimension, category_id=category,
                            matched_category_ids=[category], event_refs=[ref], evidence_quote=quote)


def test_generated_plan_accepts_all_valid_evidence_stages_without_truncation():
    stages = list(get_args(ActivityStage))
    dimension = DELIVERY.model_dump()
    category = dimension["categories"][1]
    category["evidence_stages"] = [*stages, stages[0]]

    plan = OutcomePlan.model_validate_json(json.dumps({"outcome_dimensions": [dimension]}))
    assert plan.outcome_dimensions[0].categories[1].evidence_stages == stages
    assert "cancelled" in stages

    category["evidence_stages"] = [*stages[:-1], "unrecognized_stage"]
    with pytest.raises(ValidationError, match="evidence_stages"):
        OutcomePlan.model_validate({"outcome_dimensions": [dimension]})


def test_terminal_evidence_and_independent_dimensions():
    proposal = "The laboratory proposed a new contract."
    record = {"rounds": [
        {"timeline": [{"timeline_id": "early", "description": "The laboratory completed delivery."}]},
        {"timeline": [{"timeline_id": "last", "description": proposal, "actors": ["actor_a"]}]},
    ]}
    for invalid in (
        claim("delivery", "delivered", "early", "The laboratory completed delivery."),
        claim("delivery", "delivered", "last", proposal),
        claim("delivery", "delivered", "last", "The shipment arrived."),
    ):
        outcomes = validate_judgments([DELIVERY, NEGOTIATION],
                                      [invalid, claim("contract", "proposed", "last", proposal)], record)
        assert [result.category_id for result in outcomes] == ["other", "proposed"]
    summaries = summarize_dimensions([DELIVERY, NEGOTIATION], [
        {"sample_id": 1, "classification": {"terminal_judgments": [v.model_dump() for v in outcomes]}}
    ], 2, wilson)
    assert summaries[0]["unclear_count"] == 1 and summaries[0]["categories"][0]["frequency"] is None
    assert summaries[1]["categories"][0]["frequency"] == 1
    assert summaries[1]["categories"][0]["sampling_interval_95"] is None


def test_start_does_not_match_continuation():
    question = EventQuestion(id="launch", label="Launch", criterion="New service launches",
                             milestone="started", required_actor_ids=["actor_a"])
    record = {"snapshot": {"agents": [{"agent_id": "actor_a"}]}, "rounds": [{"timeline": [
        {"timeline_id": "continues", "description": "The company continued operating the service.", "actors": ["actor_a"]},
        {"timeline_id": "starts", "description": "The company started operating the new service.", "actors": ["actor_a"]},
    ]}]}
    for event in record["rounds"][0]["timeline"]:
        answer = resolve_answer(question, {"question_id": "launch", "answer": "yes",
                                "event_refs": [event["timeline_id"]], "evidence_quote": event["description"]},
                                record, strict_occurrence=True)
        assert answer.answer == ("yes" if event["timeline_id"] == "starts" else "unclear")


def test_saved_paths_can_be_reassessed_without_rerunning_or_overwriting(tmp_path):
    class ReviewClient:
        recorder = None

        async def json_call(self, **options):
            assert options["schema"] is ClassificationBatch
            payload = json.loads(options["prompt"])
            assert len(payload["plan"]["outcome_dimensions"]) == 2
            assert all("final_state" not in sample for sample in payload["samples"])
            return ClassificationBatch(samples=[
                {"sample_id": sample["sample_id"], "questions": [{"question_id": "move", "answer": "unclear"}]}
                for sample in payload["samples"]
            ])

    store = EnsembleStore(tmp_path / "runs")
    run_id, options = prepare_experiment(request(), store=store, events=EventRepository(tmp_path / "events"))
    calls = []
    original = asyncio.run(run_experiment(run_id, options, store=store, runtime=Classifier(), round_runner=runner(calls)))
    baseline = store.load(run_id, "config")
    before = store.sample(run_id, 1)
    review = asyncio.run(reassess_experiment(
        run_id, ReassessmentRequest(outcome_dimensions=[DELIVERY, NEGOTIATION]),
        store=store, config=baseline, client=ReviewClient(),
    ))
    assert len(calls) == 8
    assert store.load(run_id, "result") == original and store.sample(run_id, 1) == before
    assert saved_reassessment(run_id, store, original) == review
    assert saved_reassessment(run_id, store, original | {"completed_samples": 2}) is None
    assert review["result"]["completed_samples"] == 4
    assert all(d["unclear_count"] == 4 for d in review["result"]["outcome_dimensions"])
