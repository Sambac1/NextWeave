import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.experiments import router
from app.domain.ensemble import StateInvariant, OutcomeDefinition
from app.simulation.measurements import validate_invariants, classify_by_rules
from app.simulation.ensemble import prepare_experiment, run_experiment
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository
from test_ensemble import request, runner


@pytest.mark.parametrize("before,after,rules", [
    ({"stock": 10}, {"stock": -1}, [{"paths": ["stock"], "rule": "range", "minimum": 0}]),
    ({"target": 5}, {"target": 6}, [{"paths": ["target"], "rule": "fixed"}]),
    ({"delivered": 5}, {"delivered": 4}, [{"paths": ["delivered"], "rule": "nondecreasing"}]),
    ({"a": 4, "b": 6}, {"a": 3, "b": 8}, [{"paths": ["a", "b"], "rule": "conserved_sum"}]),
])
def test_domain_neutral_state_constraints_fail_closed(before, after, rules):
    with pytest.raises(ValueError):
        validate_invariants(before, after, [StateInvariant(**r) for r in rules])


def test_conserved_resource_transfer_and_programmatic_outcomes():
    validate_invariants({"a": 4, "b": 6}, {"a": 2, "b": 8},
                        [StateInvariant(paths=["a", "b"], rule="conserved_sum")])
    definitions = [OutcomeDefinition(id="met", label="Met", criterion="Target achieved", conditions=[{"path": "delivered", "operator": "ge", "value": 5}]),
                   OutcomeDefinition(id="unmet", label="Unmet", criterion="Below target", conditions=[{"path": "delivered", "operator": "lt", "value": 5}])]
    assert classify_by_rules({"delivered": 5}, definitions) == "met"
    assert classify_by_rules({"delivered": 4}, definitions) == "unmet"
    assert classify_by_rules({}, definitions) == "other"


def test_structured_outcome_rules_need_no_classifier_calls(tmp_path):
    store, events = EnsembleStore(tmp_path/"runs"), EventRepository(tmp_path/"events")
    req = request()
    req.event_questions = []
    req.outcome_definitions = [OutcomeDefinition(id="odd", label="Odd", criterion="Odd owner", conditions=[{"path": "owner", "operator": "eq", "value": 1}]),
                              OutcomeDefinition(id="remaining", label="Others", criterion="Other owner", conditions=[{"path": "owner", "operator": "gt", "value": 1}])]
    run_id, req = prepare_experiment(req, events=events, store=store)
    class NoModel:
        async def json_call(self, **kw): raise AssertionError("No classification model call needed")
    result = asyncio.run(run_experiment(run_id, req, store=store, runtime=NoModel(), round_runner=runner([])))
    assert result["completed_samples"] == 4
    assert [c["count"] for c in result["categories"]] == [1, 3, 0]
    assert result["model_calls_this_attempt"] == 0


def test_heldout_scoring_and_duplicate_case_protection():
    app = FastAPI(); app.include_router(router)
    client = TestClient(app)
    observation = {"case_key": "heldout_1", "probabilities": {"a": .8, "b": .2}, "observed_category": "a"}
    data = client.post("/api/forecast-evaluation", json={"observations": [observation]}).json()
    assert data["multiclass_brier"] == pytest.approx(.08)
    assert data["uniform_baseline_brier"] == .5
    assert data["calibration_bins"][0]["accuracy"] == 1
    assert client.post("/api/forecast-evaluation", json={"observations": [observation, observation]}).status_code == 422
