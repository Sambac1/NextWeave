"""Decision-relevant dossiers and continuity, with no provider requests."""
import asyncio
from copy import deepcopy
import json

import pytest

from app.domain import ActionProposal, Beliefs, Desire, GameMasterSettlement, RelationshipProjection, WorldSnapshot
from app.domain.decision_core import DecisionCore, CurrentPlan, legacy_core, public_core, validate_core
from app.layers.pipeline.seams import to_agent_profiles
from app.layers.pipeline import profile_service
from app.llm.budget import BudgetExceeded
from app.simulation.action_audit import _program_violations
from app.simulation.bdi_updates import process_agent_information
from app.simulation.context import actor_context, compact_json
from app.simulation.decision_profile import activate_decision_core
from app.simulation.resources import validate_resources
from app.simulation.transitions import apply_round_update
from test_ensemble import actor


def core():
    return {"goals":[{"id":"g1","description":"Maintain supply","timeline_refs":["h1"]}],
            "means":[{"description":"May modify owned production schedules","conditions":"Within its own facilities","timeline_refs":["h1"]}],
            "resources":[{"id":"stock","description":"Inventory available for delivery","value":None,"unit":"units"}],
            "initial_judgments":[{"description":"Supplier response remains uncertain","perspective":"actor",
                                  "timeline_refs":["h1"]},
                                 {"description":"No source establishes the buyer's plan","unknowns":"Alternative capacity"}]}


def test_unknown_values_and_priority_are_not_replaced_by_midpoints():
    result=validate_core(core(),["h1"])
    assert result.resources[0].value is None and result.goals[0].priority=="unknown"
    assert "initial_judgments" not in public_core(result)


def test_claim_deduplication_and_recurrence_requirements():
    raw=core()
    raw["limits"]=[{"description":raw["means"][0]["description"]}]
    raw["decision_rules"]=[{"description":"Repeated preference","conditions":"Supplier rejects offer","timeline_refs":["h1","h1"]}]
    result=validate_core(raw,["h1"])
    assert not result.limits and not result.decision_rules
    raw["decision_rules"][0]["timeline_refs"]=["h1","h2"]
    assert validate_core(raw,["h1","h2"]).decision_rules


def test_bad_references_are_rejected_not_stripped():
    raw=core();raw["means"][0]["timeline_refs"]=["invented"]
    with pytest.raises(ValueError,match="unavailable evidence"):
        validate_core(raw,["h1"])


def test_quantity_without_unit_or_evidence_stays_unknown():
    raw=core();raw["resources"][0]["value"]=42
    assert validate_core(raw,["h1"]).resources[0].value is None


def test_generation_uses_minimal_schema_and_initializes_real_judgments(monkeypatch):
    calls=[]
    async def provider(*args,**kwargs):
        calls.append((args,kwargs));return {"decision_core":core()}
    monkeypatch.setattr(profile_service,"_call_json_with_retries",provider)
    monkeypatch.setattr(profile_service,"_local_retrieved_context",lambda *args:"")
    profile={"id":"lab","canonical_name":"Laboratory","entity_type":"organization","title":"Research institution",
             "history":[{"timeline_id":"h1","timestamp":"2020-01","description":"A supply trial."}]}
    result=asyncio.run(profile_service.generate_profile(api_key="fixture",model="fixture",event_title="Supply",profile=profile,cutoff="2020-01"))
    generated=to_agent_profiles([result],event_id="case",event_title="Supply")[0]
    assert generated.beliefs.situation_views[0].assessment.startswith("Supplier response")
    assert generated.beliefs.situation_views[0].confidence is None
    assert len(generated.beliefs.situation_views) == 1
    assert "Alternative capacity" in generated.decision_core.evidence_gaps
    assert generated.desires[0].priority is None and generated.desires[0].desire_id=="g1"
    assert not generated.capabilities and not generated.constraints
    assert "semantic" not in generated.public_state and len(calls)==1
    assert calls[0][1]["max_tokens"]==3500
    assert "personality" not in generated.decision_core.model_dump()
    assert result["episodic"]["history"]==[{"timeline_id":"h1"}]


def test_profile_budget_is_not_hidden_by_minimal_fallback(monkeypatch):
    async def provider(*args,**kwargs):raise BudgetExceeded("cap")
    monkeypatch.setattr(profile_service,"_call_json_with_retries",provider)
    monkeypatch.setattr(profile_service,"_local_retrieved_context",lambda *args:"")
    with pytest.raises(BudgetExceeded):
        asyncio.run(profile_service.generate_profile(api_key="fixture",model="fixture",event_title="Supply",profile={"canonical_name":"A"}))


def test_numbered_dossier_references_and_targeted_repair(monkeypatch):
    from app.knowledge import profile_evidence
    from app.simulation.registry import eligible_simulation_inputs

    quote = "The laboratory operates the test facility."
    monkeypatch.setattr(profile_evidence, "source_catalog", lambda *args, **kwargs: [{
        "evidence_id": "stored", "text": quote, "url": "https://example.test/source",
        "snapshot_sha256": "snapshot",
    }])
    calls = []

    async def provider(api_key, model, prompt, **kwargs):
        calls.append((prompt, kwargs))
        if len(calls) == 1:
            return {"decision_core": {
                "goals": [{"description": "Maintain supply", "timeline_refs": ["T01"]}],
                "means": [{"description": "Operate the test facility", "source_refs": [{"evidence_id": "S01"}]}],
                "mandate": [{"description": "Approve funding", "source_refs": [{"evidence_id": "S99"}]}],
                "limits": [{"description": "Require approval", "kind": "structural", "timeline_refs": ["T01"]}],
                "resources": [{"description": "Own external inventory", "timeline_refs": ["invented"]}],
            }}
        return {"repairs": [
            {"field": "goals", "index": 0, "item": {"description": "Replace an accepted goal"}},
            {"field": "limits", "index": 0, "item": {
                "description": "Require approval", "kind": "procedural", "timeline_refs": ["T01"],
            }},
            {"field": "mandate", "index": 0, "item": {
                "description": "Approve funding", "source_refs": [{"evidence_id": "S99"}],
            }},
        ]}

    monkeypatch.setattr(profile_service, "_call_json_with_retries", provider)
    profile = {"id": "lab", "canonical_name": "Laboratory", "entity_type": "organization",
               "history": [{"timeline_id": "hist_saved", "description": "The facility requested approval."}]}
    result = asyncio.run(profile_service.generate_profile(
        api_key="fixture", model="fixture", event_title="Supply", profile=profile,
    ))
    decision = result["decision_core"]
    assert len(calls) == 2 and calls[1][1]["attempts"] == 1
    assert "structural" not in decision["limits"][0]["kind"]
    assert decision["goals"][0]["description"] == "Maintain supply"
    assert decision["goals"][0]["timeline_refs"] == ["hist_saved"]
    source = decision["means"][0]["source_refs"][0]
    assert (source["quote"], source["url"], source["snapshot_sha256"]) == (
        quote, "https://example.test/source", "snapshot",
    )
    assert not decision["mandate"] and not decision["resources"]
    assert any("authority" in note for note in decision["review_notes"])
    assert decision["simulation_eligible"] is True
    unsupported = {"decision_core": {"simulation_eligible": False}}
    assert eligible_simulation_inputs([unsupported, result, {}]) == [result, {}]
    with pytest.raises(ValueError, match="No independent decision actors"):
        eligible_simulation_inputs([unsupported])


def test_unrepairable_dossier_is_retained_with_evidence_gap(monkeypatch):
    async def provider(*args, **kwargs):
        if kwargs["attempts"] == 1:
            raise BudgetExceeded("repair budget")
        return {"decision_core": {"means": [{"description": "Operate a facility", "timeline_refs": ["T99"]}]}}

    monkeypatch.setattr(profile_service, "_call_json_with_retries", provider)
    result = asyncio.run(profile_service.generate_profile(
        api_key="fixture", model="fixture", event_title="Supply",
        profile={"id": "lab", "canonical_name": "Laboratory"},
    ))
    decision = result["decision_core"]
    assert decision["origin"] == "generated" and decision["simulation_eligible"] is False
    assert not decision["means"] and decision["review_notes"] and decision["evidence_gaps"]


def test_actor_admission_distinguishes_evidence_and_retryable_failure(monkeypatch):
    from app.simulation.registry import NoEligibleSimulationActors, eligible_simulation_inputs, simulation_readiness

    async def unavailable(*args, **kwargs):
        raise ValueError('Malformed dossier response')

    monkeypatch.setattr(profile_service, '_call_json_with_retries', unavailable)
    fallback = asyncio.run(profile_service.generate_profile(
        api_key='fixture', model='fixture', event_title='Supply', profile={'id': 'lab', 'canonical_name': 'Laboratory'},
    ))
    assert fallback['decision_core']['simulation_eligible'] is False
    fallback['decision_core']['simulation_eligible'] = None  # Archived fallback schema.
    with pytest.raises(NoEligibleSimulationActors) as failure:
        eligible_simulation_inputs([fallback, {}], require_grounded=True)
    assert failure.value.retryable and failure.value.readiness_counts == {'retryable_error': 1, 'needs_evidence': 1}

    evidenced = validate_core(core(), ['h1']).model_dump()
    evidenced['means'][0]['valid_until'] = '2020-01-05'
    candidate = {'decision_core': evidenced}
    assert simulation_readiness(candidate, require_grounded=True, on_date='2020-01-05') == 'restricted'
    assert simulation_readiness(candidate, require_grounded=True, on_date='2020-01-06') == 'needs_evidence'
    assert eligible_simulation_inputs([fallback, candidate], on_date='2020-01-05') == [candidate]


def test_dossier_gets_literal_actor_span_from_revision_before_cutoff(monkeypatch):
    from app.knowledge.profile_evidence import source_catalog

    body = ('== Contest ==\nA different group oversees the general event.\n\n'
            'The Renewal Party presented a joint slate in June. It sought endorsement from local associations.')
    source = {'title': 'Regional contest', 'url': 'https://en.wikipedia.org/wiki/Regional_contest',
              'revision_at': '2020-07-01T10:00:00Z', 'revision_id': 123,
              'retrieved_at': '2021-01-01T00:00:00Z', 'content': body}
    future = source | {'revision_at': '2020-07-01T14:00:00Z', 'revision_id': 124,
                       'content': 'The Renewal Party introduced a later proposal.'}
    profile = {'id': 'renewal', 'canonical_name': 'Renewal Party (RP)', 'entity_type': 'organization'}
    cutoff = '2020-07-01T12:00:00Z'
    spans = source_catalog(profile, cutoff=cutoff, sources=[source, future])
    assert len(spans) == 1 and 'presented a joint slate' in spans[0]['text']
    assert spans[0]['revision_id'] == '123' and spans[0]['section'] == 'source_01/Contest'
    assert body[spans[0]['span_start']:spans[0]['span_end']] == spans[0]['text']

    async def provider(_api_key, _model, prompt, **kwargs):
        assert spans[0]['text'] in prompt
        assert 'later proposal' not in prompt
        return {'decision_core': {'means': [{'description': 'Can put forward a joint slate',
            'source_refs': [{'evidence_id': 'S01'}], 'claim_scope': 'conditional'}]}}

    monkeypatch.setattr(profile_service, '_call_json_with_retries', provider)
    result = asyncio.run(profile_service.generate_profile(
        api_key='fixture', model='fixture', event_title='Regional contest', profile=profile,
        cutoff=cutoff, source_records=[source, future],
    ))
    support = result['decision_core']['means'][0]['source_refs'][0]
    assert result['decision_core']['simulation_eligible'] is True
    assert support['revision_id'] == '123' and body[support['span_start']:support['span_end']] == support['quote']


def test_legacy_projection_keeps_archive_and_does_not_promote_old_limits():
    raw=actor().model_dump();raw["constraints"]=["Sensitive to criticism"]
    raw["desires"]=[{"desire_id":"g1","description":"Maintain supply","priority":1,"status":"active"}]
    before=deepcopy(raw);projected=legacy_core(raw)
    assert raw==before and projected.goals[0].priority=="unknown"
    assert projected.limits[0].kind=="unknown" and projected.limits[0].provenance=="legacy_unreviewed"
    assert not projected.means and not projected.decision_rules


def test_new_context_omits_persona_scores_and_duplicate_goals():
    a=actor();a.decision_core=validate_core(core(),["h1"])
    a.desires=[Desire(desire_id="g1",description="Maintain supply")]
    text=compact_json(actor_context(a))
    assert "risk_tolerance" not in text and "behavioral_model" not in text
    assert "confidence" not in text and text.count("Maintain supply")==1
    assert "null" in text  # Unknown plan remains explicit.


def action(plan=None,decision="wait"):
    return ActionProposal(agent_id="actor_a",agent_name="Laboratory",round_id=1,snapshot_version=0,
        action_id="action_1_actor_a",decision=decision,action_type="proposal" if decision=="act" else "no_action",
        action="Continue consultation" if decision=="act" else "Wait",reasoning="Current conditions",confidence=.5,plan_update=plan)


def settlement():
    return GameMasterSettlement(state_summary="Unchanged",next_world_state={},analysis="No external change",agent_knowledge_updates=[])


def test_wait_preserves_plan_and_valid_change_commits_with_round():
    a=activate_decision_core(actor());a.desires=[Desire(desire_id="g1",description="Maintain supply")]
    a.current_plan=CurrentPlan(description="Continue consultation",goal_ids=["g1"],review_when="New response")
    snapshot=WorldSnapshot(event_title="Supply",agents=[a])
    after=apply_round_update(snapshot,[action()],settlement(),RelationshipProjection())
    assert after.agents[0].current_plan==a.current_plan
    revised=CurrentPlan(description="Revise terms",goal_ids=["g1"],change_reason="Supplier rejected prior terms")
    after=apply_round_update(snapshot,[action(revised,"act")],settlement(),RelationshipProjection())
    assert after.agents[0].current_plan.updated_round==1
    assert a.current_plan.description=="Continue consultation"  # Input remains frozen.


def test_single_action_does_not_become_an_ongoing_intention():
    agent = activate_decision_core(actor())
    state = WorldSnapshot(event_title="Supply", agents=[agent])
    after = apply_round_update(state, [action(decision="act")], settlement(), RelationshipProjection())
    assert after.agents[0].current_plan is None
    assert after.agents[0].historical_actions or after.agents[0].private_action_history


def test_historical_restrictions_need_current_support_in_beliefs():
    from app.domain.agents import SituationView
    from app.simulation.decision_profile import current_constraint_beliefs

    rule = {"id": "old", "description": "Facility visitors need clearance.",
            "claim_scope": "conditional", "provenance": "source_statement",
            "timeline_refs": ["h1"], "valid_from": "2017-01-01", "valid_until": "2019-08-31"}
    state = DecisionCore(limits=[rule])
    beliefs = Beliefs(situation_views=[
        SituationView(topic="claimed restriction", assessment="The agency maintains a legal travel restriction."),
        SituationView(topic="historical record", assessment="The restriction was extended through 2019."),
        SituationView(topic="uncertain validity", assessment="It is unclear whether the restriction remains in force."),
        SituationView(topic="cited restriction", assessment="Everyone is forbidden from travelling.", constraint_ref="old"),
    ])
    filtered, rejected = current_constraint_beliefs(beliefs, state, "2026-11-06")
    assert [row.topic for row in filtered.situation_views] == ["historical record", "uncertain validity"]
    assert rejected == ["missing_current_constraint_reference", "old"]
    supported, rejected = current_constraint_beliefs(
        Beliefs(situation_views=[beliefs.situation_views[-1]]), state, "2018-02-01")
    assert not rejected and supported.situation_views[0].assessment == rule["description"]


def test_unknown_plan_goal_rejected_by_audit_and_transition():
    a=activate_decision_core(actor());snapshot=WorldSnapshot(event_title="Supply",agents=[a])
    proposal=action(CurrentPlan(description="New plan",goal_ids=["invented"]))
    assert "UNKNOWN_PLAN_GOAL" in _program_violations(proposal,a,snapshot)
    with pytest.raises(ValueError,match="unknown goal"):
        apply_round_update(snapshot,[proposal],settlement(),RelationshipProjection())


def test_new_goal_update_preserves_content_and_rejects_silent_goal_replacement():
    from app.domain.agents import PrivateObservation
    a=activate_decision_core(actor());a.desires=[Desire(desire_id="g1",description="Maintain supply")]
    a.private_observations=[PrivateObservation(observation_id="p",description="New response",received_round=0)]
    class Provider:
        async def json_call(self,**kw):
            return kw["schema"](agent_id="actor_a",round_id=1,snapshot_version=0,assessment="Updated",
                bdi_patch={"desires":[{"desire_id":"g2","description":"Unrelated goal","priority":1}]})
    after=asyncio.run(process_agent_information(WorldSnapshot(event_title="Supply",agents=[a]),Provider(),"test"))
    assert after.agents[0].desires[0].description=="Maintain supply"
    assert any(v["code"]=="goal_update_rejected" for v in after.validation_issues)


def test_resource_projection_preserves_units_and_unknown_values():
    before={"actor_resources":{"a":{"stock":{"value":None,"unit":"units"}}}}
    validate_resources(before,deepcopy(before))
    after=deepcopy(before);after["actor_resources"]["a"]["stock"]["value"]=10
    with pytest.raises(ValueError,match="model-invented"):
        validate_resources(before,after)
    validate_resources(before,after,{"actor_resources.a.stock.value":10})
    after["actor_resources"]["a"]["stock"]["unit"]="currency"
    with pytest.raises(ValueError,match="units"):
        validate_resources(before,after)
    with pytest.raises(ValueError,match="actors"):
        validate_resources(before,{})


def test_public_projection_does_not_export_subjective_judgments():
    from app.knowledge.hierarchy import public_profiles
    a=actor().model_dump();a["decision_core"]=validate_core(core(),["h1"]).model_dump()
    views,_=public_profiles([a],"case","Supply")
    assert "initial_judgments" not in views[0]["decision_core"]
    assert "behavioral_model" not in views[0] and "capability_notes" not in views[0]


def test_strength_dimension_generation_no_longer_calls_provider(monkeypatch):
    async def forbidden(*args,**kwargs):raise AssertionError("No model needed for a numeric ladder")
    monkeypatch.setattr(profile_service,"_call_json_with_retries",forbidden)
    assert asyncio.run(profile_service._generate_power_dimensions(event_title="Supply"))==[]


def test_saved_sample_bdi_projection_does_not_expose_private_memories(monkeypatch):
    from app.api import experiments

    class Store:
        def sample(self, run_id, sample_id):
            return {"snapshot": {"round_id": 3, "agents": [{
                "agent_id": "a", "name": "Actor", "beliefs": {"situation_views": []},
                "desires": [], "private_observations": [{"content": "secret"}],
                "private_action_history": ["secret"], "public_state": {"secret": 1},
            }]}}

    monkeypatch.setattr(experiments, "_existing", lambda run_id: (Store(), {}))
    result = experiments.experiment_sample_actor_states("test", 1)
    assert result["round_id"] == 3
    assert result["state_scope"] == "latest_saved_snapshot"
    assert result["actors"][0]["beliefs"] == {"situation_views": []}
    assert set(result["actors"][0]) == {"agent_id", "name", "beliefs", "desires", "current_plan"}
    assert "secret" not in str(result)


def test_question_forecast_requires_usable_bdi_and_accepts_a_cited_current_intention():
    from app.domain.decision_core import validate_core
    from app.simulation.decision_profile import initial_plan
    from app.simulation.registry import simulation_readiness

    actor = {"name": "Example Board", "agent_type": "organization"}
    mandate = {"description": "The board can approve the transaction.", "timeline_refs": ["historical"]}
    intention = {"description": "The board committed to put the proposal to a vote.", "provenance": "source_statement",
                 "claim_scope": "conditional", "timeline_refs": ["historical"], "valid_until": "2026-11-20"}

    def profile(intentions):
        core = validate_core({"mandate": [mandate], "initial_intentions": intentions},
                             allowed_refs=["historical"], as_of="2026-11-10")
        return core, actor | {"decision_core": core.model_dump()}

    _, reference = profile([])
    assert simulation_readiness(reference, require_grounded=True, require_bdi=True) == "needs_bdi"
    assert simulation_readiness(reference, require_grounded=True) == "restricted"
    core, active = profile([intention])
    assert initial_plan(core).description == intention["description"]
    assert simulation_readiness(active, require_grounded=True, require_bdi=True) == "restricted"
    assert simulation_readiness(active, require_grounded=True, require_bdi=True, on_date="2026-11-21") == "needs_bdi"
    inferred, _ = profile([intention | {"provenance": "inferred"}])
    assert initial_plan(inferred) is None
