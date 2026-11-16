"""Question-led actors: coverage, attribution and bounded partial dossiers."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib

import pytest

from app.knowledge.decision_seats import (
    SeatGapRepair, SeatPlan, bind_plan, repair_missing_seats, seat_figures, bind_dossier, planning_sources,
    actor_coverage, review_dossier_owners, OwnershipReview, OwnershipIssue,
)
from app.domain.decision_core import DecisionCore, DecisionItem
from app.layers.pipeline.helpers import _filter_by_min_evidence
from app.simulation.registry import simulation_readiness


def fixture():
    quote = 'Boreal Council stated its conditions for talks with Atlas on July 2, 2025.'
    catalog = {'S1': {'text':quote, 'title':'Atlas–Boreal relations',
        'url':'https://example.test/relations','revision_id':17,'revision_at':'2025-07-03T00:00:00Z',
        'snapshot_sha256':hashlib.sha256(quote.encode()).hexdigest()}}
    plan = SeatPlan.model_validate({'seats':[{'name':'Boreal Council','entity_type':'organization',
        'role':'Counterpart for the proposed talks','required':True,'decision_ids':['counterpart_response'],
        'evidence':[{'source_id':'S1','quote':quote}], 'basic_actions':[{
            'kind':'communicate','description':'State conditions for talks.',
            'conditions':'No agreement or implementation is implied; authorization for commitments stays unknown.',
            'evidence':[{'source_id':'S1','quote':quote}]}]}],
        'requirements':[{'decision_id':'counterpart_response','description':'Respond to proposed talks.',
                         'owner':'Boreal Council'}]})
    return bind_plan(plan, catalog, 'Will Atlas hold talks with Boreal Council?'), plan, catalog


def test_counterpart_missing_from_timeline_is_a_retained_candidate_not_a_fake_event():
    plan, _, _ = fixture()
    rows = seat_figures(plan, [])
    kept, removed, _ = _filter_by_min_evidence(rows, .1, 3, 100, 1)
    assert len(kept) == 1 and not removed
    assert kept[0]['evidence_count'] == 0
    assert not kept[0].get('history')


def test_a_registered_person_can_represent_a_government_without_becoming_an_organization():
    from app.domain.agency import normalize_entity_record
    from app.knowledge.decision_seats import restore_seat_type

    plan, _, _ = fixture()
    seat = plan["seats"][0]
    seat.update(owner_name="Morgan Hale", entity_type="person", represents="Boreal",
                role="Boreal leader whose government controls approval.")
    figure = seat_figures(plan, [])[0]
    assert normalize_entity_record(figure)["entity_type"] == "person"
    binding = bind_dossier(DecisionCore(), figure).model_dump()
    assert binding["seat"]["entity_type"] == "person"
    assert binding["seat"]["represents"] == "Boreal"
    binding["seat"].pop("entity_type")
    saved = {"name": "Morgan Hale", "agent_type": "organization", "decision_core": binding}
    projected = normalize_entity_record(restore_seat_type(saved, plan))
    assert projected["agent_type"] == "person"
    assert saved["agent_type"] == "organization" and "entity_type" not in saved["decision_core"]["seat"]


def test_unknown_intentions_do_not_erase_supported_nonbinding_participation():
    from app.simulation.context import actor_context
    from app.simulation.registry import profile_from_input

    plan, _, _ = fixture()
    profile = seat_figures(plan, [])[0]
    core = bind_dossier(DecisionCore(evidence_gaps=['Intentions and resources are unknown.']), profile)
    assert not core.goals and not core.resources
    assert core.means[0].provenance == 'inferred'
    assert 'No agreement' in core.means[0].conditions
    assert core.means[0].holder == 'Boreal Council'
    assert core.seat.owner_name == 'Boreal Council'
    actor = {'name':'Boreal Council','entity_type':'organization','decision_core':core.model_dump()}
    assert simulation_readiness(actor, require_grounded=True, on_date='2025-07-04') == 'restricted'
    assert simulation_readiness(actor | {'name':'Atlas'}, require_grounded=True) == 'retryable_error'
    policy = {'risk_tolerance':.5, 'compromise_tendency':.5, 'response_speed':.5}
    assert actor_context(profile_from_input(actor | {'agent_type':'organization', 'beliefs':{}, 'decision_policy':policy, 'role':'Counterpart',
        'description':'Counterpart to the proposed talks.', 'decision_subject':'Boreal Council'}, None, 'Talks'))['decision_readiness'] == 'restricted'


def test_basic_role_cannot_grant_physical_execution_or_another_actors_authority():
    plan, _, _ = fixture()
    profile = seat_figures(plan, [])[0]
    bad = deepcopy(profile)
    bad['decision_seat']['basic_actions'][0]['kind'] = 'execute'
    with pytest.raises(ValueError, match='execution authority'):
        bind_dossier(DecisionCore(), bad)
    with pytest.raises(ValueError, match='different actor'):
        bind_dossier(DecisionCore(means=[DecisionItem(description='Approve another body decision.', holder='Atlas')]), profile)
    with pytest.raises(ValueError, match='registered decision seat'):
        bind_dossier(DecisionCore(), profile | {'canonical_name':'Atlas'})


def test_plan_rejects_duplicated_decision_ownership_but_allows_independent_approvals():
    _, plan, catalog = fixture()
    second = plan.seats[0].model_copy(deep=True, update={'name':'Atlas'})
    duplicate = plan.model_copy(deep=True, update={'seats':[plan.seats[0], second]})
    with pytest.raises(ValueError, match='multiple seats'):
        bind_plan(duplicate,catalog,'Will Atlas hold talks with Boreal Council?')
    second.decision_ids = ['initiator_proposal']
    independent = duplicate.model_copy(deep=True, update={'seats':[plan.seats[0],second],
        'requirements':[*plan.requirements, type(plan.requirements[0])(
            decision_id='initiator_proposal',description='Submit proposal.',owner='Atlas')]})
    assert len(bind_plan(independent,catalog,'Will Atlas hold talks with Boreal Council?')['seats']) == 2


def test_plan_quotes_are_literal_and_unresolved_rights_stay_visible():
    plan, raw, catalog = fixture()
    ref = plan['seats'][0]['source_refs'][0]
    assert catalog['S1']['text'][ref['span_start']:ref['span_end']] == ref['quote']
    invalid = raw.model_copy(deep=True)
    invalid.seats[0].evidence[0].quote = 'An invented authorization to enter.'
    with pytest.raises(ValueError, match='absent'):
        bind_plan(invalid,catalog,'Question')
    raw.requirements.append(type(raw.requirements[0])(decision_id='formal_approval',description='Find approving body.',owner=None))
    bound = bind_plan(raw,catalog,'Question')
    coverage = actor_coverage(bound,[],'2025-07-04')
    assert coverage['status'] == 'limited'
    assert coverage['missing_required'] == ['Boreal Council']
    assert coverage['unresolved_requirements'] == ['Find approving body.']


def test_essential_owner_must_appear_in_its_citation():
    _, raw, catalog = fixture()
    raw.requirements[0].essential = True
    supported = bind_plan(raw, catalog, 'Will Atlas hold talks with Boreal Council?')
    profile = seat_figures(supported, [])[0]
    core = bind_dossier(DecisionCore(), profile)
    actor = {'name':'Boreal Council', 'entity_type':'organization', 'decision_core':core.model_dump()}
    assert not actor_coverage(supported, [actor], '2025-07-04').get('missing_essential')
    catalog['S2'] = dict(catalog['S1'], text='The foreign delegation reported that a meeting had taken place.')
    raw.seats[0].evidence = [raw.seats[0].evidence[0].model_copy(update={'source_id': 'S2', 'quote': ''})]
    bound = bind_plan(raw, catalog, 'Will Atlas hold talks with Boreal Council?')
    assert not bound['seats']
    assert actor_coverage(bound, [])['missing_essential'] == [raw.requirements[0].description]


def test_normalized_quote_spacing_restores_the_literal_source_span():
    _, plan, catalog = fixture()
    original = catalog['S1']['text'].replace('Boreal Council', 'Boreal\u00a0Council')
    catalog['S1']['text'] = original
    bound = bind_plan(plan,catalog,'Will Atlas hold talks with Boreal Council?')
    ref=bound['seats'][0]['source_refs'][0]
    assert ref['quote'] == original
    assert original[ref['span_start']:ref['span_end']] == ref['quote']


def test_evidence_id_selection_copies_original_text_without_model_transcription():
    _,plan,catalog=fixture()
    plan.seats[0].evidence[0].quote=''
    plan.seats[0].basic_actions[0].evidence[0].quote=''
    bound=bind_plan(plan,catalog,'Question')
    assert bound['seats'][0]['source_refs'][0]['quote']==catalog['S1']['text']
    assert bound['seats'][0]['basic_actions'][0]['source_refs'][0]['quote']==catalog['S1']['text']


def test_expanded_owner_chooses_literal_source_abbreviation_and_preserves_decision_ids():
    from app.knowledge.decision_seats import literal_owner
    assert literal_owner('United States Congress','The U.S. Congress holds the authority.')=='U.S. Congress'
    assert literal_owner('Federal Review Board','The FRB considers the request.')=='FRB'
    assert literal_owner('Imaginary New Institution','No such organization is named.') is None


def test_observers_are_not_forced_into_the_decision_roster():
    plan,raw,catalog=fixture()
    raw.requirements[0].kind='observation'
    bound=bind_plan(raw,catalog,'Question')
    assert not bound['seats'] and bound['status']=='observation_only'
    assert actor_coverage(bound,[])['status']=='observation_only'


def test_observer_pseudo_seat_cannot_invalidate_the_decision_roster():
    from pydantic import ValidationError

    _, plan, catalog = fixture()
    raw = plan.model_dump()
    name = 'Official information from the relevant government or consensus reporting ' * 3
    raw['requirements'].append({'decision_id':'confirmation','description':'Confirm public occurrence.',
                                'owner':name,'kind':'observation'})
    observer = raw['seats'][0] | {'name':name,'decision_ids':['confirmation']}
    raw['seats'].append(observer)
    bound = bind_plan(SeatPlan.model_validate(raw),catalog,'Question')
    assert [seat['owner_name'] for seat in bound['seats']] == ['Boreal Council']
    assert bound['requirements'][-1]['owner'] is None

    observer['decision_ids'].append('counterpart_response')
    with pytest.raises(ValidationError, match='at most 120 characters'):
        SeatPlan.model_validate(raw)


def test_seat_planning_retries_a_structured_validation_failure_once(monkeypatch):
    import json
    from types import SimpleNamespace
    from pydantic import ValidationError
    import app.knowledge.decision_seats as seats
    from app.llm.client import LLMError

    _, plan, catalog = fixture()
    source = catalog['S1'] | {'content':catalog['S1']['text'], 'revision_at':'2025-07-03T00:00:00Z'}
    raw = plan.model_dump()
    for field in ('evidence','basic_actions'):
        refs = raw['seats'][0][field] if field=='evidence' else raw['seats'][0][field][0]['evidence']
        refs[0]['source_id'] = 'E1'
    calls = []
    class Runtime:
        def __init__(self, *_):
            pass
        async def json_call(self, **kwargs):
            payload = json.loads(kwargs['prompt'])
            calls.append(payload)
            if len(calls)==1:
                try:
                    SeatPlan.model_validate(raw | {'seats':[raw['seats'][0] | {'name':'x'*121}]})
                except ValidationError as cause:
                    raise LLMError('invalid plan') from cause
            return SeatPlan.model_validate(raw)

    class Events:
        records = {'decision_context':{'question':'Question','rules':'The council decides.'}}
        def load_sources(self, _):
            return [source]
        def load_artifact(self, _eid, name, default):
            return self.records.get(name,default)
        def save_artifact(self, _eid, name, value):
            self.records[name] = value

    monkeypatch.setattr(seats,'LLMClient',Runtime)
    req = SimpleNamespace(event_id='fixture', event_title='Question', evidence_cutoff='2025-07-04',
                          max_key_figures=4, required_decision_subject='Boreal Council', api_key=None,
                          model='fixture', base_url=None, language='en-US')
    events = Events()
    assert asyncio.run(seats.prepare_seat_plan(req,events))['status']=='ready'
    assert len(calls)==2 and calls[0]['required_subject']=='Boreal Council'
    assert calls[1]['repair']['format_issues']==[{'loc':['seats',0,'name'],'type':'string_too_long'}]


def test_unidentified_institution_stays_an_unresolved_requirement_not_an_invented_actor():
    _,raw,catalog=fixture()
    raw.seats[0].name='Unidentified Named Office'
    raw.requirements[0].owner='Unidentified Named Office'
    bound=bind_plan(raw,catalog,'Question')
    assert bound['status']=='limited' and not bound['seats']
    assert bound['requirements'][0]['owner'] is None and bound['gaps']
    assert not planning_sources([{'content':'An agency approved a measure.'}], 'Question','invalid-cutoff')


def test_local_institution_title_binds_an_office_in_its_own_authority_clause():
    _, raw, catalog = fixture()
    local = '== Boreal ==\nAccording to the constitution, the President may act only if the Parliament rejects the proposal.'
    catalog['S1']['text'] = local
    pres = type(raw.seats[0]).model_validate(raw.seats[0].model_dump() | {
        'name':'President of Boreal', 'entity_type':'person', 'decision_ids':['sign'],
        'evidence':[{'source_id':'S1'}], 'basic_actions':[],
    })
    assembly = pres.model_copy(update={'name':'Boreal Parliament (Lower and Upper Houses)', 'entity_type':'government',
                                       'decision_ids':['reject']})
    plan = SeatPlan.model_validate({'seats':[pres.model_dump(),assembly.model_dump()], 'requirements':[
        {'decision_id':'sign','description':'Make a decision within the rule.','owner':'President of Boreal'},
        {'decision_id':'reject','description':'Consider the proposal.','owner':assembly.name},
    ]})
    bound = bind_plan(plan, catalog, 'Will the proposal proceed?')
    assert [row['owner_name'] for row in bound['seats']] == ['President of Boreal','Boreal Parliament']
    assert all(row['entity_type'] == 'government' for row in bound['seats'])
    assert all(row['source_refs'][0]['quote'] == local for row in bound['seats'])
    for figure in seat_figures(bound, []):
        core = bind_dossier(DecisionCore(), figure)
        assert simulation_readiness({'name':figure['canonical_name'],'decision_core':core.model_dump()},
                                    require_grounded=True) == 'restricted'
    for wrong in ('== Atlas ==\nThe President may act if the Parliament rejects the proposal.',
                  '== Boreal ==\nOnly voting is discussed.\n== Atlas ==\nThe President may act if the Parliament rejects the proposal.',
                  '== Boreal ==\nThe Atlas Parliament rejects the proposal and the President of Atlas may act.'):
        catalog['S1']['text'] = wrong
        assert bind_plan(plan, catalog, 'Will the proposal proceed?')['seats'] == []


def test_gap_recovery_preserves_the_existing_seat_and_marks_narrower_authority():
    _, raw, catalog = fixture()
    raw.requirements.append(type(raw.requirements[0])(decision_id='approve',description='Atlas grants approval.',owner=None))
    before = bind_plan(raw,catalog,'Will Atlas hold talks with Boreal Council?')
    catalog['S2'] = catalog['S1'] | {'text':'Jordan Lee led the Atlas delegation in the earlier talks.'}
    seat = raw.seats[0].model_dump() | {'name':'Jordan Lee','entity_type':'person','decision_ids':['approve'],
        'evidence':[{'source_id':'S2'}], 'basic_actions':[{'kind':'request','description':'Request instructions for talks.',
            'conditions':'A fresh appointment remains unknown.','evidence':[{'source_id':'S2'}]}]}
    repair = SeatGapRepair.model_validate({'seats':[seat | {'decision_ids':['counterpart_response']},
                                                  seat | {'evidence':[{'source_id':'S1'}]},seat]})
    assert repair_missing_seats(raw,before,repair,catalog,'Question',2,allowed_citations={'S1'}) == before
    repaired = repair_missing_seats(raw,before,repair,catalog,'Question',2,allowed_citations={'S1','S2'})
    assert repaired['seats'][0] == before['seats'][0]
    assert repaired['seats'][1]['conditional_participation']
    figures = seat_figures(repaired,[])
    agents = []
    for fig in figures:
        actor_name = fig['canonical_name']
        core = bind_dossier(DecisionCore(mandate=[DecisionItem(description='Claims full approval authority.',holder=actor_name)]),fig)
        if actor_name == 'Jordan Lee':
            assert not core.mandate and [x.description for x in core.means] == ['Request instructions for talks.']
        agents.append({'name':actor_name,'decision_core':core.model_dump()})
    coverage = actor_coverage(repaired,agents)
    assert coverage['status'] == 'limited'
    assert [x['support'] for x in coverage['decision_dependencies']] == ['represented','conditional']


def test_optional_gap_repair_format_error_preserves_the_working_plan():
    from app.knowledge.decision_seats import recover_seat_gaps
    from app.llm.client import LLMError

    _, raw, catalog = fixture()
    raw.requirements.append(type(raw.requirements[0])(decision_id='approve',description='Atlas grants approval.',owner=None))
    before = bind_plan(raw,catalog,'Question')

    class Runtime:
        calls = 0

        async def json_call(self, **kwargs):
            self.calls += 1
            assert kwargs['schema'] is SeatGapRepair and kwargs['max_attempts'] == 1
            raise LLMError('Optional response exceeded the role field length')

    runtime = Runtime()
    payload = {'max_seats':3, 'question':'Question', 'rules':'Approve the proposal.', 'as_of':'2025-07-04', 'sources':catalog}
    result = asyncio.run(recover_seat_gaps(raw,before,catalog,payload,runtime,'test'))
    assert runtime.calls == 1
    assert result.pop('gap_repair') == {'status':'unavailable','added':0}
    assert result == before


@pytest.mark.parametrize('source,extra,accepted', [
    ('Jordan Lee was expected to attend the talks.', '', False),
    ('Jordan Lee was injured. A possible meeting with the other side was under discussion.', '', False),
    ('Jordan Lee led the delegation in the prior talks.', 'Jordan Lee later died.', False),
    ('Jordan Lee led the delegation in the prior talks.', 'Morgan Hale was killed.', True),
])
def test_gap_recovery_requires_actual_personal_participation(source,extra,accepted):
    from app.knowledge.decision_seats import _documented_participant
    assert _documented_participant('Jordan Lee',[{'quote':source}],{'S1':{'text':extra}}) is accepted


def test_planning_catalog_preserves_cutoff_and_raw_offsets_for_full_packet():
    good = {'content':'Council can respond to a proposal.', 'revision_at':'2025-01-01T00:00:00Z'}
    future = good | {'content':'Council approved a new measure.', 'revision_at':'2025-03-01T00:00:00Z'}
    catalog = planning_sources([good,future], 'Council', '2025-02-01')
    assert len(catalog) == 1
    assert catalog['S1']['text'] == good['content']
    assert catalog['S1']['snapshot_sha256'] == hashlib.sha256(good['content'].encode()).hexdigest()


def test_semantic_owner_audit_detects_a_parent_dossier_written_as_its_person(monkeypatch):
    import app.knowledge.decision_seats as seats
    plan, _, _ = fixture()
    figure = seat_figures(plan,[])[0]
    text = 'Atlas personally controls this decision, not Boreal Council.'
    class Model:
        def __init__(self, *args):pass
        async def json_call(self, **kwargs):
            return OwnershipReview(issues=[OwnershipIssue(owner_name='Boreal Council',
                field='decision_scope',quote=text,problem='The scope belongs to another actor.')])
    monkeypatch.setattr(seats,'LLMClient',Model)
    issues, verdict = asyncio.run(review_dossier_owners([{'source_figure':'Boreal Council',
        'decision_core':{'decision_scope':text}}],[figure],api_key='fixture',model='fixture',base_url=None))
    assert len(issues) == 1 and not verdict['approved']


def test_fabricated_owner_audit_quote_cannot_change_an_actor(monkeypatch):
    import app.knowledge.decision_seats as seats
    plan, _, _ = fixture();figure=seat_figures(plan,[])[0]
    class Model:
        def __init__(self,*args):pass
        async def json_call(self,**kwargs):
            return OwnershipReview(issues=[OwnershipIssue(owner_name='Boreal Council',field='decision_scope',
                quote='A quote that was never in the profile.',problem='Invented finding.')])
    monkeypatch.setattr(seats,'LLMClient',Model)
    with pytest.raises(ValueError,match='not in'):
        asyncio.run(review_dossier_owners([{'source_figure':'Boreal Council','decision_core':{'decision_scope':'Its own role.'}}],
            [figure],api_key='fixture',model='fixture',base_url=None))


def test_recent_counterpart_conditions_survive_repetitive_question_matches():
    from app.benchmark.runner import frozen_sources, forecast_inputs
    from app.benchmark.dynamics import contract_state
    filler = '\n\n'.join(f'Atlas visits the plant and reviews the visit programme number {i}.' for i in range(30))
    conditions = 'In July 2025, Boreal Council stated conditions for talks with the visiting party.'
    current = 'In June 2026, Boreal Council announced its current position on cooperation.'
    text = filler + '\n\n== Current cooperation ==\n' + conditions + '\n\n' + current
    case = {'question':'Will Atlas visit Boreal Council in 2026?', 'rules':'Yes if the visit occurs.',
        'as_of':'2026-11-01T00:00:00Z', 'outcome_deadline':'2027-01-01T00:00:00Z',
        'evidence':[{'title':'Atlas–Boreal Council relations','url':'https://example.test/relations',
                     'text':text,'revision_id':1,'revision_at':'2026-10-31T00:00:00Z','is_seed':True}]}
    sources = frozen_sources(case,'expanded')
    selected = '\n'.join(row['content'] for row in sources)
    assert conditions in selected and current in selected
    assert forecast_inputs(case,sources)['evidence_packet'] == contract_state(case,sources)['evidence_packet']


def test_batch_ownership_repair_regenerates_only_the_inconsistent_dossier(tmp_path, monkeypatch):
    from app.layers.pipeline import profile_service
    import app.knowledge.decision_seats as seats
    plan, _, _ = fixture()
    first = seat_figures(plan,[])[0]
    second = deepcopy(first)
    second.update(id='atlas', canonical_name='Atlas', figure='Atlas')
    second['decision_seat'].update(owner_name='Atlas',seat_id='seat_atlas',decision_ids=['propose'])
    generated = []
    async def dimensions(**kw):return []
    async def provider(**kw):
        p=kw['profile'];generated.append((p['canonical_name'], bool(p.get('_ownership_repair'))))
        return {'agent_id':p['id'],'source_figure':p['canonical_name'],'entity_type':'organization',
                'decision_core':DecisionCore(decision_scope='Its own decision.').model_dump()}
    audits=[]
    async def audit(profiles,figures,**kw):
        audits.append(True)
        issues = [{'owner_name':'Atlas','field':'decision_scope','quote':'Its own decision.',
                   'problem':'Fixture owner mismatch.'}] if len(audits)==1 else []
        return issues,{'approved':not issues,'issues':issues}
    monkeypatch.setattr(profile_service,'_generate_power_dimensions',dimensions)
    monkeypatch.setattr(profile_service,'generate_profile',provider)
    monkeypatch.setattr(seats,'review_dossier_owners',audit)
    profiles,_=asyncio.run(profile_service.generate_profiles(api_key='fixture',model='fixture',
        event_title='Talks',figures=[first,second],timeline=[],checkpoint_dir=tmp_path))
    assert len(profiles)==2
    assert generated==[('Boreal Council',False),('Atlas',False),('Atlas',True)]
    assert len(audits)==2


def test_planner_and_dossier_citations_share_one_admitted_namespace(monkeypatch):
    from app.layers.pipeline import profile_service
    from app.knowledge import profile_evidence
    plan,_,_=fixture();profile=seat_figures(plan,[])[0]
    reference=profile['decision_seat']['basic_actions'][0]['source_refs'][0]
    calls=[]
    async def provider(*args,**kwargs):
        calls.append(args[2])
        return {'decision_core':{'decision_scope':'Boreal Council as the counterpart.',
            'means':[{'id':'respond','description':'Communicate conditions for talks.',
                      'holder':'Boreal Council','claim_scope':'conditional','conditions':'No agreement is implied.',
                      'source_refs':[{'evidence_id':reference['evidence_id']}]}]}}
    monkeypatch.setattr(profile_evidence,'source_catalog',lambda *args,**kwargs:[])
    monkeypatch.setattr(profile_service,'_call_json_with_retries',provider)
    result=asyncio.run(profile_service._generate_decision_profile(api_key='fixture',model='fixture',
        event_title='Talks',profile=profile,cutoff='2025-07-04'))
    assert not result['validation_diagnostics']
    assert result['decision_core']['means'][0]['id']=='respond'
    assert result['decision_core']['means'][0]['source_refs'][0]['quote']==reference['quote']
    assert '"REGISTERED_DECISION_SEAT"' in calls[0]
