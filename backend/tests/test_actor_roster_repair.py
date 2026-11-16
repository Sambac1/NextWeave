"""Cross-scenario identity proof, decision-unit coverage and partial-plan resume."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

from app.knowledge.actor_identity import discovered_person_aliases
from app.knowledge.decision_seats import (
    VERSION, SeatPlan, bind_plan, prepare_seat_plan, should_replan_required_seat,
)
from app.layers.pipeline.schemas import LayerStageRequest
from app.storage.json_store import EventRepository


def source(text):
    return {'text':text, 'content':text, 'title':'Talks', 'url':'https://example.test/talks',
            'revision_id':'10', 'revision_at':'2025-07-01T00:00:00Z'}


def person_plan():
    return SeatPlan.model_validate({'seats':[{'name':'Rhea Nolan', 'entity_type':'person',
        'role':'Consider submitting a proposal.', 'required':True, 'decision_ids':['propose'],
        'evidence':[{'source_id':'E2'}]}], 'requirements':[{'decision_id':'propose',
        'description':'Consider submitting a proposal.', 'owner':'Rhea Nolan', 'essential':True}]})


def test_separate_identity_and_action_passages_resolve_a_unique_surname():
    catalog = {'E1':source('President Rhea Nolan addressed the delegation.'),
               'E2':source('Nolan said she would consider submitting a proposal.')}
    bound = bind_plan(person_plan(), catalog, 'Will the delegation submit a proposal?')
    seat = bound['seats'][0]
    assert seat['owner_name'] == 'Rhea Nolan'
    assert seat['source_refs'][0]['quote'] == catalog['E2']['text']
    assert seat['identity_source_refs'][0]['quote'] == catalog['E1']['text']
    assert seat['identity_link']['alias'] == 'Nolan'
    assert seat['entity_type'] == 'person' and not seat.get('scoped_identity')


def test_a_shared_surname_cannot_connect_different_people_or_create_authority():
    catalog = {'E1':source('Rhea Nolan addressed the delegation. Julian Nolan led another group.'),
               'E2':source('Nolan said a proposal was possible.')}
    rejected = bind_plan(person_plan(), catalog, 'Will a proposal be made?')
    assert not rejected['seats']
    assert rejected['binding_failures'][0]['reason'] == 'identity_not_bound'
    assert rejected['requirements'][0]['owner'] is None
    catalog['E2'] = source('The delegation said a proposal was possible.')
    assert not bind_plan(person_plan(), catalog, 'Question')['seats']


def test_safe_alias_discovery_does_not_depend_on_a_model_proposing_the_mapping():
    figures = [{'name':name, 'entity_type':'person'} for name in ('Nolan','Rhea Nolan','President Rhea Nolan')]
    aliases = discovered_person_aliases(figures)
    assert aliases['Nolan'] == 'Rhea Nolan'
    assert aliases['President Rhea Nolan'] == 'Rhea Nolan'
    ambiguous = figures + [{'name':'Julian Nolan', 'entity_type':'person'}]
    assert 'Nolan' not in discovered_person_aliases(ambiguous)
    assert not discovered_person_aliases([{'name':name, 'entity_type':'organization'}
                                         for name in ('Atlas Board','Atlas Audit Committee')])


def test_question_level_principal_cannot_be_replaced_by_its_representative():
    text = 'Atlas president Rhea Nolan announced that Atlas would consider a proposal.'
    catalog = {'E2':source(text)}
    raw = person_plan().model_dump()
    raw['question_actors'] = [{'name':'Atlas','entity_type':'country','participation':'decision',
                               'decision_ids':['propose'], 'evidence':[{'source_id':'E2'}]}]
    wrong = bind_plan(SeatPlan.model_validate(raw), catalog, 'Will Atlas submit a proposal?')
    assert not wrong['seats']
    assert wrong['binding_failures'][0]['reason'] == 'question_decision_unit_mismatch'
    raw['seats'][0].update(name='Atlas', entity_type='country', representatives=[
        {'name':'Rhea Nolan','role':'President speaking for Atlas.','evidence':[{'source_id':'E2'}]}])
    raw['requirements'][0]['owner'] = 'Atlas'
    repaired = bind_plan(SeatPlan.model_validate(raw), catalog, 'Will Atlas submit a proposal?')
    assert repaired['status'] == 'ready'
    assert repaired['seats'][0]['owner_name'] == 'Atlas'
    assert repaired['seats'][0]['representatives'][0]['name'] == 'Rhea Nolan'


def test_an_affected_entity_is_not_forced_to_make_decisions():
    catalog = {'E2':source('Rhea Nolan announced aid for the affected Atlas region.')}
    raw = person_plan().model_dump()
    raw['question_actors'] = [{'name':'Atlas','entity_type':'country','participation':'affected',
                              'decision_ids':[], 'evidence':[{'source_id':'E2'}]}]
    bound = bind_plan(SeatPlan.model_validate(raw), catalog, 'Will aid reach Atlas?')
    assert [seat['owner_name'] for seat in bound['seats']] == ['Rhea Nolan']


def test_same_policy_partial_roster_requires_replanning_and_is_not_a_cache_hit(tmp_path, monkeypatch):
    import app.knowledge.decision_seats as module

    events = EventRepository(tmp_path)
    eid = events.create('Will Nolan submit a proposal?', [], None)['event_id']
    events.save_sources(eid, [source('Rhea Nolan considers submitting a proposal.')])
    req = SimpleNamespace(event_id=eid, event_title='Will Nolan submit a proposal?',
        evidence_cutoff='2025-07-02', model='fixture', api_key='fixture', base_url=None,
        language='en-US', max_key_figures=8, required_decision_subject='')
    raw = person_plan()
    raw.seats[0].evidence[0].source_id = 'E1'
    calls = []

    class Runtime:
        def __init__(self, *args):
            pass
        async def json_call(self, **kwargs):
            calls.append(kwargs)
            return raw

    monkeypatch.setattr(module, 'LLMClient', Runtime)
    ready = asyncio.run(prepare_seat_plan(req, events))
    partial = deepcopy(ready)
    partial.update(seats=[], status='limited')
    partial['requirements'][0]['owner'] = None
    events.save_artifact(eid, 'decision_seat_plan', partial)
    assert should_replan_required_seat(partial, '')
    dangling = deepcopy(ready)
    dangling['seats'] = []
    assert should_replan_required_seat(dangling, '')
    repaired = asyncio.run(prepare_seat_plan(req, events))
    assert len(calls) == 2 and repaired['status'] == 'ready'
    assert repaired['requirements'][0]['owner'] == 'Rhea Nolan'
    assert events.load_artifact(eid, 'decision_seat_attempts', [])[0]['proposed_plan']


def test_resume_returns_to_actor_planning_when_a_same_policy_decision_owner_is_missing(tmp_path, monkeypatch):
    from app.layers.pipeline import stage, steps
    import app.knowledge.decision_seats as module

    events = EventRepository(tmp_path)
    eid = events.create('Will a proposal be submitted?', [], None)['event_id']
    partial = {'version':VERSION, 'status':'limited', 'seats':[], 'requirements':[
        {'decision_id':'propose','description':'Submit a proposal.','owner':None,'kind':'decision','essential':True}]}
    events.save_artifact(eid, 'decision_seat_plan', partial)
    events.save_artifact(eid, 'profile_inputs', {'checkpoint_key':'old-checkpoint'})
    calls = []

    async def prepare(*args, **kwargs):
        calls.append('plan')
        return partial

    async def run(**kwargs):
        assert not kwargs['req'].resume_profiles
        assert kwargs['req'].checkpoint_key != 'old-checkpoint'
        calls.append('actors')
        return {'profiles':[], 'quality':{}}

    monkeypatch.setattr(module, 'prepare_seat_plan', prepare)
    monkeypatch.setattr(steps, 'run', run)
    request = LayerStageRequest(event_id=eid, event_title='Will a proposal be submitted?',
                                api_key='fixture', model='fixture', resume_profiles=True)
    result = asyncio.run(stage.run_layer_stage(request=request, timeline=[], events=events))
    assert calls == ['plan','actors']
    assert result['quality']['profile_status'] == 'limited'


def test_expired_commitments_are_not_kept_as_opening_intentions():
    from app.domain.decision_core import validate_core
    commitment = {'description':'Pause the measure until the vote.', 'provenance':'source_statement',
                  'claim_scope':'conditional', 'timeline_refs':['h1'], 'valid_until':'2025-07-01'}
    expired = validate_core({'initial_intentions':[commitment]}, ['h1'], as_of='2025-07-02')
    assert not expired.initial_intentions and expired.review_notes
    active = validate_core({'initial_intentions':[commitment]}, ['h1'], as_of='2025-06-30')
    assert len(active.initial_intentions) == 1


def test_source_deadline_is_checked_when_model_leaves_valid_until_empty():
    from app.domain.decision_core import DecisionCore, DecisionItem, SourceSupport
    from app.knowledge.profile_evidence import ground_opening_intentions
    def core(quote, description):
        return DecisionCore(as_of='2025-07-11', initial_intentions=[DecisionItem(description=description,
            provenance='source_statement', claim_scope='conditional', source_refs=[SourceSupport(
                evidence_id='source', title='2025 negotiations', quote=quote)])])
    expired = core('On 8 June, the board said it would pause measures until 3 July.', 'Pause measures until 3 July.')
    assert not ground_opening_intentions(expired).initial_intentions
    assert expired.initial_intentions[0].valid_until == ''  # The projection does not mutate its input.
    active = core('On 8 June, the board said it would pause measures until 30 July.', 'Pause measures until 30 July.')
    assert ground_opening_intentions(active).initial_intentions[0].valid_until == '2025-07-30'
    next_year = core('On 8 June, the board said it would pause measures until January.', 'Pause measures until January.')
    assert ground_opening_intentions(next_year).initial_intentions[0].valid_until == ''
    earliest = core('On 8 June, the board committed to vote not before 30 July.', 'Vote not before 30 July.')
    assert ground_opening_intentions(earliest).initial_intentions[0].valid_until == ''
    exclusive = core('On 8 June, the board committed to finish before 11 July.', 'Finish before 11 July.')
    assert not ground_opening_intentions(exclusive).initial_intentions
