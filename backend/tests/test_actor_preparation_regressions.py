"""Principal identity, shared evidence and honest preparation status."""
import asyncio
import json

from app.domain.decision_core import DecisionCore
from app.knowledge.decision_seats import (
    VERSION, SeatPlan, bind_plan, seat_figures, should_replan_required_seat,
)
from app.knowledge.profile_evidence import background_catalog, source_catalog
from app.knowledge.public_evidence import public_sources
from app.knowledge.retrieval import _named_in
from app.layers.pipeline import profile_service
from app.layers.pipeline.seams import to_agent_profiles
from app.storage.extraction_status import ExtractionStatusRepository
from app.storage.json_store import EventRepository


def source(text, **extra):
    return {'url':'https://en.wikipedia.org/wiki/Negotiations', 'title':'Negotiations',
            'revision_id':'17', 'revision_at':'2025-07-03T00:00:00Z', 'content':text, **extra}


def seat_plan(name, quote, entity_type='organization'):
    catalog = {'E1':source(quote) | {'text':quote}}
    raw = SeatPlan.model_validate({
        'seats':[{'name':name, 'entity_type':entity_type, 'role':'Decide whether to submit a proposal.',
                  'required':True, 'decision_ids':['propose'], 'evidence':[{'source_id':'E1'}]}],
        'requirements':[{'decision_id':'propose', 'description':'Decide whether to submit a proposal.',
                         'essential':True, 'owner':name}],
    })
    return bind_plan(raw, catalog, f'Will {name} submit a proposal?')


def test_punctuation_does_not_allow_a_reporting_service_to_impersonate_its_parent():
    assert not _named_in('U.S. Congress', 'The U.S. Congressional Research Service published a report.')
    assert _named_in('U.S. Congress', 'The U.S. Congress approved the proposal.')
    assert not _named_in('A.C. Board', 'A.C. Boardman commented on the proposal.')
    assert _named_in('中国', '中国宣布进行谈判。')
    bound = seat_plan('United States Congress', 'The U.S. Congressional Research Service published a report.')
    assert not bound['seats']
    assert bound['requirements'][0]['owner'] is None


def test_persons_are_bound_by_their_literal_name_without_requiring_an_attached_title():
    bound = seat_plan('Prime Minister Morgan Hale', 'Morgan Hale proposed a meeting.', 'person')
    assert bound['seats'][0]['owner_name'] == 'Morgan Hale'
    assert bound['requirements'][0]['owner'] == 'Morgan Hale'
    assert bound['seats'][0]['source_refs'][0]['quote'] == 'Morgan Hale proposed a meeting.'
    assert not seat_plan('Prime Minister Morgan Hale', 'A different delegation met in the capital.', 'person')['seats']


def test_registered_country_alias_reuses_historical_entity_and_keeps_its_type():
    plan = seat_plan('Federated Republic', 'FR submitted its proposal.', 'country')
    existing = [{'id':'fr', 'canonical_name':'Federated Republic', 'aliases':['FR'],
                 'entity_type':'country', 'evidence_count':9, 'history':[{'timeline_id':'h'}]}]
    rows = seat_figures(plan, existing)
    assert len(rows) == 1 and rows[0]['id'] == 'fr'
    assert rows[0]['canonical_name'] == 'Federated Republic' and rows[0]['history'][0]['timeline_id'] == 'h'
    assert 'FR' in rows[0]['aliases']
    assert rows[0]['decision_seat']['owner_name'] == 'Federated Republic'
    assert plan['requirements'][0]['owner'] == 'Federated Republic'


def test_bullet_evidence_and_pronoun_context_preserve_exact_offsets_and_separation():
    text = ('== Talks ==\n* Renewal Council announced that talks were necessary. It sought a compromise.\n'
            '* Another party rejected the invitation. It refused further meetings.\n\n'
            '{|\n! Actor !! Resources\n|-\n| Renewal Council || 20\n|}')
    spans = source_catalog({'canonical_name':'Renewal Council'}, cutoff='2025-07-04', sources=[source(text)])
    assert spans and any('It sought a compromise.' in row['text'] for row in spans)
    assert all(text[row['span_start']:row['span_end']] == row['text'] for row in spans)
    assert all('Another party' not in row['text'] and '20' not in row['text'] for row in spans)
    assert not source_catalog({'canonical_name':'Renewal Council'}, cutoff='2025-07-02', sources=[source(text)])


def test_shared_background_is_citeable_without_implicit_actor_ownership():
    text = ('== Talks ==\n* Morgan Hale, Renewal Council chair, said an agreement was possible.\n'
            '* Atlas officials opposed the proposal.')
    packet = public_sources([source(text)])
    rows = background_catalog(packet, cutoff='2025-07-04')
    assert len(rows) == 2
    assert all(text[row['span_start']:row['span_end']] == row['text'] for row in rows)
    assert not background_catalog([source(text)], cutoff='2025-07-04')
    assert not background_catalog(packet, cutoff='2025-07-02')


def test_missing_bdi_gets_one_bounded_review_and_projects_to_the_displayed_agent(monkeypatch):
    quote = 'Renewal Council submitted a proposal to continue talks.'
    plan = seat_plan('Renewal Council', quote)
    profile = seat_figures(plan, [])[0]
    calls = []
    text = quote + '\n\n* Its chair Morgan Hale said an agreement was possible and sought continued talks.'

    async def provider(_key, _model, prompt, **kwargs):
        data = json.loads(prompt.rsplit('\n', 1)[-1])
        calls.append(data)
        if len(calls) == 1:
            return {'decision_core': {'means':[{'id':'propose', 'description':'Submit a proposal for talks.',
                'holder':'Renewal Council', 'claim_scope':'conditional', 'source_refs':[{'evidence_id':'S01'}]}]}}
        assert kwargs['attempts'] == 1
        number = next(key for key, value in data['evidence']['source_spans'].items() if 'agreement was possible' in value)
        return {'decision_core': {
            'initial_judgments':[{'description':'An agreement remains possible.', 'perspective':'actor',
                'provenance':'inferred', 'claim_scope':'conditional', 'source_refs':[{'evidence_id':number}]}],
            'goals':[{'description':'Keep negotiations open.', 'provenance':'inferred', 'claim_scope':'conditional',
                      'source_refs':[{'evidence_id':number}]}],
            'initial_intentions':[],
        }}

    monkeypatch.setattr(profile_service, '_call_json_with_retries', provider)
    result = asyncio.run(profile_service.generate_profile(api_key='fixture', model='fixture', event_title='Talks',
        profile=profile, cutoff='2025-07-04', source_records=public_sources([source(text)])))
    assert len(calls) == 2 and not result['validation_diagnostics']
    assert result['evidence_summary']['shared_pages'] == 1
    assert result['decision_core']['means'][0]['id'] == 'propose'
    agents = to_agent_profiles([result], event_id='fixture', event_title='Talks')
    assert agents[0].beliefs.situation_views[0].assessment == 'An agreement remains possible.'
    assert agents[0].desires[0].description == 'Keep negotiations open.'
    assert agents[0].current_plan is None  # An uncommitted objective is not an intention.


def test_unattributable_bdi_stays_empty_after_one_review(monkeypatch):
    quote = 'Renewal Council submitted a proposal to continue talks.'
    plan = seat_plan('Renewal Council', quote)
    calls = []

    async def provider(*args, **kwargs):
        calls.append(1)
        return {'decision_core':{}}

    monkeypatch.setattr(profile_service, '_call_json_with_retries', provider)
    result = asyncio.run(profile_service.generate_profile(api_key='fixture', model='fixture', event_title='Talks',
        profile=seat_figures(plan, [])[0], cutoff='2025-07-04', source_records=public_sources([source(quote)])))
    assert len(calls) == 2
    core = DecisionCore.model_validate(result['decision_core'])
    assert not core.initial_judgments and not core.goals and not core.initial_intentions


def test_updated_policy_replans_stale_rosters_even_without_a_required_person():
    assert should_replan_required_seat({'version':'question-decision-seats-7', 'status':'ready'}, '')
    assert not should_replan_required_seat({'version':VERSION, 'status':'ready'}, '')


def test_ineligible_saved_roster_is_partial_without_rewriting_its_history(tmp_path):
    events = EventRepository(tmp_path)
    event_id = events.create('Talks', [], None)['event_id']
    plan = seat_plan('Renewal Council', 'Renewal Council submitted a proposal to continue talks.')
    events.save_artifact(event_id, 'decision_seat_plan', plan)
    events.save_artifact(event_id, 'agents', [{'name':'Renewal Council', 'decision_core':{
        'simulation_eligible':False, 'seat':plan['seats'][0] | {'owner_name':'Renewal Council'}}}])
    events.save_artifact(event_id, 'extraction_status', {'status':'completed', 'progress':100})
    events.save_artifact(event_id, 'timeline_cache', {'timeline':[{'timeline_id':'h'}]})
    status = ExtractionStatusRepository(events).get(event_id)
    assert status['status'] == 'partial' and status['resume_available']
    assert events.load_artifact(event_id, 'extraction_status', {})['status'] == 'completed'


def test_profile_generation_keeps_the_confirmed_cutoff_and_upgrades_legacy_selected_packet(tmp_path, monkeypatch):
    events = EventRepository(tmp_path)
    event_id = events.create('Talks', [], None)['event_id']
    events.save_sources(event_id, [source('Renewal Council submitted its proposal.', retrieved_at='2026-01-01T00:00:00Z')])
    events.save_artifact(event_id, 'forecast_spec', {'as_of':'2025-07-04T12:00:00Z'})
    calls = []

    async def dimensions(**kwargs):
        return []

    async def generate(**kwargs):
        calls.append(kwargs)
        return {'agent_id':'renewal', 'source_figure':'Renewal Council'}

    monkeypatch.setattr('app.storage.json_store.EventRepository', lambda: events)
    monkeypatch.setattr(profile_service, '_generate_power_dimensions', dimensions)
    monkeypatch.setattr(profile_service, 'generate_profile', generate)
    asyncio.run(profile_service.generate_profiles(api_key='fixture', model='fixture', event_title='Talks',
        figures=[{'canonical_name':'Renewal Council'}], case_id=event_id))
    assert calls[0]['cutoff'] == '2025-07-04T12:00:00Z'
    assert calls[0]['source_records'][0]['public_packet_version']
    assert 'public_packet_version' not in events.load_sources(event_id)[0]


def test_extraction_does_not_promote_a_limited_profile_result_to_completed(tmp_path, monkeypatch):
    from app.domain import FiguresRequest
    from app.knowledge import extraction

    events = EventRepository(tmp_path)
    event_id = events.create('Talks', [], None)['event_id']
    events.save_artifact(event_id, 'timeline_cache', {'timeline':[{'timeline_id':'h'}]})

    async def prepared(*args):
        return {'run_id':'fixture', 'figures':[{'name':'Renewal Council'}], 'timeline':[{'timeline_id':'h'}],
                'quality':{'profile_status':'limited', 'profile_error':'Opening decision state is incomplete.'}}

    monkeypatch.setattr(extraction, '_extract_figures', prepared)
    monkeypatch.setattr(extraction.KnowledgeStore, 'publish', lambda *args: {})
    asyncio.run(extraction.extract_figures(FiguresRequest(event_id=event_id, event_title='Talks'), events))
    status = ExtractionStatusRepository(events).get(event_id)
    assert status['status'] == 'partial'
    assert status['error'] == 'Opening decision state is incomplete.'
    assert status['resume_available']
