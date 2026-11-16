"""Offline tests for timeline, profile, and simulation invariants."""
import asyncio
import json

import pytest

from app.domain import TimelineEvent, WorldSnapshot
from app.domain import ActionProposal
from app.layers.simulation.executor import annotate_executors
from app.domain.behavior import BehavioralModel, BehaviorStatement
from app.domain.ensemble import EnsembleRequest
from app.domain.protocol import ScenarioProtocol
from app.knowledge.brief import EvidenceBrief, BriefExcerpt, render_brief
from app.knowledge.evidence import validate_event
from app.knowledge.sentences import complete_excerpt, sentence_records
from app.simulation.activation import choose_active_agents
from app.simulation.bdi_updates import process_agent_information
from app.simulation.context import context_preflight, information_context
from app.simulation.ensemble import ExperimentRuntime, prepare_experiment
from app.simulation.planning import plan_runtime
from app.simulation.protocol import can_advance_quiet, public_digest
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository
from tests.test_agent_protocol import agent


def date_check(text, quote=None, **extra):
    return validate_event({'description':'Event', 'timestamp':'2025-06',
                           'evidence_text':quote or text, **extra},
                          {'content':text, 'path':'source/Lead', 'source_title':'Case'})


def test_year_only_event_cannot_borrow_neighboring_month_even_inside_same_sentence():
    quote = 'the crisis sparked by the earlier event, which began in 2023'
    value, _ = date_check('In June 2025, a war followed ' + quote + '.', quote,
                         date_original='2023')
    assert value is None
    quote = 'The event began in 2023.'
    assert date_check(quote + ' In June 2025, a different event began.', quote)[0] is None


@pytest.mark.parametrize('tail', ['for the first time since 2005', 'after the earlier crisis in 2005'])
def test_explicit_occurrence_survives_comparative_dates(tail):
    text = f'On 12 June 2025, the agency made its finding {tail}.'
    value, _ = date_check(text, timestamp='2025-06-12', date_original='12 June 2025')
    assert value['timestamp'] == '2025-06-12'


@pytest.mark.parametrize('text', ['Before March 2024, the factory closed.',
                                 'By April 2026, 138000 sites had been damaged.'])
def test_bounds_are_not_point_events(text):
    assert date_check(text)[0] is None


def test_date_label_of_same_clause_can_complete_short_evidence():
    text = 'June 2025: The agency made a finding.'
    assert date_check(text, 'The agency made a finding.')[0]['timestamp'] == '2025-06'


def test_complete_sentence_budget_never_keeps_an_incomplete_tail():
    source = 'Dr. Chen met the U.S. delegation. Talks resumed the following week. An unfinished wor'
    records = sentence_records(source)
    assert [r['quote'] for r in records] == [
        'Dr. Chen met the U.S. delegation.', 'Talks resumed the following week.']
    first = records[0]['quote']
    assert complete_excerpt(source, len(first)+10) == first
    assert complete_excerpt('x'*4000+'.', 3000) == ''


def test_brief_restores_a_partial_quote_only_from_complete_source_text():
    first = 'The port authority announced a new coordination process.'
    full = first + ' The group imposed its own administrative review process.'
    clipped = full[:-10]
    brief = EvidenceBrief(excerpts=[BriefExcerpt(source_index=1, quote='The group imposed its own')])
    output = render_brief(brief, [{'title':'Source', 'summary':clipped, 'content':full}])
    assert full.split('. ')[1] in output
    output = render_brief(brief, [{'title':'Source', 'summary':clipped}])
    assert first in output and 'The group imposed' not in output
    selected = sentence_records(full)[1]
    output = render_brief(EvidenceBrief(excerpts=[BriefExcerpt(source_index=1, sentence_ids=[selected['id']])]),
                          [{'title':'Source','content':full}])
    assert selected['quote'] in output and first not in output


def test_public_role_card_reference_resolves_to_supplied_evidence_not_other_private_bdi():
    a,b = agent('a','Team A','A_PRIVATE'),agent('b','Team B','B_PRIVATE')
    b.behavioral_model = BehavioralModel(contextual_roles=[BehaviorStatement(
        description='A documented public role', timeline_refs=['h0'])])
    history=[TimelineEvent(timeline_id=f'h{i}',timestamp='2025-06',description=f'Public event {i}',
                           evidence_text=f'Exact public source {i}',confidence=1,importance=1) for i in range(20)]
    s=WorldSnapshot(event_title='Case',agents=[a,b],timeline=history)
    bundle=information_context(s,a)
    assert 'h0' in bundle['allowed_timeline_refs']
    assert any(e['timeline_id']=='h0' and e['evidence_text']=='Exact public source 0' for e in bundle['public_state']['timeline'])
    assert 'A_PRIVATE' in json.dumps(bundle) and 'B_PRIVATE' not in json.dumps(bundle)
    class Client:
        async def json_call(self, **kw):
            who='a' if '_agent_a_' in kw['call_name'] else 'b'
            return kw['schema'](agent_id=who,round_id=1,snapshot_version=0,assessment='Evidence reviewed',
                bdi_patch={'beliefs':{'situation_views':[{'topic':'public','assessment':'Supported',
                            'confidence':.6,'timeline_refs':['h0']}]}})
    after=asyncio.run(process_agent_information(s,Client(),'test'))
    assert not after.validation_issues
    assert after.agents[0].beliefs.situation_views[0].timeline_refs == ['h0']


def low_score_client(step, version):
    class Client:
        async def json_call(self, **kw):
            if kw['schema'].__name__=='MasterActivationBatch':
                return kw['schema'](round_id=step,snapshot_version=version,reviews=[{
                    'agent_id':'a','master_score':.1,'decision_point_now':False,'inaction_consistency':'stable',
                    'feasible_action_exists':False,'decision_window':'closed','cross_agent_dependency':False,'reasons':['wait']}])
            return kw['schema'](agent_id='a',round_id=step,snapshot_version=version,self_score=.1,
                                decision_required=False,feasible_action_exists=False,preferred_mode='wait',reasons=['wait'])
    return Client()


def test_initial_and_periodic_deliberation_do_not_depend_on_high_activation_scores():
    s=WorldSnapshot(event_title='Case',agents=[agent('a','Team A')])
    s,active,decisions=asyncio.run(choose_active_agents(s,[],low_score_client(1,0),'test'))
    assert len(active)==1 and 'initial_deliberation' in decisions[0].reasons
    s.round_id=1;s.snapshot_version=1
    s,active,_=asyncio.run(choose_active_agents(s,[],low_score_client(2,1),'test'))
    assert not active
    s.round_id=3;s.snapshot_version=3
    record={'quiet_candidate':True,'post_round_public_digest':public_digest(s)}
    assert not can_advance_quiet(s,record,ScenarioProtocol(idle_policy='advance_if_unchanged'))
    s,active,decisions=asyncio.run(choose_active_agents(s,[],low_score_client(4,3),'test'))
    assert len(active)==1 and 'periodic_deliberation' in decisions[0].reasons


def test_context_preview_is_free_and_does_not_create_experiment(tmp_path):
    request=EnsembleRequest(event_title='Case',agents=[agent('a','Team A').model_dump()],
                            context_max_chars=8000,initial_private_information={'a':['private '*1800]})
    store=EnsembleStore(tmp_path/'runs')
    measured=prepare_experiment(request,events=EventRepository(tmp_path/'events'),store=store,preview_only=True)
    assert not measured['fits'] and measured['model_calls']==0
    assert measured['recommended_chars']>measured['minimum_chars']
    assert not list(store.root.glob('*/config.json'))
    snapshot=WorldSnapshot(event_title='Case',agents=[agent('a','Team A')])
    request.context_max_chars=48000;request.planning_mode='rules'
    plan=asyncio.run(plan_runtime(request,snapshot,None,'test'))
    assert context_preflight(snapshot)['minimum_chars'] < plan['settings']['context_max_chars'] <= 48000


def test_large_context_passes_preflight_planning_and_runtime_without_old_48000_cap(tmp_path):
    request = EnsembleRequest(event_title='Case', agents=[agent('a', 'Team A').model_dump()],
                              planning_mode='rules', initial_private_information={'a': ['private ' * 9000]})
    store = EnsembleStore(tmp_path / 'runs')
    events = EventRepository(tmp_path / 'events')
    measured = prepare_experiment(request, events=events, store=store, preview_only=True)
    assert measured['fits'] and measured['model_calls'] == 0
    assert 48000 < measured['minimum_chars'] < measured['recommended_chars'] <= request.context_max_chars
    run_id, request = prepare_experiment(request, events=events, store=store)
    baseline = WorldSnapshot.model_validate(store.load(run_id, 'initial_snapshot'))
    plan = asyncio.run(plan_runtime(request, baseline, None, run_id))
    assert plan['settings']['context_max_chars'] >= measured['recommended_chars']

    class Client:
        calls = 0

        async def json_call(self, **kwargs):
            self.calls += 1
            return {'ok': True}

    client = Client()
    runtime = ExperimentRuntime(client, request, store, run_id)
    runtime.context_max_chars = plan['settings']['context_max_chars']
    assert asyncio.run(runtime.json_call(system='Fixture', prompt='x' * 100000)) == {'ok': True}
    runtime.context_max_chars = 64000
    with pytest.raises(ValueError, match='character budget'):
        asyncio.run(runtime.json_call(system='Fixture', prompt='x' * 100000))
    assert client.calls == 1


def test_context_estimate_does_not_hide_demand_above_supported_maximum(tmp_path):
    request = EnsembleRequest(event_title='Case', agents=[agent('a', 'Team A').model_dump()],
                              context_max_chars=512000,
                              initial_private_information={'a': ['private ' * 70000]})
    measured = prepare_experiment(request, events=EventRepository(tmp_path / 'events'),
                                  store=EnsembleStore(tmp_path / 'runs'), preview_only=True)
    assert not measured['fits'] and measured['minimum_chars'] > 512000
    assert measured['recommended_chars'] > measured['minimum_chars']


def test_new_hard_trigger_is_not_hidden_by_unrelated_events_or_old_history():
    a=agent('a','Team A')
    a.activation.last_deliberation_round=1
    a.activation.last_review_round=1
    hard=TimelineEvent(timeline_id='new_deadline',timestamp='2026-01',description='A response is due',
        actors=['a'],event_type='required_response',origin='simulated',round_id=1,confidence=1,importance=1)
    others=[TimelineEvent(timeline_id=f'o{i}',timestamp='2026-01',description='Other public event',
        actors=['b'],origin='simulated',round_id=1,confidence=1,importance=.5) for i in range(10)]
    s=WorldSnapshot(event_title='Case',agents=[a],timeline=[hard,*others],round_id=1,snapshot_version=1)
    _,active,decisions=asyncio.run(choose_active_agents(s,[],low_score_client(2,1),'test'))
    assert len(active)==1 and 'hard_event_trigger' in decisions[0].reasons


def test_wait_is_not_rewritten_as_an_executed_action_or_sent_for_executor_annotation():
    a=agent('a','Team A')
    a.public_state={'subordinates':[{'name':'Member'}]}
    proposal=ActionProposal(agent_id='a',agent_name='Team A',round_id=1,snapshot_version=0,
        action_id='action_1_a',decision='wait',action_type='no_action',action='Wait for a response',
        reasoning='A decision has been made to wait',confidence=.6)
    class Client:
        async def text_call(self, **kw):
            raise AssertionError('Waiting needs no executor annotation call')
    output=asyncio.run(annotate_executors(WorldSnapshot(event_title='Case',agents=[a]),[proposal],Client(),'test'))
    assert output == [proposal]
