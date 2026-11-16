"""Production failures and indicator semantics, with no provider requests."""
import asyncio
import json
from copy import deepcopy

import pytest

from app.domain import GameMasterSettlement, WorldSnapshot, TimelineEvent
from app.domain.agency import resolve_entity_type
from app.domain.ensemble import EventQuestion, OutcomeCondition, OutcomePlan, QuestionAnswer
from app.functional_agents.simulation import GameMasterAgent
from app.llm.budget import BudgetExceeded
from app.simulation.activation import choose_active_agents
from app.simulation.clock import resolve_start, step_clock, align_events
from app.simulation.ensemble import _plan, classify, result_view, wilson, prepare_experiment, run_experiment
from app.simulation.event_metrics import condition_value, event_metrics, observe, resolve_answer
from app.storage.compact_trace import CompactTrace
from app.storage.ensemble_store import EnsembleStore
from app.storage.json_store import EventRepository
from tests.test_agent_protocol import agent
from tests.test_ensemble import request


def record(answer='yes', **answer_fields):
    return {'sample_id': 1, 'status': 'completed',
            'snapshot': {'agents': [{'agent_id': 'a'}], 'world_state': {}},
            'rounds': [{'step': 1, 'timeline': [{'timeline_id': 'e1', 'description': 'A meeting occurred.'}]}],
            'classification': {'questions': [{'question_id': 'meeting', 'answer': answer,
                                               'event_refs': ['e1'], **answer_fields}]}}


def question(**fields):
    return EventQuestion(id='meeting', label='Meeting', criterion='A meeting occurs', **fields)


def test_semantic_settlement_failure_is_repaired_without_new_decisions(tmp_path):
    calls = []
    class Client:
        recorder = CompactTrace(tmp_path)
        async def json_call(self, **kw):
            calls.append(kw)
            return GameMasterSettlement(state_summary='After', next_world_state={}, analysis='Settled',
                agent_knowledge_updates=[] if len(calls) == 1 else [{'agent_id': 'a', 'information': []}])
    baseline = WorldSnapshot(event_title='Case', agents=[agent('a', 'Team')])
    before = baseline.model_dump()
    result = asyncio.run(GameMasterAgent(Client()).settle(baseline, [], [], 'test'))
    assert len(calls) == 2 and result.agent_knowledge_updates[0].agent_id == 'a'
    assert "missing=['a']" in calls[1]['prompt'] and 'SEMANTIC VALIDATION ERROR' in calls[1]['prompt']
    assert baseline.model_dump() == before
    trace = [json.loads(line) for line in (tmp_path / 'model_calls.jsonl').read_text().splitlines()]
    assert trace[0]['diagnostic']['known_agent_ids'] == ['a']
    assert trace[0]['diagnostic']['returned_agent_ids'] == []


@pytest.mark.parametrize('budget_failure', [False, True])
def test_semantic_repair_is_bounded_and_respects_budget(budget_failure):
    class Client:
        calls = 0
        async def json_call(self, **kw):
            self.calls += 1
            if budget_failure and self.calls == 2:
                raise BudgetExceeded('Fixture budget')
            return GameMasterSettlement(state_summary='After', next_world_state={}, analysis='No updates')
    client = Client()
    with pytest.raises(BudgetExceeded if budget_failure else ValueError):
        asyncio.run(GameMasterAgent(client).settle(WorldSnapshot(event_title='Case', agents=[agent('a', 'Team')]), [], [], 'test'))
    assert client.calls == 2


def test_absence_is_unknown_and_negative_needs_full_horizon_evidence():
    q = question()
    r = record('no', reason='Not recorded')
    assert resolve_answer(q, r['classification']['questions'][0], r).answer == 'unclear'
    r['rounds'][0]['timeline'][0]['description'] = 'The complete period ended with no meeting.'
    raw = QuestionAnswer(question_id=q.id, answer='no', event_refs=['e1'],
                         evidence_kind='explicit_negative', evidence_quote='no meeting')
    assert resolve_answer(q, raw, r).answer == 'unclear'
    raw.evidence_scope = 'whole_horizon'
    assert resolve_answer(q, raw, r).answer == 'no'
    r['rounds'].append({'step': 2, 'timeline': []})
    assert resolve_answer(q, raw, r).answer == 'unclear'
    r['rounds'].pop()
    raw.evidence_quote = 'Invented sentence'
    assert resolve_answer(q, raw, r).answer == 'unclear'


def test_terminal_question_cannot_use_an_earlier_interval():
    r = record()
    r['rounds'].append({'step': 2, 'timeline': []})
    assert resolve_answer(question(scope='terminal'), r['classification']['questions'][0], r).answer == 'unclear'


def test_indicator_counts_once_and_retains_unknown_and_unfinished_denominator():
    yes = record(event_refs=['e1', 'e1'])
    no = record('no', evidence_kind='explicit_negative', evidence_scope='whole_horizon', evidence_quote='No meeting')
    no['sample_id'] = 2
    no['rounds'][0]['timeline'][0]['description'] = 'No meeting took place during the complete period.'
    unknown = record('no', reason='No recorded meeting'); unknown['sample_id'] = 3
    unfinished = {'sample_id': 4, 'status': 'failed', 'error': 'Fixture failure'}
    result = event_metrics(question(), [yes, no, unknown, unfinished], 4, wilson)
    assert (result['count'], result['no_count'], result['unclear_count'], result['unfinished_count']) == (1, 1, 1, 1)
    assert result['frequency'] is None and result['sampling_interval_95'] is None
    assert result['interpretation_bounds'] == [.25, .75]
    assert result['assessable_samples'] == 2


def test_zero_is_reported_only_when_every_trajectory_is_measurable():
    r = record('no', evidence_kind='explicit_negative', evidence_scope='whole_horizon', evidence_quote='No meeting')
    r['rounds'][0]['timeline'][0]['description'] = 'No meeting during the whole horizon.'
    result = event_metrics(question(), [r], 1, wilson)
    assert result['frequency'] == 0 and result['sampling_interval_95'] is not None
    assert event_metrics(question(), [], 2, wilson)['frequency'] is None


def test_missing_state_and_wrong_boolean_type_never_become_false():
    conditions = [OutcomeCondition(path='deal.signed', operator='eq', value=True)]
    assert condition_value({}, conditions) is None
    assert condition_value({'deal': {'signed': 0}}, conditions) is None
    q = question(conditions=conditions, scope='terminal')
    r = record('no')
    assert resolve_answer(q, r['classification']['questions'][0], r).answer == 'unclear'
    r['snapshot']['world_state'] = {'deal': {'signed': False}}
    assert resolve_answer(q, r['classification']['questions'][0], r).answer == 'no'


def test_duration_requires_explicit_continuity_and_real_calendar_intervals():
    q = question(conditions=[OutcomeCondition(path='quiet', operator='eq', value=True)], required_duration_days=14)
    p = OutcomePlan(categories=request().outcome_definitions, event_questions=[q])
    r = record()
    r['rounds'].append({'step': 2, 'timeline': []})
    r['measurement_history'] = [observe({'quiet': True}, p, {}, step=0),
        observe({'quiet': True}, p, step_clock('2026-10-31', 1, 'week', 2), step=1),
        observe({'quiet': True}, p, step_clock('2026-10-31', 2, 'week', 2), step=2)]
    raw = r['classification']['questions'][0]
    assert resolve_answer(q, raw, r).answer == 'unclear'
    q.assume_continuity = True
    assert resolve_answer(q, raw, r).answer == 'yes'
    r['measurement_history'][1]['questions'][q.id] = None
    assert resolve_answer(q, raw, r).answer == 'unclear'


def test_changed_endpoint_does_not_prove_short_duration():
    q = question(conditions=[OutcomeCondition(path='quiet', operator='eq', value=True)],
                 required_duration_days=10, assume_continuity=True)
    p = OutcomePlan(categories=request().outcome_definitions, event_questions=[q])
    r = record(); r['rounds'].append({'step': 2, 'timeline': []})
    r['measurement_history'] = [observe({'quiet': True}, p, {}, step=0),
        observe({'quiet': True}, p, step_clock('2026-10-31', 1, 'week', 2), step=1),
        observe({'quiet': False}, p, step_clock('2026-10-31', 2, 'week', 2), step=2)]
    assert resolve_answer(q, r['classification']['questions'][0], r).answer == 'unclear'


def test_false_boundaries_are_not_proof_of_no_event_between_them():
    q = question(conditions=[OutcomeCondition(path='occurred', operator='eq', value=True)])
    p = OutcomePlan(categories=request().outcome_definitions, event_questions=[q])
    r = record('no')
    r['measurement_history'] = [observe({'occurred': False}, p, {}, step=0), observe({'occurred': False}, p, {}, step=1)]
    assert resolve_answer(q, r['classification']['questions'][0], r).answer == 'unclear'
    q.condition_coverage = 'cumulative_horizon'
    assert resolve_answer(q, r['classification']['questions'][0], r).answer == 'no'


def test_unmodeled_actors_and_untracked_text_duration_are_unknown():
    r = record()
    for q in (question(required_actor_ids=['missing']), question(coverage_status='limited'),
              EventQuestion(id='meeting', label='Quiet', criterion='连续72小时没有袭击')):
        assert resolve_answer(q, r['classification']['questions'][0], r).answer == 'unclear'


def test_clock_crosses_months_and_leap_days_without_inventing_occurrence_day():
    assert step_clock('2026-10-31', 1, 'week', 3)['interval_end'] == '2026-11-07'
    assert step_clock('2028-01-31', 1, 'month', 2)['interval_end'] == '2028-02-29'
    assert step_clock('2028-01-31', 2, 'month', 2)['interval_end'] == '2028-03-31'
    e = TimelineEvent(timeline_id='sim_1_a', timestamp='2026-10', description='Event', confidence=1, importance=1)
    align_events([e], step_clock('2026-10-31', 1, 'week', 3))
    assert e.timestamp == '2026-11' and e.date_precision == 'interval'
    assert e.date_range == {'start': '2026-10-31', 'end': '2026-11-07'}
    e.timestamp = '2026-10-15'
    with pytest.raises(ValueError, match='falls outside the interval'):
        align_events([e], step_clock('2026-10-31', 1, 'week', 3))


def test_start_date_respects_month_precision_and_caller_boundary():
    req = request(start_date='2026-10-28')
    with pytest.raises(ValueError, match='latest historical date'):
        resolve_start(req, [{'timestamp': '2026-10'}])
    req.start_date = '2026-11-01'
    assert resolve_start(req, [{'timestamp': '2026-10'}]) == ('2026-11-01', 'explicit_caller_date')


def test_overlapping_sample_matches_are_not_forced_into_one_outcome():
    class Client:
        async def json_call(self, **kw):
            return kw['schema'](samples=[{'sample_id': 1, 'category_id': 'left',
                'matched_category_ids': ['left', 'right'], 'reason': 'Both apply', 'questions': []}])
    r = record(); r['snapshot']['state_summary'] = 'Left and right'
    p = OutcomePlan(categories=request().outcome_definitions)
    result = asyncio.run(classify([r], p, Client(), 'fixture'))
    assert result[0].category_id == 'other'


def test_automatic_partition_audit_rejects_overlaps_before_sampling():
    class Client:
        calls = 0
        async def json_call(self, **kw):
            self.calls += 1
            if kw['schema'].__name__ == 'OutcomePlanAudit':
                return kw['schema'](mutually_exclusive=False, overlapping_category_ids=['left', 'right'], reason='Overlap')
            return kw['schema'](categories=request().outcome_definitions)
    req = request(); req.outcome_definitions = []
    client = Client()
    with pytest.raises(ValueError, match='overlap check'):
        asyncio.run(_plan(req, WorldSnapshot(event_title='Case', agents=[agent('a', 'Team')]), client, 'test'))
    assert client.calls == 6


def test_legacy_indicator_view_is_conservative_and_does_not_rewrite_files(tmp_path):
    store = EnsembleStore(tmp_path / 'runs')
    run_id, req = prepare_experiment(request(), events=EventRepository(tmp_path / 'events'), store=store)
    p = OutcomePlan(categories=req.outcome_definitions, event_questions=[question()])
    store.save(run_id, 'plan', p.model_dump())
    store.save_sample(run_id, 1, record('no', reason='Not recorded'))
    original = {'categories': [{'id': 'left', 'frequency': 1}], 'event_frequencies': [{'frequency': 0}], 'sample_count': 4}
    store.save(run_id, 'result', original)
    before = store.directory(run_id).joinpath('result.json').read_bytes()
    view = result_view(run_id, store, store.load(run_id, 'config'), original)
    assert view['legacy_measurement_view']
    assert view['event_frequencies'][0]['frequency'] is None
    assert view['event_frequencies'][0]['unclear_count'] == 1
    assert view['event_frequencies'][0]['unfinished_count'] == 3
    assert store.directory(run_id).joinpath('result.json').read_bytes() == before
    assert original['event_frequencies'][0]['frequency'] == 0


def test_mandatory_deliberation_skips_model_activation_gates():
    class Client:
        async def json_call(self, **kw):
            raise AssertionError('The formal decision is already required by the program')
    _, active, decisions = asyncio.run(choose_active_agents(
        WorldSnapshot(event_title='Case', agents=[agent('a', 'A'), agent('b', 'B')]), [], Client(), 'test'))
    assert len(active) == 2 and all(d.mandatory for d in decisions)


def test_entity_type_uses_identity_not_a_scenario_name_list():
    assert resolve_entity_type({'agent_type': 'person', 'title': 'Armed political organization and militant group'})[0] == 'organization'
    assert resolve_entity_type({'agent_type': 'person', 'title': 'Chief executive of a technology company'})[0] == 'person'
    assert resolve_entity_type({'agent_type': 'person', 'description': 'Person representing a political organization'})[0] == 'person'


def test_explicit_state_measurements_do_not_need_a_classification_model(tmp_path):
    from tests.test_ensemble import runner
    req = request(initial_world_state={'signed': False}, start_date='2026-10-31')
    req.outcome_definitions[0].conditions = [OutcomeCondition(path='signed', operator='eq', value=False)]
    req.outcome_definitions[1].conditions = [OutcomeCondition(path='signed', operator='eq', value=True)]
    req.event_questions = [question(conditions=[OutcomeCondition(path='signed', operator='eq', value=True)], scope='terminal')]
    class Client:
        async def json_call(self, **kw):
            raise AssertionError('Structured state rules are evaluated locally')
    store = EnsembleStore(tmp_path / 'runs')
    rid, req = prepare_experiment(req, events=EventRepository(tmp_path / 'events'), store=store)
    result = asyncio.run(run_experiment(rid, req, store=store, runtime=Client(), round_runner=runner([])))
    assert result['completed_samples'] == 4 and result['event_frequencies'][0]['no_count'] == 4
    assert result['event_frequencies'][0]['frequency'] == 0
    sample = store.sample(rid, 1)
    assert len(sample['measurement_history']) == 3
    assert sample['rounds'][0]['simulation_interval']['interval_end'] == '2026-11-30'


def test_automatic_plan_repairs_partition_and_omits_uncovered_questions():
    class Client:
        calls = 0
        async def json_call(self, **kw):
            self.calls += 1
            if kw['schema'].__name__ == 'OutcomePlanAudit':
                return kw['schema'](mutually_exclusive=self.calls == 4,
                    overlapping_category_ids=[] if self.calls == 4 else ['left', 'right'],
                    uncovered_question_ids=['meeting'], reason='Logical review')
            return kw['schema'](categories=request().outcome_definitions, event_questions=[question()])
    req = request(); req.outcome_definitions = []; req.event_questions = []
    client = Client()
    plan = asyncio.run(_plan(req, WorldSnapshot(event_title='Case', agents=[agent('a', 'Team')]), client, 'test'))
    assert client.calls == 4 and plan.partition_validation == 'model_reviewed'
    assert not plan.event_questions
    assert any('were removed' in warning for warning in plan.measurement_warnings)


def test_legacy_resume_preserves_plan_budget_and_committed_rounds(tmp_path):
    store = EnsembleStore(tmp_path / 'runs'); events = EventRepository(tmp_path / 'events')
    rid, req = prepare_experiment(request(max_model_calls=1500, max_total_tokens=50000000), events=events, store=store)
    config = store.load(rid, 'config'); config['engine_version'] = 'ensemble-5'; store.save(rid, 'config', config)
    store.save(rid, 'plan', {'categories': [c.model_dump() for c in req.outcome_definitions], 'event_questions': []})
    store.save_sample(rid, 1, {'sample_id': 1, 'status': 'completed', 'rounds': [{'step': 1}], 'snapshot': {}})
    before = store.directory(rid).joinpath('samples/0001.json').read_bytes()
    _, resumed = prepare_experiment(request(resume_run_id=rid), events=events, store=store)
    assert resumed.max_model_calls == 1500 and resumed.max_total_tokens == 50000000
    assert store.directory(rid).joinpath('samples/0001.json').read_bytes() == before
    assert len(store.load(rid, 'plan')['categories']) == 2


def test_case_experiment_discovery_is_read_only_and_does_not_expose_options(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import app.api.experiments as api
    store = EnsembleStore(tmp_path / 'runs'); events = EventRepository(tmp_path / 'events')
    rid, _ = prepare_experiment(request(), events=events, store=store)
    config = store.load(rid, 'config'); config['request']['event_id'] = 'case'; store.save(rid, 'config', config)
    monkeypatch.setattr(api, 'EnsembleStore', lambda: store)
    app = FastAPI(); app.include_router(api.router)
    response = TestClient(app).get('/api/experiments?event_id=case')
    assert response.status_code == 200
    assert response.json()['runs'][0]['run_id'] == rid
    assert 'request' not in response.json()['runs'][0] and 'api_key' not in response.text


def test_inconsistent_profile_repairs_only_its_own_schema_and_reuses_cache(tmp_path, monkeypatch):
    import app.layers.pipeline.profile_service as service
    calls = {'dimensions': 0, 'profiles': 0}
    async def dimensions(**kwargs):
        calls['dimensions'] += 1
        return []
    async def profile(**kwargs):
        calls['profiles'] += 1
        return {'agent_id': 'a', 'source_figure': 'Team', 'entity_type': kwargs['profile']['entity_type'],
                'semantic': {'identity': {'position': 'Armed political organization'}}}
    monkeypatch.setattr(service, '_generate_power_dimensions', dimensions)
    monkeypatch.setattr(service, 'generate_profile', profile)
    args = dict(api_key='fixture', model='fixture', event_title='Case',
                figures=[{'name': 'Team', 'entity_type': 'person'}], checkpoint_dir=tmp_path)
    profiles, _ = asyncio.run(service.generate_profiles(**args))
    assert profiles[0]['entity_type'] == 'organization'
    assert calls == {'dimensions': 1, 'profiles': 2}
    asyncio.run(service.generate_profiles(**args))
    assert calls == {'dimensions': 1, 'profiles': 2}


def test_automatic_plan_drops_unobservable_questions_without_reopening_the_partition():
    class Client:
        calls = 0
        async def json_call(self, **kw):
            self.calls += 1
            if kw['schema'].__name__ == 'OutcomePlanAudit':
                return kw['schema'](mutually_exclusive=True, reason='OK')
            q = EventQuestion(id='quiet', label='Quiet', criterion='连续72小时没有袭击')
            return kw['schema'](categories=request().outcome_definitions, event_questions=[q, question()])
    req = request(); req.outcome_definitions = []; req.event_questions = []
    client = Client()
    plan = asyncio.run(_plan(req, WorldSnapshot(event_title='Case', agents=[agent('a', 'Team')]), client, 'test'))
    assert client.calls == 2
    assert [q.criterion for q in plan.event_questions] == ['A meeting occurs']
    assert plan.measurement_warnings


def test_duplicate_event_references_are_only_returned_once():
    r = record(event_refs=['e1', 'e1'])
    result = event_metrics(question(), [r], 1, wilson)
    assert [e['timeline_id'] for e in result['answers_by_sample'][0]['evidence']] == ['e1']


def test_unlimited_task_budget_accepts_real_nulls_and_counts_usage():
    from app.domain.knowledge import FiguresRequest
    from app.llm.budget import ModelBudget
    req = FiguresRequest(event_id='case', event_title='test', max_model_calls=None,
                         max_total_tokens=None, extraction_deadline_seconds=None)
    budget = ModelBudget(req)
    for _ in range(2):
        ticket = budget.reserve(100, 100)
        budget.settle(ticket, {'prompt_tokens': 100, 'completion_tokens': 50}, 200)
    assert budget.report()['actual_http_requests'] == 2
    assert budget.report()['limits']['max_calls'] is None
    assert budget.report()['limits']['max_total_tokens'] is None
    assert request(max_model_calls=None, max_total_tokens=None, max_runtime_seconds=None).max_runtime_seconds is None


def test_generated_log_absence_is_counted_separately_from_strict_nonoccurrence():
    r=record('no', reason='The generated trajectory contains no meeting.')
    result=event_metrics(question(),[r],1,wilson)
    assert result['no_count']==0 and result['unclear_count']==1
    assert result['frequency'] is None
    assert result['trace_metrics']['not_matched_count']==1
    assert result['trace_metrics']['match_frequency']==0
    assert result['answers_by_sample'][0]['trace_answer']=='not_matched'


def test_trace_matching_survives_strict_validation_and_reload():
    r=record('unclear',evidence_kind='not_observed',reason='No full meeting event in the generated log.')
    first=resolve_answer(question(),r['classification']['questions'][0],r)
    r['classification']['questions']=[first.model_dump()]
    result=event_metrics(question(),[r],2,wilson)
    assert result['trace_metrics']['reviewed_count']==1
    assert result['trace_metrics']['unfinished_count']==1
    assert result['trace_metrics']['not_matched_count']==1
    assert result['answers_by_sample'][0]['trace_reason']=='No full meeting event in the generated log.'


def test_trace_matching_does_not_convert_missing_classification_or_invalid_citation_to_a_zero():
    r=record();r['classification']['questions']=[]
    result=event_metrics(question(),[r],1,wilson)
    assert result['trace_metrics']['match_frequency'] is None
    assert result['trace_metrics']['unclear_count']==1
    r=record('yes',event_refs=['invented'])
    result=event_metrics(question(),[r],1,wilson)
    assert result['trace_metrics']['matched_count']==0
    assert result['trace_metrics']['reviewed_count']==0


def test_duration_cannot_be_claimed_as_a_positive_trace_match_without_observations():
    r=record('yes')
    q=EventQuestion(id='meeting',label='Meeting',criterion='A meeting lasted for 72 hours')
    result=event_metrics(q,[r],1,wilson)
    assert result['trace_metrics']['match_frequency'] is None
    assert result['unclear_count']==1
