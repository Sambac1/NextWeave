import asyncio
import json

import pytest

from app.domain import ForecastRound, WorldSnapshot, TimelineEvent
from app.domain.forecast import ContractOutlook
from app.domain.identity import leading_actor_span
from app.simulation.main_forecast import READOUT_POLICY, run_checkpointed_round
from app.simulation.outlook import PublicReadout, read_public_outlook
from app.simulation.question_report import build_question_report
from app.simulation.readout_grounding import ReadoutGrounding
from app.storage.round_stages import RoundStages
from test_ensemble import actor
from test_main_forecast import RULES, RUN


def test_weighted_conditional_estimates_are_not_erased_for_lack_of_a_binary_verdict():
    config = {'request':{'steps':1, 'question_simulation':{'algorithm':'conditional-stratified-v2'},
                         'initial_world_state':{'benchmark_contract':RULES}}}
    def sample(p):
        return {'status':'completed','rounds':[{'step':1,'decision_sampling':{'audited_silent_round':1},
            'contract_outlook':{'p_yes':p,'basis':'The action remains unexecuted.', 'estimate_kind':'terminal_estimate'}}]}
    report = build_question_report(RUN, config, [sample(.02), sample(.08)], masses=[.25,.75])
    assert report['p_yes'] == pytest.approx(.065)
    assert report['status'] == 'complete' and report['verified_outcomes'] == 0
    assert report['probability_mass']['yes'] == pytest.approx(.065)
    partial = build_question_report(RUN, config, [sample(.02), {'status':'interrupted','rounds':[], 'error':'Round timeout.'}], masses=[.25,.75])
    assert partial['p_yes'] is None and partial['conditional_estimate'] == pytest.approx(.02)
    assert partial['trajectories'][1]['error'] == 'Round timeout.'
    assert partial['trajectories'][1]['planned_rounds'] == 1


def test_possessive_principal_and_acronym_forms_preserve_executor_identity():
    country = actor().model_copy(update={'name':'Federated Republic', 'aliases':['FR'], 'agent_type':'country'})
    assert leading_actor_span("Federated Republic's leadership announced talks.", [country], country.agent_id) == 0
    assert leading_actor_span('The F.R. announced talks.', [country], country.agent_id) == 4
    assert leading_actor_span('A delegation requested that the Federated Republic act.', [country], country.agent_id) is None
    board = actor().model_copy(update={'agent_id':'independent', 'name':"Federated Republic's leadership"})
    assert leading_actor_span("Federated Republic's leadership announced talks.", [country,board], country.agent_id) is None


def test_timeout_retries_only_the_uncommitted_operation_and_checks_the_budget():
    calls, checks = [], []
    async def operation():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError('Transient provider delay')
        return 'completed'
    result = asyncio.run(run_checkpointed_round(operation, timeout=1, check=lambda:checks.append(1)))
    assert result == 'completed' and len(calls) == len(checks) == 2


def test_strict_qualifier_review_rejects_related_activity_and_invalidates_old_cached_estimates(tmp_path):
    question = 'Will the board approve and publish the acquisition closing notice?'
    quote = 'The board discussed the acquisition without approving it or publishing a closing notice.'
    contract = {'question':question, 'rules':'Yes requires approval AND publication of the signed closing notice.',
        'forecast_criteria':{'yes':'Approval and publication.','no':'No qualifying closing by the deadline.'},
        'as_of':'2026-06-01T00:00:00Z','outcome_deadline':'2026-07-01T00:00:00Z'}
    before = WorldSnapshot(event_title=question, world_state={'benchmark_contract':contract,
        'contract_outlook':{'p_yes':.93,'basis':'Discussion satisfies the outcome.', 'evidence_ref':'history', 'evidence_quote':quote}})
    event = TimelineEvent(timeline_id='public1', timestamp='2026-06-10', description=quote, origin='simulated', confidence=.8, importance=.8)
    frame = ForecastRound(step=1,input_snapshot_version=0,output_snapshot_version=1,state_summary=quote,timeline=[event])
    after = before.model_copy(deep=True)
    clock = {'step':1,'horizon_steps':5,'interval_end_utc':'2026-06-10T00:00:00Z'}
    generated, audits = [], []
    class Runtime:
        async def json_call(self, *, schema, prompt, **kwargs):
            data = json.loads(prompt)
            if schema is PublicReadout:
                assert data['strict_criteria'] and not data['previous_estimate']
                key = next(key for key,row in data['evidence'].items() if row['text'] == quote)
                generated.append(data)
                return schema(outlook=ContractOutlook(p_yes=.93 if len(generated)==1 else .06,
                    basis='Discussion already satisfies the contract.' if len(generated)==1 else 'Approval and publication remain unexecuted.',
                    evidence_ref='copied_wrong_id' if len(generated)==2 else key,evidence_quote=quote))
            assert schema is ReadoutGrounding
            audits.append(data)
            return schema(supported=False,issues=[{'claim':'Discussion satisfies closing.',
                'problem':'The rule also requires approval and publication.', 'evidence_key':data['cited_evidence_key'],
                'status':'unsupported'}]) if len(audits)==1 else schema(supported=True)
    async def check():
        return await read_public_outlook(before,after,frame,contract,clock,Runtime(),RoundStages(tmp_path,{}),{},'fixture', strict_criteria=True)
    outlook, issue = asyncio.run(check())
    assert issue is None and outlook.p_yes == .06 and outlook.readout_policy == READOUT_POLICY
    assert len(generated)==len(audits)==2


def test_clock_alone_does_not_reinterpret_unchanged_background(tmp_path):
    quote='The board has not yet published the acquisition closing notice.'
    contract={'question':'Will the board publish the closing notice?', 'rules':'A published closing notice is required.',
        'as_of':'2026-06-01T00:00:00Z','outcome_deadline':'2026-07-01T00:00:00Z'}
    history=TimelineEvent(timeline_id='history', timestamp='2026-06-01',description=quote, confidence=.9,importance=.8)
    prior={'p_yes':.4,'basis':'Publication remains unconfirmed.','evidence_ref':'history','evidence_quote':quote,'readout_policy':READOUT_POLICY}
    before=WorldSnapshot(event_title=contract['question'],timeline=[history],world_state={'benchmark_contract':contract,'contract_outlook':prior})
    frame=ForecastRound(step=2,input_snapshot_version=1,output_snapshot_version=2,state_summary='No new publication.')
    clock={'step':2,'horizon_steps':5,'interval_end_utc':'2026-06-15T00:00:00Z'}
    class Runtime:
        async def json_call(self, *, schema, prompt, **kwargs):
            if schema is ReadoutGrounding:return schema(supported=True)
            data=json.loads(prompt);key=next(key for key,row in data['evidence'].items() if row['text']==quote)
            return schema(outlook=ContractOutlook(p_yes=.3,basis='Less time remains for the required publication.',
                evidence_ref=key,evidence_quote=quote,change_driver='settled_event'))
    async def check():
        return await read_public_outlook(before,before,frame,contract,clock,Runtime(),RoundStages(tmp_path,{}),{},'clock',strict_criteria=True)
    result,issue=asyncio.run(check())
    assert issue is None and result.p_yes==.4
    assert result.evidence_ref=='history' and result.driver_ref is None


def test_terminal_readout_evaluates_realized_path_without_a_fresh_background_forecast(tmp_path):
    quote = 'The board adjourned without approving or publishing the closing notice.'
    contract = {'question':'Will the board approve and publish the closing notice?',
        'rules':'Yes requires approval AND publication by July 1.',
        'forecast_criteria':{'yes':'Approval and publication.','no':'No qualifying closing.'},
        'as_of':'2026-06-01T00:00:00Z','outcome_deadline':'2026-07-01T00:00:00Z',
        'historical_evidence':[{'source':'Frozen background','excerpt':'The board hoped to close the transaction.'}]}
    before = WorldSnapshot(event_title=contract['question'], world_state={
        'benchmark_contract':contract,'contract_outlook':{'p_yes':.93,'basis':'A possible closing.',
        'evidence_ref':'history_1','evidence_quote':'The board hoped to close the transaction.',
        'readout_policy':READOUT_POLICY}})
    event=TimelineEvent(timeline_id='settled',timestamp='2026-07-01',description=quote,
        origin='simulated',confidence=.8,importance=.8)
    frame=ForecastRound(step=5,input_snapshot_version=4,output_snapshot_version=5,state_summary=quote,timeline=[event])
    clock={'step':5,'horizon_steps':5,'interval_end_utc':'2026-07-01T00:00:00Z'}
    class Runtime:
        async def json_call(self, *, schema, prompt, **kwargs):
            if schema is ReadoutGrounding:return schema(supported=True)
            data=json.loads(prompt)
            assert data['evaluation_scope']=='completed_simulated_trajectory'
            assert not data['previous_estimate']
            assert all(row['kind'] not in {'historical','frozen_excerpt'} for row in data['evidence'].values())
            assert data['realized_rounds'][0]['settled_events'][0]['text']==quote
            key=next(key for key,row in data['evidence'].items() if row['text']==quote)
            return schema(outlook=ContractOutlook(p_yes=.04,basis='The realized path records no approval or publication.',
                evidence_ref=key,evidence_quote=quote))
    async def check():
        return await read_public_outlook(before,before,frame,contract,clock,Runtime(),RoundStages(tmp_path,{}),{},
                                         'terminal',strict_criteria=True)
    result,issue=asyncio.run(check())
    assert issue is None and result.p_yes==.04 and result.estimate_kind=='terminal_estimate'
