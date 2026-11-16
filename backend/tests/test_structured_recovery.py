import asyncio
import json
from contextlib import asynccontextmanager
import pytest
from app.llm.client import strict_structured_json, LLMError, LLMClient
from app.domain import LLMOptions, WorldSnapshot, RelationshipProjection
from app.domain.decision_core import CurrentPlan
from app.domain.ensemble import OutcomePlanAudit
from app.storage.round_stages import RoundStages
from app.application.forecasting import ForecastRoundOrchestrator
from test_ensemble import actor
from test_compact_dossiers import action, settlement


def test_single_complete_json_with_prose_or_fences_is_accepted():
    obj=json.dumps({'mutually_exclusive':True,'reason':'Valid'})
    for raw in [obj,'Here is the result:\n'+obj+'\nDone.', '```json\n'+obj+'\n```']:
        assert strict_structured_json(raw,OutcomePlanAudit).mutually_exclusive


@pytest.mark.parametrize('text',[
 '{"mutually_exclusive":true,"reason":"first"}{"mutually_exclusive":false,"reason":"second"}',
 '{"mutually_exclusive":true,"reason":"valid"} false',
 '{"mutually_exclusive":true,"reason":"cut',
 '{"mutually_exclusive":true,"mutually_exclusive":false,"reason":"duplicate"}',
 '{"outer":{"mutually_exclusive":true,"reason":"nested"}',
 '{"mutually_exclusive":true,"reason":"valid"}, "another_field":true',
])
def test_ambiguous_or_incomplete_content_is_never_silently_salvaged(text):
    with pytest.raises((LLMError,ValueError)):strict_structured_json(text,OutcomePlanAudit)


def test_long_plan_is_preserved_word_for_word():
    description='Maintain the offer subject to independent verification. '*30
    assert 600<len(description)<3000
    assert CurrentPlan(description=description).description==description
    with pytest.raises(ValueError):CurrentPlan(description='x'*3001)


def test_format_retry_contains_the_actual_invalid_output_and_error(monkeypatch):
    from app.llm import client as module
    requests=[]
    invalid='{"mutually_exclusive":true,"reason":"first"}{"mutually_exclusive":false,"reason":"second"}'
    class Response:
        is_error=False
        def json(self):return {'choices':[{'finish_reason':'stop','message':{'content':invalid if len(requests)==1 else '{"mutually_exclusive":false,"reason":"Repaired"}'}}]}
    class HTTP:
        async def post(self,*a,**kw):requests.append(kw['json']);return Response()
    @asynccontextmanager
    async def transport(**kw):yield HTTP()
    monkeypatch.setattr(module,'shared_http_client',transport)
    client=LLMClient(LLMOptions(api_key='fixture',model='fixture',base_url='https://example.test/v1'))
    client.strict_json=True;client.max_attempts=2
    result=asyncio.run(client.json_call(call_name='test',system='Audit',prompt='Test',schema=OutcomePlanAudit))
    assert not result.mutually_exclusive and len(requests)==2
    messages=str(requests[1]['messages'])
    assert 'VALIDATION ERROR' in messages and invalid in messages and 'PREVIOUS INVALID OUTPUT' in messages


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_settlement_gets_one_compact_repair_without_accepting_a_second_root(monkeypatch, repair_succeeds):
    from app.functional_agents.simulation import GameMasterAgent
    from app.llm import client as module

    requests = []
    a = actor()
    snapshot = WorldSnapshot(event_title="Supply", agents=[a])
    valid = json.dumps({
        "state_summary": "Supply remains stable.", "next_world_state": {}, "analysis": "",
        "agent_knowledge_updates": [{"agent_id": a.agent_id, "information": []}],
    })

    class Response:
        is_error = False

        def json(self):
            content = valid if repair_succeeds and len(requests) == 3 else valid + "{}"
            return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    class HTTP:
        async def post(self, *args, **kwargs):
            requests.append(kwargs["json"])
            return Response()

    @asynccontextmanager
    async def transport(**kwargs):
        yield HTTP()

    monkeypatch.setattr(module, "shared_http_client", transport)
    client = LLMClient(LLMOptions(api_key="fixture", model="fixture", base_url="https://example.test/v1"))
    client.strict_json = True
    client.max_attempts = 2
    call = GameMasterAgent(client).settle(snapshot, [], [], "test")
    if repair_succeeds:
        assert asyncio.run(call).state_summary == "Supply remains stable."
    else:
        with pytest.raises(LLMError, match="Ambiguous trailing JSON content"):
            asyncio.run(call)
    assert len(requests) == 3
    assert "FORMAT REPAIR" in requests[-1]["messages"][1]["content"]
    assert requests[-1]["temperature"] == 0


def test_round_cache_is_input_scoped_and_detects_corrupt_payloads(tmp_path):
    calls=[]
    async def compute():calls.append(1);return {'n':len(calls)}
    async def scenario():
        cache=RoundStages(tmp_path,{'sample':1})
        assert await cache.run('test',{},compute,dict)=={'n':1}
        assert await cache.run('test',{},compute,dict)=={'n':1}
        assert await RoundStages(tmp_path,{'sample':2}).run('test',{},compute,dict)=={'n':2}
        assert await cache.run('test',{'changed':True},compute,dict)=={'n':3}
        for p in tmp_path.glob('*.json'):
            value=json.loads(p.read_text());value['payload']={'tampered':True};p.write_text(json.dumps(value))
        assert await cache.run('test',{},compute,dict)=={'n':4}
    asyncio.run(scenario())


def test_settlement_failure_reuses_information_activation_and_audited_actions(tmp_path):
    from app.domain import ActivationDecision
    from app.simulation.transitions import apply_round_update
    s=WorldSnapshot(event_title='Supply',agents=[actor()]);calls=[]
    orchestrator=ForecastRoundOrchestrator(object(),update_relationships=False,checkpoint_dir=tmp_path)
    async def information(state,prefix):calls.append('information');return state
    async def activation(state,interventions,prefix):
        calls.append('activation');return state,state.agents,[ActivationDecision(agent_id='actor_a',selected=True,score=1,master_score=1,self_score=1,trigger_score=1)]
    async def actions(*args):calls.append('actions');return [action(decision='act')]
    async def settle(*args):
        calls.append('settlement')
        if calls.count('settlement')==1:raise LLMError('Fixture invalid JSON')
        return settlement()
    orchestrator.ordinary_agent.process_information=information
    orchestrator.activation_master.select=activation
    orchestrator.action_auditor.produce=actions
    orchestrator.game_master.settle=settle
    async def scenario():
        with pytest.raises(LLMError):await orchestrator.run(s,[],'same_sample_round')
        updated,_=await orchestrator.run(s,[],'same_sample_round')
        assert updated.round_id==1 and s.round_id==0
    asyncio.run(scenario())
    assert calls==['information','activation','actions','settlement','settlement']


def test_successful_peer_proposal_survives_another_actors_format_failure(tmp_path):
    from app.domain import ActionProposal
    from app.functional_agents.simulation import ActionAuditMasterAgent
    a=actor();b=a.model_copy(update={'agent_id':'actor_b','name':'Board'})
    state=WorldSnapshot(event_title='Supply',agents=[a,b]);counts={'actor_a':0,'actor_b':0}
    class Ordinary:
        async def decide(self,snapshot,agent,*args):
            counts[agent.agent_id]+=1
            if agent.agent_id=='actor_b' and counts['actor_b']==1:raise LLMError('Fixture plan format failure')
            return action(decision='act').model_copy(update={'agent_id':agent.agent_id,'agent_name':agent.name,'action_id':f'action_1_{agent.agent_id}'})
    class Master:
        async def json_call(self,**kw):
            return kw['schema'](round_id=1,snapshot_version=0,reviews=[{'agent_id':x.agent_id,'accepted':True} for x in state.agents])
    auditor=ActionAuditMasterAgent(Master(),Ordinary())
    auditor.proposal_cache=RoundStages(tmp_path,{'sample':1,'round':1})
    async def scenario():
        with pytest.raises(LLMError):await auditor.produce(state,state.agents,[],'fixture')
        proposals=await auditor.produce(state,state.agents,[],'fixture')
        assert {p.agent_id for p in proposals}=={'actor_a','actor_b'}
    asyncio.run(scenario())
    assert counts=={'actor_a':1,'actor_b':2}


def test_retrieval_timing_does_not_invalidate_but_changed_evidence_does(tmp_path):
    count=[]
    async def generate():count.append(1);return {'ok':True}
    def state(t,ids):return {'agents':[{'agent_id':'a','retrieval_context':{'duration_ms':t,'selected_ids':ids}}]}
    async def scenario():
        for t in [1.1,7.9]:
            cache=RoundStages(tmp_path,{'snapshot':state(t,['e1'])})
            await cache.run('information',state(t,['e1']),generate,dict)
        assert len(count)==1
        cache=RoundStages(tmp_path,{'snapshot':state(1.1,['e2'])})
        await cache.run('information',state(1.1,['e2']),generate,dict)
        assert len(count)==2
    asyncio.run(scenario())
