"""Explanations may be shortened; audit decisions remain strict."""
import asyncio
import json
from contextlib import asynccontextmanager
import pytest
from pydantic import ValidationError
from app.domain import LLMOptions, WorldSnapshot
from app.domain.ensemble import OutcomePlanAudit
from app.llm.client import LLMClient
from app.simulation.ensemble import _plan
from test_ensemble import request


@pytest.mark.parametrize('text', ['x'*1500, 'x'*1501, '这三个分类存在重叠。'*500, 'Explanation. '*400])
def test_reason_is_bounded_without_changing_decisions(text):
    raw={'mutually_exclusive':False,'overlapping_category_ids':['left','right'],
         'reason':text,'uncovered_question_ids':['q1']}
    result=OutcomePlanAudit.model_validate_json(json.dumps(raw))
    assert len(result.reason)<=1500
    assert result.reason == (text if len(text)<=1500 else text[:1499].rstrip()+'…')
    assert not result.mutually_exclusive
    assert result.overlapping_category_ids==['left','right'] and result.uncovered_question_ids==['q1']
    assert raw['reason']==text


@pytest.mark.parametrize('patch', [{'reason':None},{'reason':42},{'mutually_exclusive':'false'},
                                  {'overlapping_category_ids':'left,right'}])
def test_invalid_critical_fields_are_not_rescued(patch):
    with pytest.raises(ValidationError):
        OutcomePlanAudit.model_validate({'mutually_exclusive':True,'reason':'brief',**patch})


def test_strict_client_accepts_long_reason_without_an_http_retry(monkeypatch):
    from app.llm import client as module
    calls=[]
    class Response:
        is_error=False
        def json(self):
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
                'mutually_exclusive':True,'overlapping_category_ids':[], 'reason':'A detailed explanation. '*200})}}]}
    class HTTP:
        async def post(self,*args,**kwargs):calls.append(kwargs);return Response()
    @asynccontextmanager
    async def transport(**kwargs):yield HTTP()
    monkeypatch.setattr(module,'shared_http_client',transport)
    client=LLMClient(LLMOptions(api_key='fixture',model='fixture',base_url='https://example.test/v1'))
    client.strict_json=True
    client.max_attempts=2
    result=asyncio.run(client.json_call(call_name='audit',system='Audit',prompt='Fixture',schema=OutcomePlanAudit))
    assert result.mutually_exclusive and len(result.reason)<=1500 and result.reason.endswith("…") and len(calls)==1


def test_verbose_negative_audit_still_blocks_an_overlapping_plan():
    class Model:
        calls=0
        async def json_call(self,**kw):
            self.calls+=1
            if kw['schema'] is OutcomePlanAudit:
                return OutcomePlanAudit(mutually_exclusive=False,overlapping_category_ids=['left','right'],reason='These categories overlap. '*200)
            return kw['schema'](categories=request().outcome_definitions)
    req=request();req.outcome_definitions=[];model=Model()
    with pytest.raises(ValueError,match='overlap check.*These categories overlap'):
        asyncio.run(_plan(req,WorldSnapshot(event_title='Fixture'),model,'test'))
    assert model.calls==6


def test_final_partition_repair_reuses_rejected_criteria_and_keeps_approved_partition():
    class Model:
        audits = 0

        async def json_call(self, **kwargs):
            if kwargs['schema'] is OutcomePlanAudit:
                self.audits += 1
                return OutcomePlanAudit(
                    mutually_exclusive=self.audits == 3,
                    overlapping_category_ids=['left', 'right'] if self.audits < 3 else [],
                    uncovered_question_ids=['unobserved'],
                    reason='Both states can hold together.',
                )
            if self.audits == 2:
                assert 'rejected_categories' in kwargs['prompt']
                assert 'exactly TWO categories' in kwargs['prompt']
                assert 'Both states can hold together.' in kwargs['prompt']
            return kwargs['schema'](
                categories=request().outcome_definitions,
                event_questions=[{'id': 'unobserved', 'label': 'Unobserved', 'criterion': 'Requires unavailable evidence'}],
            )

    req = request()
    req.outcome_definitions = []
    req.event_questions = []
    model = Model()
    result = asyncio.run(_plan(req, WorldSnapshot(event_title='Fixture'), model, 'test'))
    assert model.audits == 3
    assert result.partition_validation == 'model_reviewed'
    assert not result.event_questions
    assert result.measurement_warnings
