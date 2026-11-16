"""Recovery at the paid profile boundary; all providers are stubbed."""
import asyncio

import pytest

from app.domain import FiguresRequest
from app.knowledge import extraction
from app.layers.pipeline import profile_service, steps
from app.layers.pipeline.schemas import LayerStageRequest
from app.layers.pipeline.stage import ArtifactDir
from app.llm.budget import BudgetExceeded, ModelBudget
from app.storage.extraction_status import ExtractionStatusRepository
from app.storage.json_store import EventRepository


def case(tmp_path):
    events = EventRepository(tmp_path)
    event_id = events.create('Case', ['https://example.test'], None)['event_id']
    events.save_sources(event_id, [{'url':'https://example.test', 'title':'Case',
                                   'content':'2026-01: a documented event.'}])
    return events, event_id


def test_old_false_completion_is_reported_as_partial_and_cannot_be_loaded_as_success(tmp_path):
    events, event_id = case(tmp_path)
    events.save_artifact(event_id, 'agents', [])
    events.save_artifact(event_id, 'layer_quality', {'key_figures_selected':6,
        'warnings':['Profile generation failed entirely: request budget exhausted']})
    events.save_artifact(event_id, 'timeline_cache', {'timeline':[{'timeline_id':'h'}]})
    events.save_artifact(event_id, 'extraction_status', {'status':'completed','progress':100})
    status = ExtractionStatusRepository(events).get(event_id)
    assert status['status'] == 'partial' and status['resume_available']
    assert status['progress'] < 100 and 'budget exhausted' in status['error']
    assert extraction.figures_result_from_artifacts(event_id, events) is None


def test_background_runner_does_not_overwrite_partial_progress_with_generic_failure(tmp_path, monkeypatch):
    from app.api import knowledge

    events, event_id = case(tmp_path)
    monkeypatch.setattr(knowledge, 'events', events)

    async def fail(request, store):
        store.save_artifact(event_id, 'timeline_cache', {'timeline':[{'timeline_id':'h'}]})
        ExtractionStatusRepository(store).update(event_id,run_id='fixture',stage='profiles',
            progress=90,message='Generating profiles')
        raise BudgetExceeded('request cap')

    monkeypatch.setattr(extraction, '_extract_figures', fail)
    request = FiguresRequest(event_id=event_id,event_title='Case')
    asyncio.run(knowledge._run_figures_pipeline(request))
    status = ExtractionStatusRepository(events).get(event_id)
    assert status['status'] == 'partial' and status['error'] == 'request cap'
    assert status['resume_available']
    assert events.load_artifact(event_id,'timeline_cache',{})['timeline']


def test_resume_keeps_spent_budget_and_rejects_changed_sources(tmp_path):
    events, event_id = case(tmp_path)
    request = FiguresRequest(event_id=event_id,event_title='Case',model='fixture',
                             max_model_calls=1,resume_profiles=True)
    ledger = ModelBudget(request)
    ledger.reserve(10,100)
    events.save_artifact(event_id,'extraction_budget',ledger.report())
    events.save_artifact(event_id,'timeline_cache',{'timeline':[{'timeline_id':'h'}],
        'cache_key':extraction._timeline_cache_key(
            request.model_copy(update={'language':'zh-CN'}),events.load_sources(event_id))})
    with pytest.raises(ValueError,match='Raise the total request limit'):
        extraction.extraction_resume_budget(request,events)
    request = request.model_copy(update={'max_model_calls':3})
    previous = extraction.extraction_resume_budget(request,events)
    resumed = ModelBudget(request,previous)
    assert resumed.calls == 1 and resumed.report()['accounted_tokens_upper'] == 110
    events.save_sources(event_id,[{'url':'https://example.test','content':'Different material.'}])
    with pytest.raises(ValueError,match='saved timeline differs'):
        extraction.extraction_resume_budget(request,events)


def test_profile_successes_survive_a_sibling_budget_failure_and_are_not_rebilled(tmp_path, monkeypatch):
    calls = {'dimensions':0,'A':0,'B':0}

    async def dimensions(**kw):
        calls['dimensions'] += 1
        return [{'name':'capacity','description':'Observed capacity'}]

    async def generate(**kw):
        name = kw['profile']['canonical_name']
        calls[name] += 1
        if name == 'B' and calls[name] == 1:
            raise BudgetExceeded('profile budget')
        await asyncio.sleep(.01)
        return {'agent_id':name,'source_figure':name}

    monkeypatch.setattr(profile_service,'_generate_power_dimensions',dimensions)
    monkeypatch.setattr(profile_service,'generate_profile',generate)
    args = dict(api_key='fixture',model='fixture',event_title='Case',
                figures=[{'canonical_name':'A'},{'canonical_name':'B'}],
                checkpoint_dir=tmp_path/'checkpoints')
    with pytest.raises(BudgetExceeded):
        asyncio.run(profile_service.generate_profiles(**args))
    profiles,_ = asyncio.run(profile_service.generate_profiles(**args))
    assert {p['agent_id'] for p in profiles} == {'A','B'}
    assert calls == {'dimensions':1,'A':1,'B':2}


def test_profile_inputs_skip_actor_extraction_on_resume(tmp_path,monkeypatch):
    events,event_id = case(tmp_path)
    events.save_artifact(event_id,'profile_inputs',{'checkpoint_key':'matching',
        'figures':[{'canonical_name':'A'}],'members':{},'timeline':[{'timeline_id':'h'}],
        'name_mapping':{},'quality':{'key_figures_selected':1}})

    async def finish(**kw):
        assert kw['figure_details'] == [{'canonical_name':'A'}]
        assert kw['merged'] == [{'timeline_id':'h'}]
        return {'reused_profile_inputs':True}

    monkeypatch.setattr(steps,'_finish_profiles',finish)
    request=LayerStageRequest(event_id=event_id,event_title='Case',api_key='fixture',
                              model='fixture',resume_profiles=True,checkpoint_key='matching')
    result=asyncio.run(steps._run_body(req=request,merged=[],event_dir=ArtifactDir(events,event_id),
        sem=None,warnings=[],extract_trace=None,active_stage='figure',stage_started_at='',
        stage_started_perf=0,merged_all=[],older=[],raw_timeline_events=[],cutoff='',timeline_quality={}))
    assert result['reused_profile_inputs']


def test_contract_subject_keeps_dossier_slot_after_a_hierarchy_demotion(tmp_path, monkeypatch):
    from app.layers.pipeline.stage import TraceShim

    events, event_id = case(tmp_path)
    candidate = {"canonical_name": "J.D. Example", "entity_type": "person", "id": "example"}
    other = {"canonical_name": "Other delegation", "entity_type": "organization", "id": "other"}
    events.save_artifact(event_id, "figures_merged", {"figures": [candidate, other]})
    history = {"timeline_id": "h1", "timestamp": "2025-01-01", "description": "J.D. Example led the delegation.",
               "date_evidence": {"status": "supported"}, "evidence_text": "J.D. Example led the delegation.",
               "figures": [{"name": "JD Example", "title": "Delegation chair"}],
               "actions": [{"source": "JD Example", "action": "Led the delegation"}]}

    async def generate(**kwargs):
        assert [row["id"] for row in kwargs["figures"]] == ["example", "other"]
        assert kwargs["figures"][0]["actions_initiated"][0]["timeline_id"] == "h1"
        assert kwargs["members_by_principal"] == {"Other delegation": []}
        assert kwargs["previous_figures"] == [other]
        return [], []

    monkeypatch.setattr(steps, "generate_profiles", generate)
    request = LayerStageRequest(event_id=event_id, event_title="Case", model="fixture", api_key="fixture",
                                required_decision_subject="JD Example", resume_profiles=True)
    asyncio.run(steps._finish_profiles(
        req=request, merged=[history], event_dir=ArtifactDir(events, event_id), figure_details=[other],
        principal_members={"Other delegation": [{"name": "JD Example"}]}, name_mapping={},
        quality={}, timeline_quality={}, extract_trace=TraceShim(), fallback_warning="",
    ))
    saved = events.load_artifact(event_id, "profile_inputs", {})
    assert saved["quality"]["required_subject_review"]["status"] == "retained"
    unverified = history | {"date_evidence": {"status": "needs_review"}}
    assert steps._retain_required_subject("JD Example", [candidate], [unverified], {}, [], {}, 8)[2] == "no_supported_action"


def test_poll_only_actors_do_not_displace_a_party_with_a_supported_action():
    observer = {"role_in_event": "polling firm conducting the opinion poll",
                "history": [{"description": "The firm published an opinion poll."}]}
    assert steps._poll_only_figure(observer)
    assert not steps._poll_only_figure(observer | {"role_in_event": "political party measured in the poll"})
    assert not steps._poll_only_figure(observer | {"actions_initiated": [{"action": "Signed a coalition agreement"}]})


def test_completion_defaults_are_larger_but_explicit_limits_remain_enforced():
    request=FiguresRequest(event_id='case',event_title='Case')
    assert request.max_model_calls == 300 and request.max_total_tokens == 2000000
    explicit=FiguresRequest(event_id='case',event_title='Case',max_model_calls=1)
    budget=ModelBudget(explicit)
    budget.reserve(1,1)
    with pytest.raises(BudgetExceeded):
        budget.reserve(1,1)


def test_optional_member_style_budget_failure_does_not_discard_completed_dossiers(tmp_path,monkeypatch):
    async def dimensions(**kw):
        return [{'name':'capacity'}]

    async def generate(**kw):
        return {'agent_id':'a','source_figure':'A'}

    async def traits(**kw):
        raise BudgetExceeded('optional traits budget')

    monkeypatch.setattr(profile_service,'_generate_power_dimensions',dimensions)
    monkeypatch.setattr(profile_service,'generate_profile',generate)
    monkeypatch.setattr(profile_service,'_generate_member_traits',traits)
    monkeypatch.setattr(profile_service,'_build_member_profile',lambda *args: {'agent_id':'member'})
    profiles,_=asyncio.run(profile_service.generate_profiles(api_key='fixture',model='fixture',event_title='Case',
        figures=[{'canonical_name':'A'}],members_by_principal={'A':[{'name':'Member','id':'member'}]},
        checkpoint_dir=tmp_path/'checkpoints'))
    assert profiles[0]['agent_id']=='a'
    assert profiles[0]['member_traits_status']=='not_generated_budget_limit'
    assert profiles[0]['subordinates'][0]['style']==''


def test_successful_optional_member_traits_are_reused_on_resume(tmp_path,monkeypatch):
    calls={'dimensions':0,'principal':0,'members':0}
    async def dimensions(**kw):
        calls['dimensions']+=1
        return [{'name':'capacity'}]
    async def generate(**kw):
        calls['principal']+=1
        return {'agent_id':'a','source_figure':'A'}
    async def traits(**kw):
        calls['members']+=1
        return {'Member':{'style':'Unknown; no observed decision evidence'}}
    monkeypatch.setattr(profile_service,'_generate_power_dimensions',dimensions)
    monkeypatch.setattr(profile_service,'generate_profile',generate)
    monkeypatch.setattr(profile_service,'_generate_member_traits',traits)
    monkeypatch.setattr(profile_service,'_build_member_profile',lambda *args: {'agent_id':'member'})
    args=dict(api_key='fixture',model='fixture',event_title='Case',figures=[{'canonical_name':'A'}],
        members_by_principal={'A':[{'name':'Member','id':'member'}]},checkpoint_dir=tmp_path/'checkpoints')
    first,_=asyncio.run(profile_service.generate_profiles(**args))
    second,_=asyncio.run(profile_service.generate_profiles(**args))
    assert first==second and calls=={'dimensions':1,'principal':1,'members':1}


def test_resume_inherits_omitted_prices_and_does_not_remove_original_cost_cap(tmp_path):
    events,event_id=case(tmp_path)
    original=FiguresRequest(event_id=event_id,event_title='Case',model='fixture',
        max_model_calls=2,max_cost_usd=1,input_price_per_million=2,output_price_per_million=4)
    ledger=ModelBudget(original)
    ledger.reserve(10,10)
    events.save_artifact(event_id,'extraction_budget',ledger.report())
    events.save_artifact(event_id,'timeline_cache',{'timeline':[{'timeline_id':'h'}],
        'cache_key':extraction._timeline_cache_key(original,events.load_sources(event_id))})
    resume=FiguresRequest(event_id=event_id,event_title='Case',model='fixture',resume_profiles=True)
    previous=extraction.extraction_resume_budget(resume,events)
    assert resume.max_model_calls==2 and resume.max_cost_usd==1
    assert resume.input_price_per_million==2 and resume.output_price_per_million==4
    assert ModelBudget(resume,previous).calls==1
