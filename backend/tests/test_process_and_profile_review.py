"""Cross-domain feedback, evidence semantics, privacy and targeted repair."""
import asyncio
from copy import deepcopy
import pytest
from app.domain import WorldSnapshot, TimelineEvent, GameMasterSettlement, RelationshipProjection, Desire
from app.domain.decision_core import validate_core, applicable_core
from app.domain.process import IssueUpdate
from app.simulation.process import validate_issue_updates, suppress_continuations
from app.simulation.process_summary import process_summary, public_issue_reconvergence, reliability_counts
from app.simulation.transitions import apply_round_update
from app.simulation.bdi_updates import apply_goal_statuses, process_agent_information
from app.simulation.activation import _seed_candidates
from app.knowledge.profile_evidence import source_catalog
from app.storage.json_store import EventRepository
from test_ensemble import actor
from test_compact_dossiers import action


def event(text='The board received the delivery proposal.', **kwargs):
    return TimelineEvent(timeline_id='sim_1_1',timestamp='2026-11',description=text,
                         actors=['actor_a','board'],confidence=.8,importance=.5,**kwargs)


def snapshot():
    return WorldSnapshot(event_title='Delivery',agents=[actor(),actor().model_copy(update={'agent_id':'board','name':'Board'})],world_state={
        'process_tracking_version':'issue-process-1','process_topics':[{'id':'delivery','title':'Delivery'}]})


def change(**kwargs):
    return IssueUpdate(**{'issue_id':'delivery','title':'Delivery','before':'','after':'Awaiting a board response',
        'kind':'proposal','state_code':'proposed','actor_ids':['actor_a'],'event_refs':['sim_1_1'],
        'evidence_quote':'The board received the delivery proposal.','pending_actor_ids':['board'],**kwargs})


def test_undated_source_spans_support_structural_authority_and_metadata_is_program_owned():
    sources=[{'evidence_id':'s','text':'The board has authority to approve the annual budget.',
              'url':'https://example.org/charter','title':'Charter'}]
    core=validate_core({'mandate':[{'description':'Budget approval rests with the board.',
        'claim_scope':'structural','source_refs':[{'evidence_id':'s','quote':sources[0]['text'],'url':'https://wrong.example'}],
        'provenance':'source_statement'}]},source_catalog=sources)
    assert core.mandate[0].provenance=='source_statement'
    assert core.mandate[0].source_refs[0].url=='https://example.org/charter'
    assert not core.mandate[0].timeline_refs
    with pytest.raises(ValueError,match='source span'):
        validate_core({'mandate':[{'description':'Invented authority','source_refs':[{'evidence_id':'s','quote':'Another claim entirely'}]}]},source_catalog=sources)


def test_incidents_and_other_actor_losses_do_not_become_operational_attributes():
    core=validate_core({'limits':[{'description':'Qatar was struck by missiles in March.'}],
        'resources':[{'description':'Iranian crude oil export value halted by a blockade.'}],
        'means':[{'description':'The board approved a project in May.','claim_scope':'historical_observation'}]})
    assert not core.limits and not core.resources and not core.means
    assert len(core.review_notes)==3


def test_validity_projection_keeps_original_and_does_not_renew_expired_rules():
    core=validate_core({'limits':[{'description':'A temporary suspension applies.','valid_until':'2026-08-31'},
        {'description':'The MoU included a 60-day suspension of hostilities.'}]})
    before=core.model_dump();reviewed=applicable_core(core,'2026-11-02')
    assert not reviewed.limits and len(reviewed.review_notes)==2
    assert core.model_dump()==before
    with pytest.raises(ValueError):validate_core({'limits':[{'description':'Bad dates','valid_from':'2026-11-02','valid_until':'2025-01-01'}]})


def test_structural_evidence_respects_identity_boundaries_and_cutoffs(tmp_path,monkeypatch):
    import app.knowledge.profile_evidence as module
    events=EventRepository(tmp_path);case=events.create('Authority',[],None)['event_id']
    events.save_sources(case,[{'url':'https://example.org/charter','title':'Charter','retrieved_at':'2026-10-01',
        'content':'The US Board has authority to approve the budget.\n\nRussia operates facilities with a different governance structure.'},
        {'url':'https://example.org/later','published_at':'2026-12-01','content':'The US Board operates a new plant under expanded authority.'}])
    monkeypatch.setattr(module,'EventRepository',lambda:events)
    catalog=source_catalog({'_case_id':case,'canonical_name':'US'},cutoff='2026-11-02')
    assert len(catalog)==1 and 'US Board' in catalog[0]['text']
    assert 'Russia' not in catalog[0]['text']


def test_goal_repair_keeps_valid_items_without_inventing_or_rewriting_goals():
    a=actor();a.desires=[Desire(desire_id='g1',description='Maintain supply'),Desire(desire_id='g2',description='Protect budget')]
    issues=[]
    apply_goal_statuses(a,[Desire(desire_id='renamed',description='Maintain supply',status='blocked'),
        Desire(desire_id='g2',description='Protect budget',status='satisfied'),
        Desire(desire_id='invented',description='Unrelated goal')],issues,2)
    assert [(g.desire_id,g.description,g.status) for g in a.desires]==[
        ('g1','Maintain supply','blocked'),('g2','Protect budget','satisfied')]
    counts=reliability_counts([{'status':'completed','snapshot':{'validation_issues':issues}}])
    assert counts['goal_ids_repaired']==1 and counts['goal_updates_rejected']==1 and counts['bdi_updates_rejected']==0


def test_conflicting_goal_statuses_are_rejected():
    a=actor();a.desires=[Desire(desire_id='g',description='Maintain supply')];issues=[]
    apply_goal_statuses(a,[Desire(desire_id='g',description='Maintain supply',status='blocked'),
                          Desire(desire_id='g',description='Maintain supply',status='satisfied')],issues,2)
    assert a.desires[0].status=='active' and issues[0]['reason']=='conflicting_status_updates'


def test_identifier_repair_retries_only_the_affected_actor():
    from app.domain import PrivateObservation
    values=snapshot().agents
    for a in values:a.private_observations=[PrivateObservation(observation_id='p',description='New information',received_round=0)]
    calls=[]
    class Provider:
        async def json_call(self,**kw):
            calls.append(kw['call_name']);board='_agent_board_' in kw['call_name']
            return kw['schema'](agent_id='board' if board else 'actor_a',round_id=1 if board or 'repair' in kw['call_name'] else 0,
                snapshot_version=0,assessment='Reviewed')
    asyncio.run(process_agent_information(WorldSnapshot(event_title='Delivery',agents=values),Provider(),'test'))
    assert len(calls)==3 and sum('_agent_board_' in c for c in calls)==1


@pytest.mark.parametrize('field,value',[
    ('before','Invented prior state'),('pending_actor_ids',['unknown']),('event_refs',['not_supplied']),
    ('evidence_quote','An unsupported invented statement.'),('cause_action_ids',['private_action']),('issue_id','not_frozen')])
def test_issue_validation_rejects_unproven_changes(field,value):
    result=GameMasterSettlement(state_summary='Proposal',next_world_state={},analysis='',
        public_events=[event()],issue_updates=[change(**{field:value})])
    with pytest.raises(ValueError):validate_issue_updates(snapshot(),result,[])


def test_pending_response_triggers_the_recipient_and_can_be_resolved():
    s=snapshot();proposal=action(decision='act');update=change(cause_action_ids=[proposal.action_id])
    result=GameMasterSettlement(state_summary='Proposal',next_world_state=s.world_state,analysis='',public_events=[event()],issue_updates=[update])
    validate_issue_updates(s,result,[proposal]);after=apply_round_update(s,[proposal],result,RelationshipProjection())
    for a in after.agents:a.activation.last_deliberation_round=1
    seeds=_seed_candidates(after,[])
    assert seeds['board']['mandatory'] and 'pending_public_response' in seeds['board']['sources']
    assert not seeds['actor_a']['mandatory']
    reply=IssueUpdate(issue_id='delivery',title='Delivery',before=update.after,after='The delivery plan is approved.',
        kind='response',state_code='accepted',actor_ids=['board'],event_refs=['sim_2_1'],
        evidence_quote='The delivery plan is approved.',pending_actor_ids=[])
    e=event('The delivery plan is approved.').model_copy(update={'timeline_id':'sim_2_1'})
    settlement=GameMasterSettlement(state_summary='Approved',next_world_state=s.world_state,analysis='',public_events=[e],issue_updates=[reply])
    validate_issue_updates(after,settlement,[]);final=apply_round_update(after,[],settlement,RelationshipProjection())
    assert final.process_issues[0].pending_actor_ids==[] and final.process_issues[0].state=='The delivery plan is approved.'
    assert not _seed_candidates(final,[])['board']['mandatory'] and not s.process_issues


def test_idle_narrative_is_not_new_but_delayed_effects_and_observations_can_be():
    s=snapshot();result=GameMasterSettlement(state_summary='Unchanged',next_world_state={},analysis='',public_events=[event()])
    suppress_continuations(s,result,[])
    assert not result.public_events and result.continuing_state_notes
    s.agents[0].historical_actions=[{'action_id':'earlier_delivery'}]
    result=GameMasterSettlement(state_summary='Delivered',next_world_state={},analysis='',public_events=[event(cause_action_ids=['earlier_delivery'])])
    suppress_continuations(s,result,[]);assert result.public_events
    s.world_state['program_observations']=[{'step':1,'description':'An externally configured delivery arrived.'}]
    result=GameMasterSettlement(state_summary='Delivered',next_world_state={},analysis='',public_events=[event()])
    suppress_continuations(s,result,[]);assert result.public_events


def test_explicit_continuation_is_not_a_new_event_even_if_another_actor_acts():
    s=snapshot();result=GameMasterSettlement(state_summary='Unchanged',next_world_state={},analysis='',public_events=[event(change_kind='continuation')])
    suppress_continuations(s,result,[action(decision='act')]);assert not result.public_events


def test_process_projection_preserves_records_and_excludes_private_data_and_failed_denominators():
    r={'sample_id':1,'status':'completed','classification':{'category_id':'open'},
       'snapshot':{'agents':[{'private_observations':['SECRET']}]},'rounds':[{'step':1,'timeline':[event().model_dump()],
       'agent_actions':[{'decision':'act','visibility':'secret','action':'SECRET'}]}]}
    failed=deepcopy(r);failed.update(sample_id=2,status='failed');original=deepcopy([r,failed])
    summary=process_summary([r,failed],2)
    assert [r,failed]==original and 'SECRET' not in str(summary)
    assert summary['completed_samples']==1 and summary['paths'][0]['count']==1
    assert summary['paths'][0]['steps'][0]['action_count']==0 and summary['mode']=='recorded_public_events'


def test_conditional_and_accepted_states_do_not_merge():
    records=[{'sample_id':i,'status':'completed','rounds':[{'step':1,'issue_updates':[change(state_code=code).model_dump()]}]}
             for i,code in enumerate(['conditional','accepted'],1)]
    assert process_summary(records,2)['path_count']==2


def test_public_issue_reconvergence_requires_structured_divergence_and_later_exact_match():
    def record(sample_id, first, second, selected):
        return {'sample_id': sample_id, 'status': 'completed', 'classification': {'category_id': 'same'},
                'rounds': [{'step': 1, 'issue_updates': [first.model_dump()], 'agent_actions': [selected.model_dump()]},
                           {'step': 2, 'issue_updates': [second.model_dump()] if second else []}]}

    proposed = change(state_code='proposed')
    accepted = change(state_code='accepted', after='The board approved the delivery.', pending_actor_ids=[])
    first = record(1, proposed, accepted, action(decision='act'))
    second = record(2, accepted, None, action())
    assert public_issue_reconvergence([first, second]) == {
        'public_issue_divergence_pairs': 1, 'public_issue_reconvergence_pairs': 1,
    }
    second['rounds'][0]['issue_updates'][0]['after'] = 'The board approved a different delivery.'
    assert public_issue_reconvergence([first, second]) == {
        'public_issue_divergence_pairs': 1, 'public_issue_reconvergence_pairs': 0,
    }
    second['rounds'][0]['issue_updates'][0]['state_code'] = 'unknown'
    assert public_issue_reconvergence([first, second]) == {
        'public_issue_divergence_pairs': 0, 'public_issue_reconvergence_pairs': 0,
    }
    second['rounds'][0]['issue_updates'] = [accepted.model_dump()]
    second['rounds'][0]['agent_actions'][0]['visibility'] = 'direct'
    assert public_issue_reconvergence([first, second]) == {
        'public_issue_divergence_pairs': 0, 'public_issue_reconvergence_pairs': 0,
    }


def test_known_goal_id_cannot_apply_status_to_a_replacement_goal():
    a=actor();a.desires=[Desire(desire_id='g',description='Maintain supply')];issues=[]
    apply_goal_statuses(a,[Desire(desire_id='g',description='An unrelated replacement goal',status='satisfied')],issues,2)
    assert a.desires[0].status=='active' and issues[0]['reason']=='goal_description_changed'


def test_actual_settlement_agent_repairs_only_its_invalid_issue_payload():
    from app.functional_agents.simulation import GameMasterAgent
    s=snapshot();proposal=action(decision='act');calls=[]
    original=proposal.model_dump()
    class Model:
        async def json_call(self,**kw):
            calls.append(kw)
            assert kw['schema'] is GameMasterSettlement
            assert 'process_topics' in kw['prompt']
            return GameMasterSettlement(state_summary='Awaiting response',next_world_state=s.world_state,analysis='A new public proposal',
                public_events=[event()],issue_updates=[change(before='Invented prior state' if len(calls)==1 else '',cause_action_ids=[proposal.action_id])],
                agent_knowledge_updates=[{'agent_id':a.agent_id,'information':[]} for a in s.agents])
    result=asyncio.run(GameMasterAgent(Model()).settle(s,[proposal],[],'fixture'))
    assert len(calls)==2 and calls[1]['call_name'].endswith('settlement_repair')
    assert result.issue_updates[0].before=='' and proposal.model_dump()==original
    assert 'pending_public_response' in _seed_candidates(apply_round_update(s,[proposal],result,RelationshipProjection()),[])['board']['sources']


def test_weighted_settlement_passes_observed_branch_state_to_the_next_round():
    from app.functional_agents.simulation import GameMasterAgent

    class Model:
        async def json_call(self, **kwargs):
            assert 'only the selected action' in kwargs['system']
            assert 'legitimately produce no new public effect' in kwargs['system']
            kwargs_action = kwargs['prompt'].split('\nAUDITED ACTIONS\n', 1)[1].split('\n\n', 1)[0]
            import json
            [selected] = json.loads(kwargs_action)
            proposed = selected['decision'] == 'act'
            return GameMasterSettlement(
                state_summary='Awaiting response' if proposed else 'No change',
                next_world_state=s.world_state,
                analysis='Only the selected action has been resolved.',
                public_events=[event(cause_action_ids=[selected['action_id']])] if proposed else [],
                issue_updates=[change(cause_action_ids=[selected['action_id']])] if proposed else [],
                agent_knowledge_updates=[{'agent_id': a.agent_id, 'information': []} for a in s.agents],
            )

    s = snapshot()
    master = GameMasterAgent(Model())
    master.weighted_decisions = True
    proposed, waited = [action(decision='act')], [action()]
    proposal_result = asyncio.run(master.settle(s, proposed, [], 'proposal'))
    wait_result = asyncio.run(master.settle(s, waited, [], 'wait'))
    proposal_state = apply_round_update(s, proposed, proposal_result, RelationshipProjection())
    wait_state = apply_round_update(s, waited, wait_result, RelationshipProjection())
    assert proposal_state.process_issues[0].pending_actor_ids == ['board']
    assert 'pending_public_response' in _seed_candidates(proposal_state, [])['board']['sources']
    assert not wait_state.process_issues and not wait_result.public_events


def test_private_cause_is_redacted_without_discarding_a_public_consequence():
    from app.functional_agents.simulation import GameMasterAgent
    s=snapshot();proposal=action(decision='act').model_copy(update={'visibility':'direct'})
    class Model:
        calls=0
        async def json_call(self,**kw):
            self.calls+=1
            return GameMasterSettlement(state_summary='Public proposal received',next_world_state=s.world_state,analysis='Settled',
                public_events=[event(cause_action_ids=[proposal.action_id])],issue_updates=[change(cause_action_ids=[proposal.action_id])],
                agent_knowledge_updates=[{'agent_id':a.agent_id,'information':[]} for a in s.agents])
    model=Model();result=asyncio.run(GameMasterAgent(model).settle(s,[proposal],[],'fixture'))
    assert model.calls==1 and len(result.issue_updates)==1 and len(result.public_events)==1
    assert result.issue_updates[0].cause_action_ids==[] and result.public_events[0].cause_action_ids==[]
    assert proposal.visibility=='direct'


def test_invalid_optional_issue_does_not_discard_a_valid_core_settlement():
    from app.functional_agents.simulation import GameMasterAgent
    s=snapshot();proposal=action(decision='act')
    class Model:
        calls=0
        async def json_call(self,**kw):
            self.calls+=1
            return GameMasterSettlement(state_summary='Public proposal received',next_world_state=s.world_state,analysis='Settled',
                public_events=[event()],issue_updates=[change(cause_action_ids=['invented_action'])],
                agent_knowledge_updates=[{'agent_id':a.agent_id,'information':[]} for a in s.agents])
    model=Model();result=asyncio.run(GameMasterAgent(model).settle(s,[proposal],[],'fixture'))
    assert model.calls==2 and not result.issue_updates and result.public_events
    after=apply_round_update(s,[proposal],result,RelationshipProjection())
    assert after.round_id==1 and after.timeline and not after.process_issues
    assert after.validation_issues[-1]['code']=='issue_update_rejected'


def test_isolating_optional_issues_never_bypasses_invalid_core_settlement():
    from app.functional_agents.simulation import GameMasterAgent
    s=snapshot()
    class Model:
        calls=0
        async def json_call(self,**kw):
            self.calls+=1
            return GameMasterSettlement(state_summary='Invalid',next_world_state=s.world_state,analysis='Missing recipients',
                public_events=[event()],issue_updates=[change(cause_action_ids=['bad'])],agent_knowledge_updates=[])
    model=Model()
    with pytest.raises(ValueError):asyncio.run(GameMasterAgent(model).settle(s,[action(decision='act')],[],'fixture'))
    assert model.calls==2


def test_invalid_unchanged_annotation_is_skipped_without_overwriting_previous_state():
    from app.simulation.process import retain_valid_issue_updates
    s=snapshot()
    result=GameMasterSettlement(state_summary='Settled',next_world_state=s.world_state,analysis='',public_events=[event()],
        issue_updates=[change(kind='unchanged',before='',after='A new untracked state')])
    retain_valid_issue_updates(s,result,[])
    assert not result.issue_updates and result._process_diagnostics
    assert not s.process_issues
