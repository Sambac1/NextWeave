"""Cross-domain observation, stage distinctions, privacy, and sample counting."""
from copy import deepcopy
from app.simulation.activities import observations,activity_frequencies,annotate_events
from app.domain.world import TimelineEvent


def stages(text):
    return {(r['family'],r['stage']) for r in observations({'description':text})}


def record(number,text,status='completed'):
    return {'sample_id':number,'status':status,'rounds':[{'step':1,'timeline':[
        {'timeline_id':f'sim_{number}_1','description':text,'actors':['company','board']}]}]}


def test_business_proposal_does_not_count_as_delivery_or_approval():
    found=stages('The company proposed to complete delivery if the board approved the budget.')
    assert ('proposal','proposed') in found
    assert not any(stage in {'accepted','completed'} for _,stage in found)


def test_project_approval_and_completion_are_separate_realized_events():
    found=stages('The board approved the investment. The operator completed delivery.')
    assert ('agreement','accepted') in found
    assert ('outcome','completed') in found
    found=stages('董事会批准了投资。运营方完成交付。')
    assert ('agreement','accepted') in found and ('outcome','completed') in found


def test_conditional_acceptance_is_not_unconditional_agreement():
    found=stages('The company accepted the proposal in principle, subject to financing.')
    assert ('agreement','conditional') in found
    assert ('agreement','accepted') not in found
    assert ('outcome','completed') not in found
    assert ('agreement','accepted') not in stages('The firm proposed a price above an agreed low threshold.')


def test_negations_and_hypothetical_outcomes_do_not_count_as_completion():
    assert ('outcome','completed') not in stages('The project was not completed.')
    assert ('outcome','completed') not in stages('该项目尚未完成。')
    assert ('outcome','completed') not in stages('The firm promised the project would be completed next month.')


def test_each_sample_counts_once_and_partial_paths_stay_out_of_denominator():
    a=record(1,'The company submitted a proposal.')
    a['rounds'].append({'step':2,'timeline':[{'timeline_id':'sim_1_2','description':'The company submitted another proposal.'}]})
    b=record(2,'The company completed delivery.')
    c=record(3,'The company submitted a proposal.',status='failed')
    before=deepcopy([a,b,c])
    result=activity_frequencies([a,b,c],4)
    proposal=next(g for g in result['groups'] if g['family']=='proposal')
    assert proposal['count']==1 and proposal['frequency']==.5
    assert proposal['sample_ids']==[1] and proposal['evidence_count']==2
    assert result['unfinished_samples']==2
    assert [a,b,c]==before


def test_private_intentions_and_round_summary_never_enter_observations():
    r=record(1,'No public changes.')
    r['rounds'][0]['state_summary']='The company completed delivery.'
    r['rounds'][0]['agent_actions']=[{'action':'The company completed delivery.'}]
    r['snapshot']={'agents':[{'private_observations':['The company completed delivery.']}]}
    result=activity_frequencies([r],1)
    assert not result['groups']


def test_only_exact_public_event_quotes_can_support_model_annotations():
    event={'description':'The board authorized the agreement.', 'activity_annotations':[
        {'family':'agreement','stage':'accepted','quote':'The board authorized the agreement.'},
        {'family':'outcome','stage':'completed','quote':'The company completed delivery.'}]}
    found=observations(event)
    assert len(found)==1 and found[0]['method']=='settlement_annotation'
    assert found[0]['stage']=='accepted'


def test_structure_rule_provenance_survives_snapshot_roundtrip():
    event=TimelineEvent(timeline_id='e',timestamp='2026-11',description='The operator completed delivery.',confidence=1,importance=1)
    annotate_events([event])
    restored=TimelineEvent.model_validate(event.model_dump())
    assert restored.activity_annotations[0].method=='legacy_structure_rule'
    assert observations(restored.model_dump())[0]['method']=='legacy_structure_rule'
