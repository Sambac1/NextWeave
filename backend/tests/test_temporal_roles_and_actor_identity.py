"""Cross-domain regressions for date binding and independent decision identities."""
import asyncio
import json

import pytest

from app.knowledge.evidence import validate_event
from app.knowledge.actor_identity import guard_alias_mapping, equivalent_names, delegated_seat_supported
from app.layers.pipeline import llm_figure_service
from app.layers.pipeline.helpers import _merge_canonical_figures


@pytest.mark.parametrize('text,description,date', [
    ('On September 2, 2026, President Donald Trump signed the funding bill into law, extending federal funding through December 11 and preventing the government shutdown that would have begun on October 1.',
     'President Donald Trump signed the funding bill into law.', '2026-09-02'),
    ('On April 3, 2025, Northstar signed the supply agreement, extending delivery access through July 1 and preventing an interruption that would begin on May 1.',
     'Northstar signed the supply agreement.', '2025-04-03'),
    ('On February 2, 2025, the authority issued the operating permit, allowing service until June 1.',
     'The authority issued the operating permit.', '2025-02-02'),
    ('On August 1, 2024, the commission approved the count, opening an appeal period through September 1.',
     'The commission approved the count.', '2024-08-01'),
])
def test_occurrence_date_survives_validity_and_hypothetical_dates(text, description, date):
    raw = {'description': description, 'timestamp': date, 'date_original': date,
           'evidence_text': text, 'time_role': 'occurrence', 'modality': 'occurred'}
    result, reason = validate_event(raw, {'content':text,'path':'source/History','source_title':'Case'})
    assert reason == 'accepted' and result['timestamp'] == date
    clause = result['date_evidence']['occurrence_clause']
    assert text[clause['start']:clause['end']] == clause['quote']
    assert result['date_evidence']['supported_dates'] == [date]


@pytest.mark.parametrize('text,description', [
    ('On March 1, 2025, the company plans to sign a contract that takes effect on June 1.', 'The company signed a contract.'),
    ('On March 1, 2025, the company opened a factory; on June 1, the regulator closed a laboratory.', 'The regulator closed the laboratory.'),
    ('On March 1, 2025, the board approved the proposal, while the director signed it on June 1.', 'The director signed the proposal.'),
])
def test_date_of_another_action_is_not_borrowed(text, description):
    result, _ = validate_event({'description':description,'timestamp':'2025-03-01',
                               'date_original':'March 1, 2025','evidence_text':text},
                              {'content':text,'path':'source/History'})
    assert result is None


def figure(name, entity_type='organization'):
    return {'name':name, 'entity_type':entity_type, 'affiliation':'Common parent', 'confidence':.9}


def test_parent_org_and_independent_units_do_not_become_aliases():
    names = ['United States federal government', 'Congress', 'Office of Management and Budget',
             'Northstar Company', 'Northstar Board', 'Northstar Audit Committee']
    mapping = {'Congress':names[0], 'Office of Management and Budget':names[0],
               'Northstar Board':'Northstar Company', 'Northstar Audit Committee':'Northstar Company'}
    accepted, review = guard_alias_mapping(mapping, [figure(n) for n in names])
    assert not accepted and len(review['rejected']) == 4
    assert len(_merge_canonical_figures([figure(n) for n in names], accepted)) == len(names)


def test_genuine_case_acronym_and_source_identity_variants_still_merge():
    names = ['Research Council', 'research council', 'Research Council (RC)', 'RC',
             'Office of Management and Budget', 'OMB', 'Tatmadaw', 'Myanmar Armed Forces']
    mapping = {'research council':'Research Council', 'RC':'Research Council (RC)',
               'OMB':'Office of Management and Budget', 'Tatmadaw':'Myanmar Armed Forces'}
    accepted, review = guard_alias_mapping(mapping, [figure(n) for n in names], timeline=[
        {'evidence_text':'Tatmadaw, also known as Myanmar Armed Forces, issued a statement.'}])
    assert accepted == mapping and not review['rejected']
    assert not equivalent_names('Research Council (East)', 'Research Council')


def test_ambiguous_acronym_and_unseen_or_institutional_aliases_are_rejected():
    names = ['Federal Review Board', 'Finance Reform Bureau', 'FRB']
    accepted, review = guard_alias_mapping({'FRB':'Federal Review Board', 'Unseen':'FRB'}, [figure(n) for n in names])
    assert not accepted and len(review['rejected']) == 2
    assert not equivalent_names('Research Council','Northstar Company', evidence=[
        'Research Council negotiates with Northstar Company on behalf of its parent.'])


def test_person_titles_and_unique_surnames_remain_aliases_but_ambiguous_names_do_not():
    people = [figure(n,'person') for n in ['Dr Ada Chen','Ada Chen','Chen']]
    accepted, _ = guard_alias_mapping({'Dr Ada Chen':'Ada Chen','Chen':'Ada Chen'}, people)
    assert len(accepted) == 2
    accepted, _ = guard_alias_mapping({'Chen':'Ada Chen'}, people + [figure('Lin Chen','person')])
    assert not accepted


def test_category_compaction_does_not_fold_independent_institutions():
    from app.layers.pipeline.helpers import _compact_key_figure_names
    rows = [figure(name) | {'canonical_name':name,'parent_actor':'Common parent',
                           'aggregation_category':'parent_state_subunit'}
            for name in ['Northstar Board','Northstar Audit Committee']]
    assert _compact_key_figure_names([row['name'] for row in rows], rows, 8) == [row['name'] for row in rows]
    assert not any(row.get('aggregated_members') for row in rows)


def test_overlap_requires_evidence_of_pure_delegation_not_affiliation():
    quote = 'Delivery Unit solely executes Northstar Board decisions without independent discretion.'
    records = [{'history':[{'description':quote}]}]
    pair = {'contained':'Delivery Unit','container':'Northstar Board','decision_class':'Transmit the approved instruction'}
    assert not delegated_seat_supported(pair, records)
    assert delegated_seat_supported(pair | {'delegation_quote':quote}, records)
    assert not delegated_seat_supported(pair | {'delegation_quote':'Both bodies share a parent and prefer the same result.'}, records)


def test_actual_alias_merge_rejects_bad_model_mapping_without_changing_actions(monkeypatch):
    names = ['Northstar Company','Northstar Board','Office of Management and Budget','OMB']
    actions = [{'source':'Northstar Board','target':'Northstar Company','action':'Approve a proposal','confidence':.9}]

    async def provider(*args, **kwargs):
        return json.dumps({'name_mapping':{'Northstar Board':'Northstar Company', 'OMB':'Office of Management and Budget'}}), {}

    monkeypatch.setattr(llm_figure_service,'_call_openrouter_with_meta',provider)
    result = asyncio.run(llm_figure_service.merge_figures('fixture','fixture','Case',
        [[figure(n) for n in names]],[actions]))
    assert result['name_mapping'] == {'OMB':'Office of Management and Budget'}
    assert result['actions'][0]['source'] == 'Northstar Board'
    assert result['alias_review']['rejected'][0]['alias'] == 'Northstar Board'


def test_overlap_call_does_not_demote_independent_body_without_delegation(monkeypatch):
    async def provider(*args, **kwargs):
        return json.dumps({'overlaps':[{'container':'Northstar Company','contained':'Northstar Board',
                                      'reason':'Same organization'}]})
    monkeypatch.setattr(llm_figure_service,'_call_openrouter',provider)
    result = asyncio.run(llm_figure_service.check_principal_overlaps('fixture','fixture','Case',
        ['Northstar Company','Northstar Board'], [figure('Northstar Company'),figure('Northstar Board')]))
    assert result == []


def test_source_authority_retains_sparse_institutions_without_fabricating_timeline_counts():
    from app.knowledge.actor_evidence import attach_authority_evidence
    from app.layers.pipeline.helpers import _filter_by_min_evidence
    text = ('The charter vests the U.N. Council with the authority to approve funding. '
            'Delivery Office (DO) can direct operating teams. '
            'Acme Board has the authority to appoint directors. '
            'The Regulator is responsible for certifying results.')
    names = ['Council','Delivery Office','Acme Board','Regulator','Affected workers']
    rows = [figure(name) | {'canonical_name':name, 'evidence_count':1} for name in names]
    source = {'content':text,'title':'Institutional rules','revision_at':'2025-01-01T00:00:00Z'}
    attach_authority_evidence(rows, sources=[source], cutoff='2025-02-01')
    kept, removed, _ = _filter_by_min_evidence(rows, .025, 2, 200, 1)
    assert {row['name'] for row in kept} == set(names[:-1])
    assert [row['name'] for row in removed] == ['Affected workers']
    assert all(row['evidence_count'] == 1 for row in rows)
    for row in kept:
        for hit in row['source_authority_evidence']:
            assert text[hit['span_start']:hit['span_end']] == hit['text']


@pytest.mark.parametrize('text', [
    'The Budget Office was criticized for its delayed approval request.',
    'Budget Office could approve requests in 1980.',
    'Budget Office said the Council can approve requests.',
    'The Council can approve a Budget Office request.',
    'Budget Office can no longer approve requests.',
    'Budget Office had the authority to approve requests until 1990.',
])
def test_source_mentions_do_not_become_authority(text):
    from app.knowledge.actor_evidence import attach_authority_evidence
    rows = [figure('Budget Office')]
    attach_authority_evidence(rows, sources=[{'content':text,'title':'Rules'}])
    assert not rows[0].get('source_authority_evidence')


def test_future_source_cannot_retain_candidate_and_changed_evidence_clears_annotation():
    from app.knowledge.actor_evidence import attach_authority_evidence
    rows = [figure('Council')]
    source = {'content':'Council can approve funding requests.', 'revision_at':'2025-03-01T00:00:00Z'}
    attach_authority_evidence(rows, sources=[source], cutoff='2025-04-01')
    assert rows[0]['source_authority_evidence']
    attach_authority_evidence(rows, sources=[source], cutoff='2025-02-01')
    assert not rows[0].get('source_authority_evidence')


def test_principal_selector_receives_literal_authority_not_just_mention_counts(monkeypatch):
    captured = []
    async def provider(*args, **kwargs):
        captured.append(args[3])
        return json.dumps({'principals':[{'name':'Council'}]})
    monkeypatch.setattr(llm_figure_service,'_call_openrouter',provider)
    records = [figure('Council') | {'canonical_name':'Council', 'evidence_count':1,
        'source_authority_evidence':[{'evidence_id':'span1','text':'Council can approve funding.'}]}]
    selected = asyncio.run(llm_figure_service.select_principals('fixture','fixture','Case',records))
    assert selected == ['Council']
    assert 'Council can approve funding.' in captured[0]
    assert 'not proof of current office' in captured[0]
