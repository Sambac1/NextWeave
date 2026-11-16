"""Evidence parity, source scope and factual interpretation; no provider calls."""
import asyncio
from copy import deepcopy
import json

from app.benchmark.dynamics import contract_state
from app.benchmark.runner import forecast_inputs, _passages
from app.domain import ActionProposal, ForecastRound, WorldSnapshot
from app.domain.forecast import ContractOutlook
from app.knowledge.public_evidence import public_sources
from app.knowledge.table_evidence import parse_table_evidence, table_passages
from app.simulation.context import public_context, information_context, cited_public_context, actor_context
from app.simulation.outlook import PublicReadout, evidence_catalog, read_public_outlook
from app.simulation.readout_grounding import ReadoutGrounding
from app.storage.round_stages import RoundStages
from test_ensemble import actor


TABLE = '''{| class="wikitable"
! rowspan="2" | Entity
! colspan="2" | Member resources
! colspan="2" | Group resources
|-
! Owned (units) !! Available (units) !! Total (units) !! Difference (units)
|-
| Agency A || 12 || 8 || 200 || 40
|-
| Agency B || 15 || 9 || 200 || 40
|}'''


def sources():
    return [{"url": f"https://en.wikipedia.org/wiki/Source_{i}", "title": f"Source {i}",
             "revision_id": i + 10, "revision_at": "2025-01-01T00:00:00Z",
             "content": (TABLE if i == 0 else (f"Background {i}. " * 230) + f"Required fact {i}.")}
            for i in range(6)]


def case():
    return {"question": "Will Agency A deliver the agreed resources?", "rules": "Terms. " * 400 + "Final approval is required.",
            "as_of": "2025-01-02T00:00:00Z", "outcome_deadline": "2025-02-01T00:00:00Z"}


def test_all_roles_receive_the_same_complete_selected_sources_and_rules():
    original = sources()
    qa = forecast_inputs(case(), original)
    contract = contract_state(case(), original, include_projection=False)
    assert qa['evidence_packet'] == contract['evidence_packet']
    assert len(contract['historical_evidence']) == 6
    assert [row['excerpt'] for row in contract['historical_evidence']] == [row['content'] for row in qa['sources']]
    assert contract['rules'] == qa['rules'] == case()['rules']
    a = actor()
    snap = WorldSnapshot(event_title=case()['question'], agents=[a], world_state={'benchmark_contract': contract})
    before = deepcopy(snap.model_dump())
    proposal = ActionProposal(agent_id=a.agent_id, agent_name=a.name, round_id=1, snapshot_version=0,
                              action_id='wait_1', decision='wait', action_type='no_action', action='Wait', reasoning='Review', confidence=.5)
    contexts = [public_context(snap, 'activation'), public_context(snap, 'decision', a),
                information_context(snap, a)['public_state'], cited_public_context(snap, 'settlement', [proposal])]
    for context in contexts:
        assert context['world_state']['benchmark_contract']['historical_evidence'] == contract['historical_evidence']
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1, state_summary='Review')
    catalog = evidence_catalog(snap, frame)
    assert {row['content'] for row in qa['sources']} <= {row.text for row in catalog.values()}
    assert snap.model_dump() == before and original == sources()


def test_grouped_headers_and_inline_cells_preserve_entity_metric_semantics():
    table, = parse_table_evidence(TABLE)
    first = {tuple(field['column_path']): field['value'] for field in table['rows'][0]['fields']}
    assert first[('Member resources', 'Available (units)')] == '8'
    assert first[('Group resources', 'Difference (units)')] == '40'
    assert table['rows'][0]['original_quote'] in TABLE
    assert not table['ambiguity']
    chunks = list(_passages(TABLE))
    assert chunks == [TABLE]
    assert 'rowspan="2"' in chunks[0] and 'colspan="2"' in chunks[0]


def test_large_or_ambiguous_tables_are_not_silently_deleted_or_guessed():
    large = TABLE.replace('Agency B', 'Agency ' + 'x' * 2400)
    chunks = table_passages(large, max_chars=400)
    assert 'x' * 2400 in ''.join(chunks)
    assert all('Member resources' in chunk and 'Group resources' in chunk for chunk in chunks)
    ambiguous = '{|\n! Entity !! Units !! Units\n|-\n| A || 8 || 40\n|}'
    parsed, = parse_table_evidence(ambiguous)
    assert parsed['ambiguity'] and not any(field['value'] in {'8', '40'} for field in parsed['rows'][0]['fields'])


def test_preserved_header_attributes_do_not_break_existing_table_recall(monkeypatch):
    from app.benchmark.runner import frozen_sources
    from app.knowledge.retrieval import EvidenceIndex
    table = '{|\n! rowspan="1" |Fieldwork date\n! Result\n|-\n| 1 January || 7\n|}'
    c = case() | {'question': 'Will Agency A win the fourth most seats?',
                  'rules': 'Parliamentary elections are scheduled to be held on January 10, 2025.',
                  'evidence': [{'url': 'https://en.wikipedia.org/wiki/Example', 'title': 'Example',
                                'revision_id': 1, 'revision_at': '2025-01-01T00:00:00Z',
                                'text': 'An unrelated introductory paragraph.\n\n== 2025 ==\n' + table}]}
    monkeypatch.setattr(EvidenceIndex, 'search', lambda *args, **kwargs: [])
    selected = frozen_sources(c, 'expanded')
    assert any(table in source['content'] for source in selected)


def test_marked_profile_background_respects_cutoff_and_does_not_grant_authority():
    from app.knowledge.profile_evidence import public_background
    marked = public_sources(sources())
    future = marked[0] | {'revision_at': '2025-01-03T00:00:00Z'}
    unselected = {'content': 'An entire unselected article.', 'revision_at': '2025-01-01T00:00:00Z'}
    assert public_background([marked[0], future, unselected], cutoff='2025-01-02') == [marked[0]]
    from app.domain.decision_core import legacy_core
    a = actor()
    a.decision_core = legacy_core(a.model_dump())
    a.decision_core.evidence_gaps = ['No current resource data in the old dossier input.']
    context = actor_context(a)
    assert context['evidence_gap_scope']['status'] == 'unverified_against_current_evidence'
    assert 'not a prohibition' in context['evidence_gap_scope']['policy']
    assert a.decision_core.evidence_gaps == ['No current resource data in the old dossier input.']


def test_profile_generation_reads_shared_background_in_addition_to_actor_snippets(monkeypatch):
    from app.layers.pipeline import profile_service
    from test_compact_dossiers import core
    calls = []

    async def provider(api_key, model, prompt, **kwargs):
        data = json.loads(prompt.rsplit('\n', 1)[-1])
        calls.append(data)
        assert data['evidence']['shared_public_background'] == public_sources(sources())
        assert 'operational claims still need an attributable numbered T/S citation' in prompt
        return {'decision_core': core()}

    monkeypatch.setattr(profile_service, '_call_json_with_retries', provider)
    result = asyncio.run(profile_service.generate_profile(
        api_key='fixture', model='fixture', event_title=case()['question'], cutoff='2025-01-02',
        profile={'canonical_name': 'Laboratory', 'entity_type': 'organization',
                 'history': [{'timeline_id': 'h1', 'timestamp': '2025-01-01', 'description': 'A supply trial.'}]},
        source_records=public_sources(sources())))
    assert len(calls) == 1 and result['decision_core']['origin'] != 'minimal_fallback'


def test_table_row_retrieval_keeps_schema_and_exact_source_span_without_timeline(tmp_path):
    from app.knowledge.okf import KnowledgeStore
    from app.simulation.ensemble import attach_evidence
    from app.simulation.evidence import DecisionEvidenceRetriever
    from app.storage.json_store import EventRepository
    events = EventRepository(tmp_path)
    event_id = events.create('Resource agreement', [], None)['event_id']
    source = {'url': 'https://en.wikipedia.org/wiki/Resources', 'title': 'Resources',
              'content': '== Capacity ==\n' + TABLE, 'revision_id': 1, 'revision_at': '2025-01-01T00:00:00Z'}
    future = source | {'url': source['url'] + '_later', 'content': TABLE.replace('200', '9999'),
                       'revision_id': 2, 'revision_at': '2025-01-03T00:00:00Z'}
    events.save_sources(event_id, [source, future])
    knowledge = KnowledgeStore(events)
    release = knowledge.publish(event_id)
    _, documents = knowledge.snapshot(event_id, version=release['version'])
    a = actor().model_copy(update={'name': 'Agency A'})
    frozen = WorldSnapshot(event_title=case()['question'], agents=[a])
    retrieval = DecisionEvidenceRetriever(frozen, documents, cutoff_at='2025-01-02T00:00:00Z')
    working = frozen.model_copy(deep=True)
    attach_evidence(working, frozen, documents, 'Agency A available resources', retriever=retrieval)
    passages = public_context(working, 'decision', working.agents[0])['source_passages']
    assert passages and '9999' not in json.dumps(passages)
    for passage in passages:
        assert source['content'][passage['span_start']:passage['span_end']] == passage['quote']
        assert passage['kind'] == 'table_row'
        schema = dict(zip(map(tuple, passage['table_schema']['columns']), passage['table_schema']['values']))
        assert schema[('Group resources', 'Difference (units)')] == '40'
    assert not working.timeline  # A structural source is not a dated event.


def test_initial_belief_review_reconciles_packet_once_without_changing_other_actors(monkeypatch):
    from app.simulation.bdi_updates import process_agent_information, InformationReview
    contract = contract_state(case(), sources()[:1], include_projection=False)
    a = actor()
    snap = WorldSnapshot(event_title=case()['question'], agents=[a], world_state={'benchmark_contract': contract})
    calls = []

    class Runtime:
        async def json_call(self, *, schema, prompt, **kwargs):
            assert schema is InformationReview
            bundle = json.loads(prompt.split('\n', 1)[1].split('\n\nEcho', 1)[0])
            assert bundle['public_state']['world_state']['benchmark_contract']['historical_evidence'][0]['excerpt'] == TABLE
            calls.append(prompt)
            return schema(agent_id=a.agent_id, round_id=1, snapshot_version=0, assessment='Current packet reviewed.')

    async def check():
        reviewed = await process_agent_information(snap, Runtime(), 'fixture')
        assert len(calls) == 1 and reviewed.agents[0].last_information_fingerprint
        await process_agent_information(reviewed, Runtime(), 'fixture')
        assert len(calls) == 1
    asyncio.run(check())


def test_observer_rewording_does_not_trigger_bdi_but_a_public_event_does():
    from app.simulation.bdi_updates import process_agent_information, InformationReview
    from app.domain import TimelineEvent
    contract = contract_state(case(), sources()[:1], include_projection=False)
    snap = WorldSnapshot(event_title=case()['question'], agents=[actor()], world_state={
        'benchmark_contract': contract, 'simulation_clock': {'sequential_harness': 'fixture'}})
    calls = []

    class Runtime:
        async def json_call(self, *, schema, prompt, **kwargs):
            bundle = json.loads(prompt.split('\n', 1)[1].split('\n\nEcho', 1)[0])
            calls.append(bundle)
            public = bundle['public_state']
            return InformationReview(agent_id='actor_a', round_id=public['round_id'],
                                     snapshot_version=public['snapshot_version'], assessment='Evidence reviewed.')

    async def check():
        reviewed = await process_agent_information(snap, Runtime(), 'fixture')
        reviewed.round_id = 1
        reviewed.state_summary = 'The same unknown outcome was described again.'
        reviewed.world_state['contract_progress'] = 'Unknown, worded differently.'
        reviewed.world_state['contract_outlook'] = {'p_yes': .7}
        same = await process_agent_information(reviewed, Runtime(), 'fixture')
        assert len(calls) == 1
        same.round_id = 2
        same.timeline.append(TimelineEvent(timeline_id='new_public_fact', timestamp='2025-01-03',
            description='Agency A submitted its delivery plan.', origin='simulated', round_id=2,
            confidence=.8, importance=.8))
        await process_agent_information(same, Runtime(), 'fixture')
        assert len(calls) == 2
    asyncio.run(check())


def test_table_data_row_is_directly_citable_without_reconstructing_the_article():
    from app.simulation.outlook import validate_outlook
    contract = contract_state(case(), sources()[:1], include_projection=False)
    snap = WorldSnapshot(event_title=case()['question'], world_state={'benchmark_contract': contract})
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1, state_summary='Review')
    catalog = evidence_catalog(snap, frame)
    ref, row = next((ref, row) for ref, row in catalog.items() if row.kind == 'frozen_table_row')
    assert row.text in TABLE and row.table_schema['values'][0] == 'Agency A'
    frame.contract_outlook = ContractOutlook(p_yes=.4, basis='Member availability is separate from group resources.',
                                             evidence_ref=ref, evidence_quote=row.text)
    value, error = validate_outlook(snap, snap, frame, event_ids={}, catalog=catalog, terminal=True)
    assert error is None and value.evidence_quote == row.text


def test_terminal_table_interpretation_gets_one_targeted_repair_without_redrawing(tmp_path):
    contract = contract_state(case(), sources()[:1], include_projection=False)
    snap = WorldSnapshot(event_title=case()['question'], world_state={'benchmark_contract': contract})
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1, state_summary='Review')
    quote = 'Agency A || 12 || 8 || 200 || 40'
    calls = []

    class Runtime:
        async def json_call(self, *, schema, prompt, **kwargs):
            data = json.loads(prompt)
            calls.append(schema.__name__)
            if schema is PublicReadout:
                key = next(key for key, row in data['evidence'].items() if row.get('table_schema'))
                basis = 'Agency A owns 40 units.' if len(calls) == 1 else 'Agency A has 8 units available; group totals are separate.'
                if len(calls) > 1:
                    assert data['previous_validation_error'] == 'unsupported_interpretation'
                    assert data['interpretation_feedback'][0]['evidence_key'] == key
                return schema(outlook=ContractOutlook(p_yes=.4, basis=basis, evidence_ref=key, evidence_quote=quote))
            assert schema is ReadoutGrounding
            # The auditor never receives the proposed probability or a target answer.
            assert 'p_yes' not in data
            if len(calls) == 2:
                return schema(supported=False, issues=[{'claim': 'Agency A owns 40 units.',
                    'problem': '40 belongs to Group resources / Difference (units).', 'evidence_key': data['cited_evidence_key'],
                    'status': 'contradicted'}])
            return schema(supported=True)

    async def check():
        cache, runtime = RoundStages(tmp_path, {'model': 'fixture'}), Runtime()
        args = (snap, snap, frame, contract, {'step': 1, 'horizon_steps': 1}, runtime, cache, {}, 'fixture')
        value, error = await read_public_outlook(*args)
        assert error is None and value.p_yes == .4 and '8 units' in value.basis
        assert calls == ['PublicReadout', 'ReadoutGrounding', 'PublicReadout', 'ReadoutGrounding']
        assert (await read_public_outlook(*args))[0] == value
        assert len(calls) == 4
    asyncio.run(check())


def test_supported_or_uncertain_checks_cannot_be_an_inconsistent_overall_veto():
    verdict = ReadoutGrounding(supported=False, issues=[{'claim': 'Availability is 8-9 units.',
        'problem': 'Both values are within this range.', 'evidence_key': 'source', 'status': 'supported'}])
    assert verdict.supported and not verdict.issues
    verdict = ReadoutGrounding(supported=True, issues=[{'claim': 'Agency A owns the group total.',
        'problem': 'The total belongs to the group.', 'evidence_key': 'source', 'status': 'contradicted'}])
    assert not verdict.supported and len(verdict.issues) == 1


def test_long_explanation_gets_a_bounded_readout_repair_without_replaying_state(tmp_path):
    from app.llm.client import LLMError
    contract = {'question': 'Will funding lapse?', 'rules': 'An actual funding lapse is required.',
                'historical_evidence': [{'excerpt': 'A funding extension was enacted; a later deadline remains.'}]}
    snap = WorldSnapshot(event_title=contract['question'], world_state={'benchmark_contract': contract})
    frame = ForecastRound(step=1, input_snapshot_version=0, output_snapshot_version=1, state_summary='Unchanged')
    original = (snap.model_dump(), frame.model_dump())
    calls = []

    class Runtime:
        async def json_call(self, *, schema, prompt, **kwargs):
            calls.append(json.loads(prompt))
            if len(calls) == 1:
                raise LLMError('outlook.basis String should have at most 600 characters [type=string_too_long]')
            assert calls[-1]['previous_validation_error'] == 'basis_too_long'
            key, row = next(iter(calls[-1]['evidence'].items()))
            return schema(outlook=ContractOutlook(p_yes=.18, basis='The next deadline remains unresolved.',
                evidence_ref=key, evidence_quote=row['text']))

    async def check():
        result, issue = await read_public_outlook(snap, snap, frame, contract, {'step':1,'horizon_steps':1},
            Runtime(), RoundStages(tmp_path, {}), {}, 'fixture')
        assert result.p_yes == .18 and issue is None and len(calls) == 2
        assert (snap.model_dump(), frame.model_dump()) == original
    asyncio.run(check())
