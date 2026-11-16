"""Narrative briefing provenance and bounded source coverage, no provider calls."""
import asyncio
import pytest
from app.knowledge.brief import NarrativeBrief, brief_catalog, narrative_document
from app.knowledge.sentences import sentence_records
from app.storage.json_store import EventRepository
from app.domain import SummaryRequest


def fixture():
    articles=[{'title':'Supply disruption','url':'https://example.com/source',
        'content':'The port closed after a storm in October 2026. The operator announced repairs and a temporary route.'}]
    rows=sentence_records(articles[0]['content'])
    brief=NarrativeBrief(paragraphs=[
        {'heading':'事件概况','text':'据所提供的材料，该港口在2026年10月的一场风暴后关闭。',
         'references':[{'source_index':1,'sentence_ids':[rows[0]['id']]}]},
        {'heading':'应对进展','text':'运营方随后宣布开展维修，并公布一条临时运输路线作为应对措施。',
         'references':[{'source_index':1,'sentence_ids':[rows[1]['id'],rows[1]['id']]}]},
    ])
    return articles,brief


def test_narrative_keeps_connected_text_and_complete_deduplicated_provenance():
    articles,brief=fixture()
    doc=narrative_document(brief,articles,brief_catalog(articles),'额外需求')
    assert doc['version']=='background-brief-2'
    assert doc['paragraphs'][0]['text'] in doc['text']
    assert len(doc['paragraphs'][1]['references'])==1
    assert doc['paragraphs'][0]['references'][0]['quote'].endswith('.')
    assert 'User-Provided Context (Modeling Assumptions)' in doc['text']
    assert articles[0]['content'] not in doc['text']


def test_invented_or_uncatalogued_citation_rejects_the_whole_narrative():
    articles,brief=fixture()
    brief.paragraphs[0].references[0].sentence_ids=['s999_999']
    with pytest.raises(ValueError,match='missing or incomplete source sentence'):
        narrative_document(brief,articles,brief_catalog(articles))
    brief.paragraphs[0].references[0].source_index=2
    with pytest.raises(ValueError,match='unavailable source'):
        narrative_document(brief,articles,brief_catalog(articles))


def test_catalog_has_late_body_coverage_and_no_clipped_sentence():
    body=' '.join(f'Observed development number {i} involved a complete source statement.' for i in range(400))
    body+=' This ending was cut off'
    catalog=brief_catalog([{'content':body}])[0]['sentences']
    assert sum(len(s['quote']) for s in catalog)<=12000
    assert any('number 399 ' in s['quote'] for s in catalog)
    assert all(s['quote'].endswith('.') for s in catalog)
    assert all('cut off' not in s['quote'] for s in catalog)


def test_combined_summary_uses_one_call_and_saves_a_citable_document(tmp_path,monkeypatch):
    from app.knowledge import operations
    articles,brief=fixture()
    repo=EventRepository(tmp_path)
    event=repo.create('Supply disruption',[], '用户设定')
    repo.save_sources(event['event_id'],articles)
    calls=[]
    class Client:
        async def json_call(self,**kwargs):
            calls.append(kwargs)
            assert kwargs['schema'] is NarrativeBrief
            return brief
    monkeypatch.setattr(operations,'EventRepository',lambda:repo)
    monkeypatch.setattr(operations,'make_client',lambda *args:(Client(),object()))
    monkeypatch.setattr(operations,'audit_finish',lambda *args,**kwargs:None)
    result=asyncio.run(operations.combine_summary(SummaryRequest(event_id=event['event_id'],event_title='Supply',article_summaries=[])))
    assert len(calls)==1
    saved=repo.load_artifact(event['event_id'],'background_brief',{})
    assert result==saved['text']==repo.load_artifact(event['event_id'],'background','')
    assert saved['additional_context']=='用户设定'
    assert saved['paragraphs'][0]['references'][0]['url']=='https://example.com/source'


def test_forecast_brief_keeps_decision_context_and_excludes_foreign_examples(tmp_path, monkeypatch):
    import json

    from app.knowledge import operations
    from app.knowledge.brief import forecast_brief_preview

    articles = [
        {"title": "Official bank rate", "url": "https://en.wikipedia.org/wiki/Official_bank_rate",
         "content": "The central bank of another country lowered its interest rate after an announcement."},
        {"title": "Example Central Bank", "url": "https://en.wikipedia.org/wiki/Example_Central_Bank",
         "revision_url": "https://en.wikipedia.org/w/index.php?oldid=123",
         "content": "== Authority ==\nThe Example Central Bank sets the interest rate independently. "
                    "The finance minister may issue a directive only under the legal framework."},
    ]
    question = "Will the Example Central Bank change its interest rate?"
    spec = {"question": question, "as_of": "2026-11-09T12:00:00Z",
            "outcome_deadline": "2026-11-12T12:00:00Z", "yes_criterion": "An official rate change.",
            "no_criterion": "No official rate change."}
    repo = EventRepository(tmp_path)
    case = repo.create(question, [], None)["event_id"]
    repo.save_sources(case, articles)
    repo.save_artifact(case, "forecast_spec", spec)
    preview = forecast_brief_preview(spec, articles)
    assert len(preview["paragraphs"]) == 1 and preview["paragraphs"][0]["heading"] == "Evidence preview"
    assert "==" not in preview["text"] and "another country" not in preview["text"]

    class Client:
        async def json_call(self, **kwargs):
            data = json.loads(kwargs["prompt"])
            assert data["evidence_cutoff"] == spec["as_of"]
            source, = data["source_sentences"]
            assert source["source_index"] == 2
            first, second = source["sentences"]
            assert first["quote"] == "The Example Central Bank sets the interest rate independently."
            return NarrativeBrief(paragraphs=[
                {"heading": "The decision", "text": "The central bank controls the interest rate independently.",
                 "references": [{"source_index": 2, "sentence_ids": [first["id"]]}]},
                {"heading": "Exceptional authority", "text": "The minister's directive power is subject to the legal framework.",
                 "references": [{"source_index": 2, "sentence_ids": [second["id"]]}]},
            ], evidence_gaps=["The sources do not give the next announcement date."])

    monkeypatch.setattr(operations, "make_client", lambda *args: (Client(), object()))
    monkeypatch.setattr(operations, "audit_finish", lambda *args, **kwargs: None)
    asyncio.run(operations.combine_summary(SummaryRequest(
        event_id=case, event_title=question, article_summaries=[],
    ), events=repo))
    result = repo.load_artifact(case, "background_brief", {})
    assert result["version"] == "forecast-background-1"
    assert len(result["paragraphs"]) == 1 and "\n" not in result["text"]
    assert result["paragraphs"][0]["references"][0]["url"] == articles[1]["revision_url"]
    assert result["evidence_gaps"] == ["The sources do not give the next announcement date."]
