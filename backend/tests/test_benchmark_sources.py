from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from app.benchmark.evidence import discover_wikipedia, fetch_revision
from app.benchmark.markets import MarketSelectionRequired, import_market

CUTOFF = datetime(2025, 1, 10, tzinfo=timezone.utc)
EVENT = "https://polymarket.com/event/lab-deal"
MARKET = {
    "id": "1", "conditionId": "condition-1", "slug": "lab-deal-close",
    "question": "Will the laboratory acquisition close?",
    "description": "Resolves Yes if the laboratory acquisition closes by February 1.",
    "outcomes": '["Yes", "No"]', "endDate": "2099-02-01T12:00:00Z",
    "createdAt": "2025-01-01T12:00:00Z", "updatedAt": "2025-01-03T12:00:00Z",
    "resolutionSource": "Official closing notice", "outcomePrices": '["0.9", "0.1"]',
    "resolvedOutcome": "Yes", "clobTokenIds": '["10", "20"]',
}
SEED = "https://en.wikipedia.org/wiki/Laboratory_deal"


def page(title, page_id, content, date="2025-01-08T12:00:00Z"):
    return {"query": {"pages": [{
        "pageid": page_id, "ns": 0, "title": title,
        "revisions": [{"timestamp": date, "revid": page_id + 100,
                       "slots": {"main": {"content": content}}}],
    }]}}


def test_import_uses_only_cutoff_revisions_and_strips_market_leakage():
    visited = []

    def handle(request):
        visited.append(request)
        if request.url.host == "gamma-api.polymarket.com":
            assert request.url.path == "/events/slug/lab-deal"
            return httpx.Response(200, json={"markets": [MARKET]})
        assert request.url.params.get("prop") == "revisions"
        assert request.url.params.get("rvstart") == CUTOFF.isoformat()
        assert request.url.params.get("rvdir") == "older"
        title = request.url.params.get("titles")
        if title == "Laboratory deal":
            return httpx.Response(200, json=page(title, 1, """
{{Latest data|[[Laboratory leak]]}}
The [[Laboratory acquisition]] was announced.
Polymarket assigned a 75% probability to closing. The proposed purchase price was 20 million dollars.
[[Completely unrelated entity]]
== References ==
[[Laboratory secret]]
"""))
        assert title == "Laboratory acquisition"
        return httpx.Response(200, json=page(title, 2, "The laboratory board approved the agreement."))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await import_market(EVENT, as_of=CUTOFF, seed_wiki_urls=[SEED], client=client)

    result = asyncio.run(run())
    assert result["group_id"] == "lab-deal"
    assert result["condition_id"] == "condition-1"
    assert result["outcomes"] == ["Yes", "No"]
    assert result["market_rules_at"] <= result["as_of"]
    assert not {"outcomePrices", "resolvedOutcome", "clobTokenIds"} & result.keys()
    assert [r["revision_id"] for r in result["evidence"]] == [101, 102]
    assert [r["is_seed"] for r in result["evidence"]] == [True, False]
    assert result["evidence"][1]["discovered_from"].endswith("oldid=101")
    assert "Latest data" not in result["evidence"][0]["text"]
    assert "75%" not in result["evidence"][0]["text"]
    assert result["evidence"][0]["redacted_market_odds"] == 1
    assert "20 million dollars" in result["evidence"][0]["text"]
    assert len(visited) == 3


def test_historical_detail_article_has_room_and_is_fetched_at_cutoff():
    requests = []

    def handle(request):
        query = request.url.params
        title = query["titles"]
        requests.append(title)
        assert query["rvstart"] == CUTOFF.isoformat()
        if title == "Laboratory deal":
            return httpx.Response(200, json=page(title, 1, """
The laboratory acquisition was announced.
== Contemporaneous analysis ==
{{main|Laboratory acquisition history}}
{{Latest data|[[Laboratory future results]]}}
"""))
        assert title == "Laboratory acquisition history"
        return httpx.Response(200, json=page(title, 2, "The laboratory board approved the agreement."))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await discover_wikipedia(
                client, question="Will the laboratory acquisition close?", rules="Yes if the acquisition closes.",
                as_of=CUTOFF, seed_urls=[SEED], max_pages=2, max_link_attempts=2,
            )

    evidence, _ = asyncio.run(run())
    assert requests == ["Laboratory deal", "Laboratory acquisition history"]
    assert evidence[1]["discovered_from"].endswith("oldid=101")
    assert "Latest data" not in evidence[0]["text"]


@pytest.mark.parametrize(("question", "main_title", "timeline_title", "weak_titles", "main_query"), [
    ("Will the U.S. invade Iran before 2027?", "2026 Iran war", "Timeline of the 2026 Iran war",
     ["Iran hostage crisis", "Timeline of Iranian history", "F-15 rescue in Iran"], "2026 iran war"),
    ("Will Labour win the 2026 UK general election?", "2026 United Kingdom general election",
     "Timeline of the 2026 United Kingdom general election",
     ["2025 United Kingdom general election", "Opinion polling for the 2026 United Kingdom general election",
      "2026 United States general election", "Labour Party leadership"], "2026 united kingdom general election"),
])
def test_automatic_search_uses_several_queries_and_screens_titles_before_fetching(
        question, main_title, timeline_title, weak_titles, main_query):
    cutoff = datetime(2026, 11, 9, tzinfo=timezone.utc)
    fetched = []

    def handle(request):
        params = request.url.params
        if params.get("list") == "search":
            query = params["srsearch"]
            titles = [*weak_titles, main_title] if query == main_query else [*weak_titles, timeline_title]
            return httpx.Response(200, json={"query": {"search": [
                {"title": title, "snippet": "Untrusted search snippet."} for title in titles]}})
        title = params["titles"]
        assert params["rvstart"] == cutoff.isoformat()
        fetched.append(title)
        index = [main_title, timeline_title].index(title)
        return httpx.Response(200, json=page(title, index + 1, f"Article excerpt {index}.",
                                              date="2026-11-08T00:00:00Z"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await discover_wikipedia(client, question=question, rules="", as_of=cutoff,
                                            seed_urls=[], max_pages=2, title_discovery=True)

    evidence, manifest = asyncio.run(run())
    searches = [row for row in manifest if row["stage"] == "search"]
    assert 3 <= len(searches) <= 4
    assert main_query in {row["query"] for row in searches}
    assert all("2027" not in row["query"] and "chain of command" not in row["query"] for row in searches)
    assert set(fetched) == {main_title, timeline_title} and len(fetched) == 2
    assert {row["title"] for row in evidence} == set(fetched)
    assert all(row["text"].startswith("Article excerpt") for row in evidence)
    rejected = {row["title"] for row in manifest if row["stage"] == "title_review" and not row["title_score"]}
    assert rejected == set(weak_titles)
    assert "Untrusted search snippet" not in str(manifest)
    from app.knowledge.forecast_evidence import frozen_sources

    sources = frozen_sources({"question": question, "rules": "", "as_of": cutoff.isoformat(),
                              "evidence": evidence}, "expanded", require_relevant=True)
    assert {row["title"] for row in sources} == {main_title, timeline_title}


def test_frozen_chronology_keeps_parent_years_and_more_than_twelve_relevant_sections():
    from app.knowledge.evidence import validate_event
    from app.knowledge.extraction import _sections
    from app.knowledge.forecast_evidence import frozen_sources

    entries = [(day, f"The United States and Boreal held ceasefire talks about crossing {day}.")
               for day in range(1, 19)]
    text = "== 2026 ==\n=== July ===\n" + "\n".join(
        f"==== {day} July ====\n{quote}\n" for day, quote in entries)
    case = {
        "question": "Will the U.S. invade Boreal before 2027?",
        "rules": "Yes if the United States commences a military offensive to establish control over Boreal.",
        "as_of": "2026-07-20T00:00:00Z",
        "evidence": [{"url": SEED, "title": "Boreal conflict chronology", "text": text,
                      "revision_id": 123, "revision_at": "2026-07-19T00:00:00Z"}],
    }
    sections = _sections(frozen_sources(case, "expanded"))
    assert len(sections) == 18
    section = next(row for row in sections if row["path"] == "source_01/2026/July/18 July")
    recovered, _ = validate_event({"timestamp": "2025-07-18", "evidence_text": entries[-1][1]}, section)
    assert recovered["timestamp"] == "2026-07-18"


def test_archive_bypasses_updated_market_and_rejects_wrong_version():
    calls = []

    def handle(request):
        calls.append(request.url.host)
        assert request.url.host == "en.wikipedia.org"
        return httpx.Response(200, json=page("Laboratory deal", 1, "The laboratory announced a planned closing."))

    async def run(captured_at):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await import_market(
                EVENT, as_of=CUTOFF, seed_wiki_urls=[SEED], client=client,
                archived_market={"captured_at": captured_at, "event_slug": "lab-deal", "market": MARKET},
            )

    assert asyncio.run(run("2025-01-03T12:00:00Z"))["market_id"] == "1"
    with pytest.raises(ValueError, match="archive was captured after the forecast cutoff"):
        asyncio.run(run("2025-01-11T12:00:00Z"))
    assert calls == ["en.wikipedia.org"]


def test_case_identity_binds_rules_cutoff_and_selected_revisions():
    async def run(*, revised_rules=MARKET["description"], text="Initial negotiation.", cutoff=CUTOFF):
        def handle(request):
            if request.url.host == "gamma-api.polymarket.com":
                return httpx.Response(200, json={"markets": [MARKET | {"description": revised_rules}]})
            return httpx.Response(200, json=page("Laboratory deal", 1, text))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return (await import_market(EVENT, as_of=cutoff, seed_wiki_urls=[SEED], client=client))["case_id"]

    original = asyncio.run(run())
    assert asyncio.run(run()) == original
    assert asyncio.run(run(revised_rules="Resolve Yes only if all steps are complete.")) != original
    assert asyncio.run(run(text="A different evidence snapshot.")) != original
    assert asyncio.run(run(cutoff=CUTOFF.replace(day=11))) != original


def test_post_cutoff_market_update_rejected_before_wiki_fetch():
    def handle(request):
        assert request.url.host == "gamma-api.polymarket.com"
        return httpx.Response(200, json={"markets": [MARKET | {"updatedAt": "2025-01-11T12:00:00Z"}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await import_market(EVENT, as_of=CUTOFF, seed_wiki_urls=[SEED], client=client)

    with pytest.raises(ValueError, match="market version does not apply at this historical cutoff"):
        asyncio.run(run())


def test_grouped_event_returns_explicit_choices_and_can_select_by_query():
    other = MARKET | {"id": "2", "slug": "lab-deal-delay"}

    def handle(request):
        if request.url.host == "gamma-api.polymarket.com":
            return httpx.Response(200, json={"markets": [MARKET, other]})
        return httpx.Response(200, json=page("Laboratory deal", 1, "The laboratory announced a planned closing."))

    async def run(url):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await import_market(url, as_of=CUTOFF, seed_wiki_urls=[SEED], client=client)

    with pytest.raises(MarketSelectionRequired) as exc:
        asyncio.run(run(EVENT))
    assert len(exc.value.choices) == 2
    assert asyncio.run(run(exc.value.choices[1]["url"]))["market_id"] == "2"


def test_live_import_without_seed_uses_title_search_then_frozen_revision():
    order = []

    def handle(request):
        if request.url.host == "gamma-api.polymarket.com":
            order.append("market")
            return httpx.Response(200, json={"markets": [MARKET]})
        if request.url.params.get("list") == "search":
            order.append("search")
            return httpx.Response(200, json={"query": {"search": [
                {"title": "Laboratory deal", "snippet": "Future snippet is not input"},
            ]}})
        order.append("revision")
        assert request.url.params.get("rvstart")
        return httpx.Response(200, json=page("Laboratory deal", 1, "The laboratory announced a planned closing."))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await import_market(EVENT, as_of=None, seed_wiki_urls=[], client=client)

    result = asyncio.run(run())
    assert order == ["market", "search", "search", "search", "search", "revision"]
    assert "Future snippet" not in str(result)
    assert result["evidence"][0]["discovered_from"] == "wiki_search"


def test_wiki_rejects_late_revisions_and_non_wikimedia_seeds():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=page(
            "Laboratory deal", 1, "This must not enter context.", "2025-01-11T12:00:00Z",
        ))

    async def run(url):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await fetch_revision(client, url, CUTOFF)

    with pytest.raises(ValueError, match="revision after the cutoff"):
        asyncio.run(run(SEED))
    with pytest.raises(ValueError, match="Wikipedia seed URL"):
        asyncio.run(run("https://en.wikipedia.org@internal.example/wiki/Laboratory_deal"))
    assert len(calls) == 1
