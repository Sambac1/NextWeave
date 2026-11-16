"""Select a bounded forecast evidence packet from frozen revisions."""
import hashlib
import re

from app.benchmark.dynamics import scheduled_vote
from app.knowledge.contract_focus import TerritorialOffensiveFocus, last_named_year, positive_condition
from app.knowledge.retrieval import EvidenceIndex, terms
from app.knowledge.scraper import split_sections
from app.knowledge.sentences import sentence_records
from app.knowledge.wikipedia import utc_timestamp


def _frozen_pages(case: dict, mode: str) -> list[dict]:
    pages = case["evidence"]
    if mode == "seed_only":
        pages = [page for page in pages if page.get("is_seed")][:1] or pages[:1]
    cutoff = utc_timestamp(case["as_of"], "as_of")
    for page in pages:
        checksum = hashlib.sha256(page["text"].encode()).hexdigest()
        if utc_timestamp(page["revision_at"], "revision_at") > cutoff or page.get("sha256", checksum) != checksum:
            raise ValueError("Evidence revision is outside the cutoff or its body hash changed")
    return pages


def _passages(text: str):
    # Keep the column headers with table rows, and make prose deep in a
    # section searchable without carrying the whole preceding section.
    for index, block in enumerate(re.split(r"(?ms)(^[ \t]*\{\|[^\n]*\n.*?^[ \t]*\|\})", text)):
        if index % 2:
            from app.knowledge.table_evidence import table_passages

            # Spans define which grouped header labels a value belongs to.
            # Preserve raw rows for citations and keep indivisible rows intact.
            yield from table_passages(block)
            continue
        for paragraph in re.split(r"\n\s*\n", block):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= 1000:
                yield paragraph
                continue
            sentences = sentence_records(paragraph)
            start = end = 0
            for sentence in sentences:
                if sentence["end"] - start > 1000 and end > start:
                    yield paragraph[start:end].strip()
                    start = sentence["start"]
                if sentence["end"] - start > 2300:
                    start = sentence["end"]
                end = sentence["end"]
            if end > start:
                yield paragraph[start:end].strip()


_DECISION_PASSAGE = re.compile(
    r"\b(?:authori[sz]\w*|mandate|appoint\w*|approval|governance|procedur\w*|"
    r"coalition|campaign|delegation|ratif\w*|eligibility|responsibilit\w*|rules)\b"
    r"|授权|批准|职责|程序|竞选|联盟|代表团|规则", re.IGNORECASE,
)


_NAMED_PHRASE = re.compile(r"\b[A-Z][\w’-]+(?:\s+(?:(?:of|the|and|de|van)\s+)?[A-Z][\w’-]+)+")
_QUESTION_WORDS = {"will", "can", "could", "is", "are", "does", "would", "when", "who", "which", "what"}


def named_subjects(question: str) -> list[str]:
    names = []
    for match in _NAMED_PHRASE.finditer(question):
        words = match.group().split()
        if words[0].casefold() in _QUESTION_WORDS:
            words = words[1:]
        if words and words[0].casefold() == "the":
            words = words[1:]
        if len(words) > 1:
            names.append(" ".join(words))
    return list(dict.fromkeys(names))


def _named_scope(question: str):
    return [re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, name.split())) + r"(?!\w)", re.I)
            for name in named_subjects(question)]


def frozen_sources(case: dict, mode: str, *, require_relevant: bool = False) -> list[dict]:
    """Rank passages from dated revisions and share the same packet across arms."""
    pages = _frozen_pages(case, mode)
    title_selected = {page["url"] for page in pages if page.get("title_match_score")}
    question = case["question"]
    focus_rule = TerritorialOffensiveFocus(question, case["rules"], int(str(case["as_of"])[:4]))
    chunks, metadata = [], {}
    priority = {}
    section_headings = {}
    for number, page in enumerate(pages, 1):
        for section in split_sections(page["text"], number, max_chars=None):
            heading = section["title"]
            if heading.casefold() in {"references", "external links", "further reading", "see also"}:
                continue
            section_headings[section["section_id"]] = section["headings"]
            section_year = last_named_year(section["content"]) if focus_rule.enabled else 0
            for index, content in enumerate(_passages(section["content"])):
                key = f"{section['section_id']}:{index}"
                metadata[key] = (page, heading, content, section["section_id"])
                if focus_rule.enabled:
                    priority[key] = focus_rule.passage_score(page["title"], heading, content, section_year=section_year)
                chunks.append({"id": key, "kind": "section",
                               "metadata": {"title": f"{page['title']} {heading}"}, "body": content})

    voting_date = scheduled_vote(case)
    index = EvidenceIndex(chunks)
    condition = positive_condition(case["rules"])
    rules = {hit["id"]: hit["score"] for hit in index.search(condition or case["rules"][:1500], limit=40)}
    question_terms = terms(question)
    query_words = set(question_terms)
    pairs = set(zip(question_terms, question_terms[1:]))
    focus = next(iter(question_terms), "")
    page_words = {page["url"]: terms(page["title"]) for page in pages}
    # Allow common inflections and demonyms in passage text (Iran/Iranian,
    # attend/attended), while leaving the shared lexical index unchanged.
    variants = {
        word: {word + ending for ending in ("s", "ed", "ing", "an", "ian", "ean", "i")}
        for word in query_words if word.isascii() and word.isalpha() and len(word) >= 4
    }

    def relevance(hit):
        page, heading, content, _ = metadata[hit["id"]]
        passage_terms = terms(content)
        passage_words = set(passage_terms)
        body_matches = query_words & passage_words
        body_matches.update(word for word, forms in variants.items() if forms & passage_words)
        title_terms = page_words[page["url"]]
        heading_terms = terms(heading)
        def overlap(tokens):
            return len(pairs.intersection(zip(tokens, tokens[1:])))
        direct = len(body_matches)
        focused = focus in body_matches and len(body_matches - {"next", "us"}) >= 2
        aligned_table = (content.lstrip().startswith("{|") and len(query_words.intersection(title_terms)) >= 3
                         and any(token.isdigit() and token in query_words for token in heading_terms))
        # The page title provides context for numeric rows. A subject named
        # in the passage itself carries more weight than a title-only match.
        return (hit["score"] + 0.3 * rules.get(hit["id"], 0) + 2 * direct + 2 * overlap(passage_terms)
                + 3 * overlap(title_terms) + overlap(heading_terms)
                + (9 if focused else 0) + (12 if aligned_table else 0) + bool(page.get("is_seed")))

    ranked = index.search(question, limit=len(chunks)) or index.search(case["rules"][:1500], limit=12)
    if focus_rule.enabled:
        supported = {key for key, score in priority.items() if score}
        if supported:
            ranked_by_id = {hit["id"]: hit for hit in ranked}
            ranked = [ranked_by_id.get(key, {"id": key, "score": 0}) for key in metadata if key in supported]
    if voting_date:
        present = {row["id"] for row in ranked}
        ranked += [{"id": key, "score": 3.0} for key, (_, heading, content, _) in metadata.items()
                   if key not in present and heading == voting_date[:4]
                   and content.startswith("{|") and re.search(r"(?mi)^![^\n]*\bFieldwork date\b", content)]
    if require_relevant:
        if not ranked and not title_selected:
            return []
        names = _named_scope(question)
        if names and not focus_rule.enabled:
            def in_scope(hit):
                page, heading, content, _ = metadata[hit["id"]]
                return any(name.search(f"{page['title']} {heading} {content}") for name in names)

            focused = [hit for hit in ranked if in_scope(hit)]
            if focused:
                # A one-hop link is a search lead. Its passages still need to
                # mention the subject or come from a page supplied by the user.
                ranked = [hit for hit in ranked if in_scope(hit)
                          or metadata[hit["id"]][0].get("discovered_from") == "user_seed"]
    selected = sorted(ranked, key=lambda hit: (priority.get(hit["id"], 0), relevance(hit)), reverse=True) if ranked else chunks[:1]
    # Reserve room for decision and institutional passages before repetitive
    # observations consume the packet. Every method receives this same packet.
    decision_passages = []
    decision_sections = set()
    for hit in selected:
        _, heading, content, section_id = metadata[hit["id"]]
        if section_id not in decision_sections and not content.startswith("{|") and _DECISION_PASSAGE.search(heading + " " + content):
            decision_sections.add(section_id)
            decision_passages.append(hit)
            if len(decision_passages) == 3:
                break
    from app.knowledge.coverage_selection import coverage_priority
    if not focus_rule.enabled:
        selected = coverage_priority(selected, metadata, case['question'], case['as_of'])
    elif supported:
        reserved = []
        for check in (focus_rule.authority_passage, focus_rule.diplomacy_passage):
            for hit in selected:
                page, heading, content, _ = metadata[hit["id"]]
                if check(page["title"], heading, content):
                    reserved.append(hit)
                    break
        selected = list({hit["id"]: hit for hit in [*selected[:1], *reserved, *selected]}.values())
    leading = 4 if require_relevant else 8
    representative = {}
    if require_relevant:
        for hit in selected:
            representative.setdefault(metadata[hit["id"]][0]["url"], hit)
    page_passages = list(representative.values())[:2]
    title_passages = {}
    if title_selected:
        for hit in [*selected, *chunks]:
            url = metadata[hit["id"]][0]["url"]
            if url in title_selected:
                title_passages.setdefault(url, hit)
    selected = list({hit["id"]: hit for hit in [
        *title_passages.values(), *selected[:leading], *page_passages, *decision_passages, *selected,
    ]}.values())
    sources: dict[str, dict] = {}
    section_counts = {}
    limit, remaining = (24, 26000) if focus_rule.enabled and mode == "expanded" else (12, 16000)
    used = 0
    for hit in selected:
        if used >= limit:
            break
        page, _, content, section_id = metadata[hit["id"]]
        headings = "\n".join(f"{'=' * level} {title} {'=' * level}" for level, title in section_headings[section_id])
        size = len(headings) + len(content) + 3
        if size > remaining or section_counts.get(section_id, 0) >= 3:
            continue
        source = sources.setdefault(page["url"], {
            "url": page["url"], "title": page["title"], "revision_id": page["revision_id"],
            "revision_at": page["revision_at"], "retrieved_at": page["revision_at"], "sections": {},
        })
        _, excerpts = source["sections"].setdefault(section_id, (headings, []))
        excerpts.append((int(hit["id"].rsplit(":", 1)[1]), content))
        remaining -= size
        section_counts[section_id] = section_counts.get(section_id, 0) + 1
        used += 1
    for source in sources.values():
        sections = source.pop("sections")
        source["content"] = "\n\n".join(
            headings + "\n" + "\n\n".join(text for _, text in sorted(excerpts))
            for headings, excerpts in sections.values()
        )
    return list(sources.values())


