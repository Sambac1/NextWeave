"""Source-extractive briefs: the model selects passages, not ownership claims."""

import re
from hashlib import sha256
from typing import Annotated

from pydantic import BaseModel, Field
from app.knowledge.evidence import locate_quote
from app.knowledge.sentences import sentence_records


class BriefExcerpt(BaseModel):
    source_index: int = Field(ge=1)
    quote: str = Field(default="", max_length=1200)
    sentence_ids: list[str] = Field(default_factory=list, max_length=8)


class EvidenceBrief(BaseModel):
    excerpts: list[BriefExcerpt] = Field(default_factory=list, max_length=10)


def render_brief(brief: EvidenceBrief, articles: list[dict]) -> str:
    selected = []
    seen = set()
    for item in brief.excerpts:
        if item.source_index > len(articles):
            continue
        source = articles[item.source_index - 1]
        content = str(source.get("content") or source.get("summary") or "")
        sentences = sentence_records(content)
        span = locate_quote(content, item.quote) if item.quote else None
        for sentence in sentences:
            matches = sentence['id'] in item.sentence_ids or (
                span and sentence['start'] < span[1] and sentence['end'] > span[0]
            )
            actual = sentence['quote']
            if matches and 20 <= len(actual) <= 1200 and actual not in seen:
                seen.add(actual)
                selected.append((item.source_index, source.get("title") or "Source", actual))
    if not selected:
        # A malformed selection must not turn an invented rewrite into the brief.
        for index, source in enumerate(articles, 1):
            for row in sentence_records(str(source.get("content") or source.get("summary") or "")):
                sentence = row['quote']
                if 40 <= len(sentence) <= 1200:
                    selected.append((index, source.get("title") or "Source", sentence))
                    break
            if len(selected) >= 5:
                break
    if not selected:
        raise ValueError("No traceable source excerpts were found for the background brief.")
    lines = [
        "Background Brief · Traceable Source Excerpts",
        "",
        "The passages below preserve the wording of their sources. Independent corroboration is a separate step. Organization and responsibility require explicit evidence beyond ambiguous pronouns.",
    ]
    for index, title, quote in selected:
        lines.extend(["", f"[Source {index} · {title}]", quote])
    lines.extend(
        [
            "",
            "Subsidiary relationships, responsibility, causation, and forecasts require explicit support in the cited passages.",
        ]
    )
    return "\n".join(lines)


class BriefReference(BaseModel):
    source_index: int = Field(ge=1)
    sentence_ids: list[str] = Field(min_length=1, max_length=12)


class BackgroundParagraph(BaseModel):
    heading: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=20, max_length=1800)
    references: list[BriefReference] = Field(min_length=1, max_length=8)


class NarrativeBrief(BaseModel):
    paragraphs: list[BackgroundParagraph] = Field(min_length=1, max_length=5)
    evidence_gaps: list[Annotated[str, Field(min_length=8, max_length=500)]] = Field(default_factory=list, max_length=3)


def brief_catalog(articles: list[dict]) -> list[dict]:
    """Bounded head, tail and distributed coverage, with stable source offsets."""
    catalog = []
    allowance = min(12000, 24000 // max(1, len(articles)))
    heading_pattern = r'(?m)^[ \t]*={2,}[ \t]*(.*?)[ \t]*={2,}[ \t]*$'
    for index, article in enumerate(articles, 1):
        content = str(article.get('content') or article.get('summary') or '')
        rows, excluded = [], False
        for row in sentence_records(content):
            headings = re.findall(heading_pattern, row['quote'])
            if headings:
                excluded = bool(re.match(r'^(?:Names|References|Notes|Footnotes|Bibliography|See also|External links|名称|参考|注释|外部链接)', headings[-1], re.I))
            quote = re.split(heading_pattern, row['quote'])[-1].strip() if headings else row['quote']
            if not excluded and 20 <= len(quote) <= 1200:
                rows.append(row | {'quote': quote})
        chosen, size = {}, 0
        # Keep the introduction, plus material beyond it. Never truncate sentences.
        for candidates, quota in ((rows, allowance * .55), (list(reversed(rows)), allowance * .20)):
            used = 0
            for row in candidates:
                if row['id'] not in chosen and used + len(row['quote']) <= quota:
                    chosen[row['id']] = row; used += len(row['quote']); size += len(row['quote'])
        remaining = [r for r in rows if r['id'] not in chosen]
        # Spread the remaining allowance over the body, rather than just its start.
        stride = max(1, len(remaining) // 15)
        for row in remaining[::stride]:
            if size + len(row['quote']) <= allowance:
                chosen[row['id']] = row; size += len(row['quote'])
        catalog.append({'source_index':index, 'title':article.get('title') or 'Source',
                        'sentences':[{'id':r['id'], 'quote':r['quote']} for r in sorted(chosen.values(), key=lambda r:r['start'])]})
    return catalog


def narrative_document(brief: NarrativeBrief, articles: list[dict], catalog: list[dict], additional_context='') -> dict:
    """Validate citation location; this is not an automatic entailment proof."""
    allowed = {c['source_index']:{s['id']:s['quote'] for s in c['sentences']} for c in catalog}
    paragraphs, seen = [], set()
    for paragraph in brief.paragraphs:
        key = re.sub(r'\s+', '', paragraph.text).casefold()
        if key in seen:
            raise ValueError('The background brief contains duplicate paragraphs.')
        seen.add(key)
        references, used = [], set()
        for reference in paragraph.references:
            if reference.source_index not in allowed:
                raise ValueError('The background brief cites an unavailable source.')
            article = articles[reference.source_index - 1]
            for sentence_id in reference.sentence_ids:
                if sentence_id not in allowed[reference.source_index]:
                    raise ValueError('The background brief cites a missing or incomplete source sentence.')
                marker = (reference.source_index, sentence_id)
                if marker in used:
                    continue
                used.add(marker)
                references.append({'source_index':reference.source_index, 'sentence_id':sentence_id,
                    'title':article.get('title') or 'Source', 'url':article.get('revision_url') or article.get('url') or '',
                    'quote':allowed[reference.source_index][sentence_id],
                    'retrieved_at':article.get('retrieved_at'), 'revision_id':article.get('revision_id')})
        paragraphs.append({'heading':paragraph.heading, 'text':paragraph.text.strip(), 'references':references})
    text = '\n\n'.join(p['heading']+'\n'+p['text']+' ['+', '.join(str(i) for i in sorted({r['source_index'] for r in p['references']}))+']' for p in paragraphs)
    if additional_context:
        text += '\n\nUser-Provided Context (Modeling Assumptions)\n'+additional_context
    signature = sha256('\n'.join(str(a.get('content') or a.get('summary') or '') for a in articles).encode()).hexdigest()
    return {'version':'background-brief-2', 'text':text, 'paragraphs':paragraphs,
            'additional_context':additional_context or '', 'source_digest':signature}


FORECAST_BRIEF_VERSION = "forecast-background-1"


def forecast_brief_catalog(question: str, articles: list[dict], *, rules="", as_of="") -> list[dict]:
    """Favor evidence about the named subject and the actual decision."""
    from app.knowledge.forecast_evidence import _named_scope
    from app.knowledge.retrieval import terms
    from app.knowledge.contract_focus import TerritorialOffensiveFocus, positive_condition

    names = _named_scope(question)
    catalog = brief_catalog(articles)
    focus_rule = TerritorialOffensiveFocus(question, rules, int(as_of[:4])) if as_of else None
    if focus_rule and focus_rule.enabled:
        focused = []
        for source in catalog:
            ranked = [(focus_rule.passage_score(source["title"], "", row["quote"]), row)
                      for row in source["sentences"]
                      if not re.match(r"^\s*[*#]*\s*(?:He|She|They|It)\b", row["quote"], re.I)]
            ranked = sorted((row for row in ranked if row[0]), key=lambda row: row[0], reverse=True)
            if ranked:
                focused.append((ranked[0][0], source | {"sentences": [row for _, row in ranked[:5]]}))
        if focused:
            return [source for _, source in sorted(focused, key=lambda row: row[0], reverse=True)]
    focus = set(terms(question)) - {"will", "before", "announce", "announces", "change", "its"}
    condition_terms = set(terms(positive_condition(rules)))
    named_article = any(pattern.search(row["title"]) for pattern in names for row in catalog)
    result = []
    for row in catalog:
        main_source = any(pattern.search(row["title"]) for pattern in names)
        ranked = []
        for sentence in row["sentences"]:
            quote = sentence["quote"]
            name_match = any(pattern.search(quote) for pattern in names)
            if named_article and not (main_source or name_match):
                continue
            overlap = len(focus & set(terms(quote)))
            role = bool(re.search(r"\b(?:authorit\w*|responsib\w*|mandate|independen\w*|directive|approval|scheduled|deadline|decision)\b", quote, re.I))
            if not (overlap or name_match or (main_source and role)):
                continue
            governing_power = bool(re.search(r"\b(?:independen\w*|mandate|power to|authority to|responsible for|can issue|may issue)\b", quote, re.I))
            score = 3 * overlap + len(condition_terms & set(terms(quote))) + 2 * name_match + 2 * role + 7 * governing_power
            ranked.append((score, sentence))
        if ranked:
            ordered = sorted(ranked, key=lambda item: -item[0])
            result.append(row | {"sentences": [sentence for _, sentence in ordered[:16 if main_source else 5]]})
    return sorted(result, key=lambda row: not any(pattern.search(row["title"]) for pattern in names))


def forecast_brief_preview(spec: dict, articles: list[dict]) -> dict:
    """Supply a compact, cited preview until narrative synthesis finishes."""
    from app.knowledge.contract_focus import positive_condition

    catalog = forecast_brief_catalog(spec["question"], articles,
                                    rules=spec.get("rules") or f'Yes if {spec.get("yes_criterion", "")}',
                                    as_of=spec["as_of"])
    date = spec["as_of"][:10]
    intro = f'The analysis asks “{spec["question"]}” and uses evidence available through {date} UTC.'
    if condition := positive_condition(spec.get("rules", "")):
        intro += f" The contract resolves Yes if {condition}."
    paragraphs = [{"heading": "Forecast scope", "text": intro, "references": []}]
    if catalog:
        match = catalog[0]
        article = articles[match["source_index"] - 1]
        sentences = match["sentences"][:2]
        paragraphs[0] = {"heading": "Evidence preview",
                           "text": intro + " " + " ".join(row["quote"].lstrip(" *#") for row in sentences),
                           "references": [{"source_index": match["source_index"], "sentence_id": row["id"],
                                           "quote": row["quote"], "title": match["title"],
                                           "url": article.get("revision_url") or article.get("url", "")} for row in sentences]}
    return {"version": "forecast-background-preview-1", "as_of": spec["as_of"], "paragraphs": paragraphs,
            "text": "\n\n".join(paragraph["text"] for paragraph in paragraphs), "additional_context": ""}
