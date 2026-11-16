"""Actor-specific original excerpts from the case's stored, cutoff-safe sources."""
from __future__ import annotations

from calendar import monthrange
from datetime import datetime, timezone
import hashlib
import re

from app.knowledge.retrieval import _named_in, terms
from app.knowledge.scraper import split_sections
from app.knowledge.sentences import sentence_records
from app.storage.json_store import EventRepository


STRUCTURAL = re.compile(
    r'\b(?:authorit\w*|mandate|governance|responsibilit\w*|approv\w*|appoint\w*|controls|operates|capacity|owns|budget|subsidiar\w*)\b'
    r'|职责|权限|批准|授权|任命|董事会|主管|拥有|产能|管辖|负责|隶属|组织结构', re.I,
)
ACTION = re.compile(
    r'\b(?:propos\w*|negotiat\w*|nominat\w*|endors\w*|campaign\w*|organis\w*|organiz\w*|launch\w*|'
    r'announc\w*|submit\w*|vot\w*|allocat\w*|block\w*|sign\w*|implement\w*|deliver\w*)\b'
    r'|提议|谈判|提名|支持|竞选|组织|提交|表决|分配|阻止|签署|执行|交付', re.I,
)
_CONTINUATION = re.compile(r'^(?:it|they|he|she|the (?:party|company|agency|committee|government))\b|^(?:它|其|该机构|该党|该公司)', re.I)
_SNIPPET_LIMIT = 1400
VERSION = 'profile-evidence-2'


def _time(value, *, end_of_period=False):
    if not value:
        return None
    value = str(value)
    try:
        if re.fullmatch(r'\d{4}-\d{2}', value):
            year, month = map(int, value.split('-'))
            day = monthrange(year, month)[1] if end_of_period else 1
            value = f'{value}-{day:02d}'
        if end_of_period and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            value += 'T23:59:59.999999+00:00'
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _available(fragment, source, boundary):
    if boundary is None:
        return True
    # A dated historical revision can be fetched later. A mutable page without
    # a revision is only safe as of its recorded capture.
    stamp = (fragment.get('revision_at') or fragment.get('retrieved_at')
             or fragment.get('published_at') or source.get('revision_at')
             or source.get('retrieved_at') or source.get('published_at'))
    available = _time(stamp, end_of_period=True)
    return available is not None and available <= boundary


def _actor_names(profile):
    raw = [profile.get('canonical_name') or profile.get('name') or profile.get('figure') or '',
           *(profile.get('aliases') or [])]
    names = [name.strip().casefold() for name in raw if isinstance(name, str) and len(name.strip()) > 1]
    names += [re.sub(r'\s+\([^()]+\)\s*$', '', name) for name in names]
    return list(dict.fromkeys(name for name in names if len(name) > 1))


def prose_blocks(text):
    """Literal prose spans, with independent list items and no table syntax."""
    for paragraph in re.finditer(r'\S.*?(?=\n[ \t]*\n|\Z)', text, flags=re.S):
        body = paragraph.group()
        if body.lstrip().startswith(('{|', '|', '!')):
            continue
        markers = list(re.finditer(r'(?m)^[ \t]*[*#]+[ \t]*', body))
        starts = [(0, markers[0].start())] if markers and markers[0].start() else []
        starts += [(mark.end(), markers[i+1].start() if i+1 < len(markers) else len(body))
                   for i, mark in enumerate(markers)]
        for start, end in starts or [(0, len(body))]:
            while start < end and body[start].isspace():
                start += 1
            while end > start and body[end-1].isspace():
                end -= 1
            if start < end and not re.fullmatch(r'={2,6}[^\n]*={2,6}', body[start:end]):
                yield paragraph.start() + start, body[start:end]


def _snippets(text):
    # Continuations stay inside one prose block, never across unrelated bullets.
    for offset, body in prose_blocks(text):
        sentences = sentence_records(body)
        for index, sentence in enumerate(sentences):
            start, end = sentence['start'], sentence['end']
            if 8 <= len(sentence['quote']) <= _SNIPPET_LIMIT:
                yield offset + start, offset + end, sentence['quote'], False
            if index and _CONTINUATION.search(sentence['quote']):
                start = sentences[index - 1]['start']
                if end - start <= _SNIPPET_LIMIT:
                    yield offset + start, offset + end, body[start:end], True


def source_catalog(profile, *, cutoff='', event_title='', sources=None):
    case_id = profile.get('_case_id')
    if sources is None:
        sources = EventRepository().load_sources(case_id) if case_id else []
    names = _actor_names(profile)
    boundary = _time(cutoff, end_of_period=True)
    if not names or (cutoff and boundary is None):
        return []
    query = set(terms(event_title, limit=30))
    rows = []
    for index, source in enumerate(sources, 1):
        fragments = source.get('sections') if source.get('sdk_fallback') else None
        for fragment in fragments or [source]:
            if not _available(fragment, source, boundary):
                continue
            body = str(fragment.get('content') or '')
            if not body:
                continue
            snapshot = hashlib.sha256(body.encode()).hexdigest()
            url = str(fragment.get('url') or source.get('url') or '')
            title = str(fragment.get('title') or source.get('title') or '')
            actor_page = re.sub(r'\s+\([^()]+\)\s*$', '', title.casefold()) in names
            revision_id = str(fragment.get('revision_id') or source.get('revision_id') or '')
            revision_at = str(fragment.get('revision_at') or source.get('revision_at') or '')
            cursor = 0
            for section in split_sections(body, index, max_chars=None):
                section_body = section['content']
                offset = body.index(section_body, cursor)
                cursor = offset + len(section_body)
                for start, end, quote, continuation in _snippets(section_body):
                    direct = any(_named_in(name, quote) for name in names)
                    if continuation and not direct:
                        continue
                    structural = bool(STRUCTURAL.search(quote))
                    action = bool(ACTION.search(quote))
                    if not direct and not (actor_page and (structural or action)):
                        continue
                    key = hashlib.sha256(f'{url}\n{snapshot}\n{offset + start}:{offset + end}'.encode()).hexdigest()[:20]
                    rows.append({
                        'evidence_id': 'profile_' + key, 'text': quote, 'url': url, 'title': title,
                        'section': str(fragment.get('path') or section['path']), 'snapshot_sha256': snapshot,
                        'revision_id': revision_id, 'revision_at': revision_at,
                        'span_start': offset + start, 'span_end': offset + end,
                        '_score': (3 * structural + 3 * action + 2 * direct,
                                   len(query & set(terms(quote))), continuation, -index),
                    })
    rows.sort(key=lambda row: row['_score'], reverse=True)
    selected = []
    for row in rows:
        if any(row['text'] in hit['text'] or hit['text'] in row['text'] for hit in selected):
            continue
        row.pop('_score')
        selected.append(row)
        if len(selected) == 8:
            break
    return selected


def public_background(sources, *, cutoff=''):
    """Already selected benchmark packets are shared background, not authority.

    Ordinary unbounded article collections retain their existing retrieval path.
    Only a packet marked by the shared-source builder enters this projection.
    """
    from app.knowledge.public_evidence import VERSION
    boundary = _time(cutoff, end_of_period=True)
    if cutoff and boundary is None:
        return []
    return [dict(source) for source in sources or []
            if source.get('public_packet_version') == VERSION and _available(source, source, boundary)]


def background_catalog(sources, *, cutoff=''):
    """Make the already selected public prose citeable, including representative speech.

    Sharing background without numbered citations prevented a dossier from using
    relevant evidence outside its eight name-matched excerpts.
    """
    rows = []
    for source in public_background(sources, cutoff=cutoff):
        text = str(source.get('content') or '')
        for offset, body in prose_blocks(text):
            spans = [{'start': 0, 'end': len(body)}] if len(body) <= _SNIPPET_LIMIT else sentence_records(body)
            for span in spans:
                quote = body[span['start']:span['end']]
                if not 8 <= len(quote) <= _SNIPPET_LIMIT:
                    continue
                start = offset + span['start']
                key = hashlib.sha256(f"{source['evidence_id']}\n{start}:{start+len(quote)}".encode()).hexdigest()[:20]
                rows.append({key: str(source.get(key) or '') for key in
                             ('url', 'title', 'revision_id', 'revision_at', 'snapshot_sha256')} |
                            {'evidence_id': 'background_' + key, 'text': quote,
                             'span_start': start, 'span_end': start + len(quote)})
    return rows


def ground_opening_intentions(core):
    """Enforce a source's explicit deadline even if the model omitted its field."""
    from calendar import monthrange
    from datetime import date, timedelta
    from app.knowledge.evidence import BOUND_DATE, date_parts
    value = core.model_copy(deep=True)
    retained = []
    for item in value.initial_intentions:
        deadlines = []
        for ref in item.source_refs:
            years = set(re.findall(r'\b[12]\d{3}\b', ref.title + ' ' + ref.section))
            anchor = int(next(iter(years))) if len(years) == 1 else None
            own_dates = date_parts(' '.join([item.description, item.conditions, item.review_condition]), anchor)
            for match in BOUND_DATE.finditer(ref.quote):
                if not re.match(r'(?:before|until|through)\b', match[0], re.I):
                    continue
                if re.search(r'\bnot\s*$', ref.quote[:match.start()], re.I):
                    continue  # "not before" is an earliest start, not an expiry.
                dates = date_parts(match[0])
                if not dates and anchor:
                    dates = date_parts(match[0], anchor)
                    surrounding = BOUND_DATE.sub('', ref.quote)
                    observed = date_parts(surrounding, anchor)
                    # A yearless deadline before the announcement month may
                    # refer to the next year. Preserve that ambiguity.
                    if not observed or any(day[:7] < min(observed)[:7] for day in dates):
                        continue
                for day in dates & own_dates:
                    exact_day = len(day) == 10
                    if len(day) == 7:
                        year, month = map(int, day.split('-'))
                        day += f'-{monthrange(year, month)[1]:02}'
                    if len(day) == 10:
                        if exact_day and re.match(r'before\b', match[0], re.I):
                            day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
                        deadlines.append(day)
        if deadlines:
            item.valid_until = min([*deadlines, *([item.valid_until] if item.valid_until else [])])
        if value.as_of and item.valid_until and item.valid_until < value.as_of[:10]:
            value.review_notes.append('A source-dated commitment expired before the evidence cutoff and was excluded from current intentions.')
        else:
            retained.append(item)
    value.initial_intentions = retained
    value.review_notes = list(dict.fromkeys(value.review_notes))[:20]
    return value
