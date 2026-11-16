"""Cheap temporal/causal coverage before score-based passage packing."""
import re

from app.knowledge.retrieval import terms

VERSION = 'causal-source-coverage-1'
DECISION = re.compile(
    r'\b(?:mandate|authority|approv\w*|appoint\w*|ratif\w*|governance|responsibilit\w*|'
    r'negotiat\w*|talks|invit\w*|consent|proposal|commit\w*|conditions?|coalition|'
    r'signaled|signalled|veto\w*|decision|request\w*)\b|职责|权限|批准|谈判|邀请|同意|条件|回应|提案', re.I)
ACTION = re.compile(
    r'\b(?:said|stated|announc\w*|criticiz\w*|condemn\w*|warn\w*|propos\w*|'
    r'negotiat\w*|talks|invit\w*|approv\w*|vot\w*|sign\w*|submit\w*|request\w*|'
    r'reject\w*|accept\w*|signaled|signalled|commit\w*)\b|表示|宣布|签署|批准|表决|邀请|提出|谈判', re.I)
CAUSAL_RIGHT = re.compile(r'\b(?:conditions?|consent|invit\w*|negotiat\w*|talks|approval|authority|mandate|veto\w*|ratif\w*|coalition)\b|条件|同意|邀请|谈判|权限|批准|联盟', re.I)


def coverage_priority(ranked, metadata, question, cutoff):
    """Reserve recent actions and decision context on relevant source pages.

    Metadata values are (page, heading, literal passage, section_id). This
    function only orders candidates; the caller still enforces its byte/row
    limits, preserves complete quotations and shares the packet across arms.
    """
    query = set(terms(question))
    year = int(str(cutoff)[:4])
    scores = {hit['id']: hit.get('score', 0) for hit in ranked}
    by_page = {}
    for key, (page, heading, text, section) in metadata.items():
        if text.lstrip().startswith('{|'):
            continue
        overlap = len(query & set(terms(page['title'])))
        if overlap < 2:
            continue
        dates = [int(value) for value in re.findall(r'\b20\d{2}\b', text) if int(value) <= year]
        latest = max(dates, default=0)
        if latest < year - 2:
            continue
        by_page.setdefault(page['url'], []).append(
            (latest, scores.get(key, 0), len(query & set(terms(text))), key, text, section))
    recent, decisions = [], []
    for rows in by_page.values():
        rows.sort(reverse=True)
        for target, predicate in ((recent, ACTION), (decisions, DECISION)):
            choices = [row for row in rows if predicate.search(row[4])]
            if target is decisions:
                choices.sort(key=lambda row: (bool(CAUSAL_RIGHT.search(row[4])), *row[:3]), reverse=True)
            if choices:
                target.append({'id': choices[0][3], 'score': choices[0][1]})
    # Context from each relevant page precedes repetitive high-scoring rows.
    combined = [*ranked[:2], *recent, *decisions, *ranked]
    return list({hit['id']: hit for hit in combined}.values())
