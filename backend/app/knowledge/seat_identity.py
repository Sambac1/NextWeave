"""Evidence-backed identity links, kept separate from decision authority."""
import re

from app.domain.identity import identity_key
from app.knowledge.actor_identity import _person_form
from app.knowledge.retrieval import _named_in


def person_alias_link(name, quotes, catalog, candidates=()):
    """Resolve a cited surname only when the full identity exists and is unique.

    The returned source proves identity only. The caller retains the separate
    decision passages, so a name mention cannot grant a decision right.
    """
    full = _person_form(name)
    words = full.split()
    if len(words) < 2:
        return None
    surname = words[-1]
    if not any(_named_in(surname, quote) for quote in quotes):
        return None
    proof = next(((key, row) for key, row in catalog.items() if _named_in(full, row['text'])), None)
    if proof is None:
        return None
    identities = {identity_key(full)}
    for candidate in candidates:
        if candidate.get('entity_type') != 'person':
            continue
        label = _person_form(candidate.get('canonical_name') or candidate.get('name') or '')
        if len(label.split()) > 1 and label.split()[-1] == surname and any(_named_in(label, row['text']) for row in catalog.values()):
            identities.add(identity_key(label))
    # Detect another written full name even when an earlier extractor missed it.
    pattern = re.compile(r'(?<!\w)(?:[A-Z][A-Za-z’\'-]+\s+){1,3}' + re.escape(surname) + r'(?!\w)', re.I)
    for row in catalog.values():
        for match in pattern.finditer(row['text']):
            tokens = match[0].split()
            capitals = []
            for token in reversed(tokens[:-1]):
                if not token[0].isupper():
                    break
                capitals.insert(0, token)
            label = _person_form(' '.join([*capitals, tokens[-1]]))
            if len(label.split()) > 1:
                identities.add(identity_key(full if _named_in(full, label) else label))
    if identities != {identity_key(full)}:
        return None
    key, row = proof
    # Preserve the literal full-name spelling, without requiring an adjacent title.
    match = re.search(re.escape(full), row['text'], re.I)
    return {'name':match[0], 'alias':next(quote[m.start():m.end()] for quote in quotes
        if (m := re.search(r'(?<!\w)' + re.escape(surname) + r'(?!\w)', quote, re.I))),
        'source_id':key}
