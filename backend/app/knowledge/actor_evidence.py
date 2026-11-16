"""Retain extracted decision candidates with literal, current authority evidence.

This is an evidence-floor exception, not an authority grant or principal selector.
It neither discovers new names nor changes timeline occurrence counts.
"""
import re

from app.knowledge.profile_evidence import _actor_names, source_catalog

VERSION = 'source-authority-retention-1'
_OPERATIONS = (
    r'(?:approv\w*|appropriat\w*|authoriz\w*|direct\w*|veto\w*|sign\w*|'
    r'appoint\w*|allocat\w*|submit\w*|issu\w*|regulat\w*|enforc\w*|'
    r'negotiat\w*|implement\w*|suspend\w*|decid\w*|certif\w*|ratif\w*|'
    r'award\w*|reject\w*|fund\w*|operat\w*|order\w*|set\b|draw\w*)'
)


def _current_authority(name, quote):
    # The name must be the grammatical holder, not an affected party near an
    # unrelated power word ("Budget", "approval requested", "X said Y can").
    subject = rf'(?<!\w){re.escape(name)}(?!\w)(?:\s*\([A-Z0-9. -]{{2,14}}\))?'
    modal = (r'(?:can|may|shall|must|has\s+(?:the\s+)?(?:authority|power|right)\s+to|'
             r'is\s+(?:authorized|empowered|required)\s+to|is\s+responsible\s+for)')
    if re.search(rf'{subject}\s+(?:(?:currently|legally|normally|also)\s+)?{modal}\s+{_OPERATIONS}', quote, re.I):
        return True
    # "The charter vests [the Council] with the authority to approve ...".
    if re.search(rf'\bvests\s+[^!?;\n]{{0,120}}{subject}\s+with\s+(?:the\s+)?(?:authority|power|right)\s+to\s+{_OPERATIONS}', quote, re.I):
        return True
    return bool(re.search(rf'{subject}(?:依法)?(?:有权|负责|可以|可)(?:批准|授权|签署|任命|分配|提交|发布|监管|执行|决定|否决|认证)', quote))


def attach_authority_evidence(figures, *, sources, cutoff='', event_title=''):
    """Annotate known candidates with at most two verified source spans each."""
    for figure in figures:
        figure.pop('source_authority_evidence', None)
        names = _actor_names(figure)
        hits = [row for row in source_catalog(figure, sources=sources, cutoff=cutoff,
                                             event_title=event_title)
                if any(_current_authority(name, row['text']) for name in names)]
        if hits:
            figure['source_authority_evidence'] = hits[:2]
    return figures
