"""Conservative alias equivalence, distinct from hierarchy and representation.

No country, organization or scenario names are hard-coded here. Unproven
identity links retain their own names; affiliation alone never joins entities.
"""
import re
import unicodedata

from app.domain.identity import identity_key

VERSION = 'actor-identity-2'
_STOP = {'of', 'the', 'and', 'for', 'in', 'on', 'de', 'la', 'und', 'et'}


def _clean(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text or '')).split())


def _base(name):
    return re.sub(r'^(?:the\s+)', '', _clean(name), flags=re.I)


def _initials(name):
    words = re.findall(r'[A-Za-z]+', name)
    return ''.join(word[0] for word in words if word.lower() not in _STOP).upper()


def _abbreviation(name):
    return bool(re.fullmatch(r'[A-Z][A-Z0-9.\-]{1,11}', name))


def _person_form(name):
    return re.sub(r'^(?:(?:president|prime minister|minister|senator|chair|captain|dr|prof|mr|ms|mrs)\.?\s+)+',
                  '', _base(name), flags=re.I).casefold()


def _person_variant(first, second, figures):
    from app.domain.agency import resolve_entity_type
    people = {_base(row.get('name') or row.get('canonical_name') or row.get('figure'))
              for row in figures if resolve_entity_type(row)[0] == 'person'}
    if _base(first) not in people or _base(second) not in people:
        return False
    a, b = _person_form(first), _person_form(second)
    if a == b:
        return True
    for short, full in ((a,b),(b,a)):
        if ' ' not in short and full.endswith(' ' + short):
            matches = {_person_form(name) for name in people if _person_form(name).endswith(' ' + short)}
            if len(matches) == 1:
                return True
    return False


def equivalent_names(first, second, names=(), evidence=()):
    a, b = _base(first), _base(second)
    if not a or not b:
        return False
    if identity_key(a) == identity_key(b):
        return True
    # A parenthetical acronym is safe; a year, regional label or arbitrary
    # qualifying phrase is not automatically the same identity as the base.
    for full, other in ((a,b),(b,a)):
        bracket = re.fullmatch(r'(.+?)\s*\(([^()]+)\)', full)
        if bracket and _abbreviation(bracket[2]) and _initials(bracket[1]) == identity_key(bracket[2]).upper():
            if identity_key(other) in {identity_key(bracket[1]), identity_key(bracket[2])}:
                return True
    for short, full in ((a,b),(b,a)):
        if _abbreviation(short) and _initials(full) == identity_key(short).upper() and ' ' in full:
            expansions = {identity_key(_base(n)) for n in names
                          if ' ' in _base(n) and _initials(_base(n)) == identity_key(short).upper()}
            if len(expansions) <= 1:
                return True
    # Identity definitions, not merely two names in one action or paragraph.
    marker = r'(?:also\s+known\s+as|abbreviated\s+as|also\s+called|referred\s+to\s+as|即|又称|简称|全称为)'
    for text in evidence:
        text = _clean(text)
        for left, right in ((a,b),(b,a)):
            pattern = rf'(?<!\w){re.escape(left)}\s*[,，(（]?\s*{marker}\s*["“\']?(?:the\s+)?{re.escape(right)}(?!\w)'
            if re.search(pattern, text, re.I):
                return True
    return False


def guard_alias_mapping(mapping, figures=(), actions=(), timeline=()):
    names = {str(row.get('name') or row.get('canonical_name') or row.get('figure') or '').strip()
             for row in figures if isinstance(row, dict)}
    names.update(str(row.get(key) or '').strip() for row in actions for key in ('source','target'))
    names.discard('')
    evidence = [str(row.get('evidence_text') or row.get('description') or '') for row in timeline]
    accepted, rejected = {}, []
    for alias, canonical in mapping.items():
        if alias == canonical:
            continue
        if alias not in names or canonical not in names:
            reason = 'identity_name_not_in_inputs'
        elif not equivalent_names(alias, canonical, names, evidence) and not _person_variant(alias, canonical, figures):
            reason = 'identity_not_established_by_form_or_source'
        else:
            accepted[alias] = canonical
            continue
        rejected.append({'alias': alias, 'canonical': canonical, 'reason': reason})
    return accepted, {'version': VERSION, 'accepted': len(accepted), 'rejected': rejected}


def discovered_person_aliases(figures):
    """Propose only title variants and unique surnames already in the inputs."""
    from app.domain.agency import resolve_entity_type
    names = sorted({str(row.get('name') or row.get('canonical_name') or row.get('figure') or '').strip()
                    for row in figures if isinstance(row, dict) and resolve_entity_type(row)[0] == 'person'})
    forms = {}
    for name in names:
        forms.setdefault(_person_form(name), []).append(name)
    fulls = {form for form in forms if len(form.split()) > 1}
    mapping = {}
    for name in names:
        form = _person_form(name)
        matches = [full for full in fulls if full.endswith(' ' + form)] if ' ' not in form else [form]
        if len(matches) == 1:
            canonical = min(forms[matches[0]], key=lambda label: (len(label), label))
            if name != canonical:
                mapping[name] = canonical
    return mapping


def delegated_seat_supported(pair, records):
    """A hierarchy label is insufficient evidence to remove a decision seat."""
    contained, container = str(pair.get('contained') or ''), str(pair.get('container') or '')
    if equivalent_names(contained, container):
        return True
    quote = str(pair.get('delegation_quote') or '').strip()
    if not pair.get('decision_class') or len(quote) < 20:
        return False
    if not all(_base(name).casefold() in quote.casefold() for name in (contained,container)):
        return False
    evidence = [str(row.get('evidence_text') or row.get('description') or '')
                for record in records for row in record.get('history', [])]
    if not any(quote in text for text in evidence):
        return False
    return bool(re.search(r'\b(?:only|solely|without (?:independent )?discretion)\b|仅负责|仅执行|无独立裁量', quote, re.I)
                and re.search(r'\b(?:execut\w*|transmit\w*|implement\w*|delegat\w*)\b|执行|传达|受委托', quote, re.I))
