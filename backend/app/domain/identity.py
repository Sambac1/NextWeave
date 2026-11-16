from __future__ import annotations

import re
import unicodedata
import uuid


def identity_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def contract_subject_key(value: object) -> str:
    return identity_key(re.sub(r"(?i)^the\s+", "", str(value or "").strip()))


def _person_tokens(value: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", value, re.UNICODE)


def person_middle_initial_variant(name: str, text: str) -> str | None:
    """Resolve a person's written middle initials against the actual source text."""
    tokens = _person_tokens(name)
    if len(tokens) < 3 or any(len(token) != 1 for token in tokens[1:-1]):
        return None
    pattern = r"(?<!\w)" + re.escape(tokens[0])
    pattern += "".join(r"(?:\s+" + re.escape(initial) + r"\.?)?" for initial in tokens[1:-1])
    pattern += r"\s+" + re.escape(tokens[-1]) + r"(?!\w)"
    match = re.search(pattern, text, re.I)
    return match[0] if match else None


def registered_name_forms(name: str) -> tuple[str, ...]:
    name = re.sub(r"(?i)^the\s+", "", str(name or "").strip())
    declared = re.search(r"\s*\(([A-Z][A-Z0-9.\-]{1,14})\)$", name)
    if declared:
        return name, name[:declared.start()], declared[1]
    return (name,) if name else ()


def matching_subjects(subject: str, candidates, labels, is_person):
    """Prefer exact names and registered aliases, then an unambiguous personal shorthand."""
    key = contract_subject_key(subject)
    if not key:
        return []
    rows = [(row, [form for name in labels(row) for form in registered_name_forms(name)]) for row in candidates]
    exact = [row for row, names in rows if any(contract_subject_key(name) == key for name in names)]
    if exact:
        return exact
    shorthand = []
    single = len(_person_tokens(subject)) == 1
    for row, names in rows:
        if not is_person(row):
            continue
        if single:
            found = any(len(tokens := _person_tokens(name)) >= 2 and identity_key(tokens[-1]) == key for name in names)
        else:
            found = any(contract_subject_key(person_middle_initial_variant(subject, name)) == contract_subject_key(name)
                        or contract_subject_key(person_middle_initial_variant(name, subject)) == key for name in names)
        if found:
            shorthand.append(row)
    return shorthand


def leading_actor_span(text: str, actors, owner: str) -> int | None:
    """Find the leading literal spelling of a registered actor, respecting ambiguity."""
    matches = []
    registered = {identity_key(label): actor.agent_id for actor in actors for label in (actor.name, *actor.aliases)}
    for actor in actors:
        labels = [actor.name, *actor.aliases]
        for member in getattr(actor, 'public_state', {}).get('subordinates', []):
            name = member.get('name') if isinstance(member, dict) else getattr(member, 'name', '')
            if name and identity_key(name) not in registered:
                labels.extend([name, f"{actor.name}'s {name}"])
        core = getattr(actor, 'decision_core', None)
        if core and actor.agent_type != 'person':
            support = [ref for field in ('mandate','means','goals','initial_judgments')
                       for item in getattr(core, field) for ref in item.source_refs]
            for label in (actor.name, *actor.aliases):
                pattern = re.compile(r'(?<!\w)' + re.escape(label) + r"['’]s\s+(?P<unit>(?:[A-Z][A-Za-z’'-]*)(?:\s+(?:[A-Z][A-Za-z’'-]*|of|the|and)){0,7})")
                for ref in support:
                    for match in pattern.finditer(ref.quote):
                        unit = match['unit']
                        if (unit.split()[-1].casefold() in {'council','committee','ministry','department','office','board','agency','government','army','corps','delegation'}
                                and identity_key(unit) not in registered and identity_key(match[0]) not in registered):
                            labels.append(match[0])
        for label in labels:
            forms = registered_name_forms(label)
            acronym = re.fullmatch(r'(?:[A-Z]\.?){2,6}', label)
            if acronym:
                letters = label.replace('.', '')
                forms = (*forms, letters, '.'.join(letters), '.'.join(letters) + '.')
            if getattr(actor, "agent_type", "") == "person":
                tokens = _person_tokens(label)
                if len(tokens) > 2 and all(len(token) == 1 for token in tokens[1:-1]):
                    forms = (*forms, f"{tokens[0]} {tokens[-1]}")
            for name in forms:
                # A registered institution may act through its delegation.
                # Keep its base-name length so a separately registered delegation wins.
                match = re.match(
                    r'^\s*(?:["“]\s*)?(?:the\s+)?(?P<name>' + re.escape(name)
                    + r")(?:['’]s\s+(?:delegation|leadership|government|administration|board|representatives))?(?=\s|[:,.;!?]|$)",
                    text, re.I,
                )
                if match:
                    matches.append((len(name), actor.agent_id, match.start("name")))
    if not matches:
        return None
    longest = max(length for length, _, _ in matches)
    strongest = [(actor_id, start) for length, actor_id, start in matches if length == longest]
    return strongest[0][1] if {actor_id for actor_id, _ in strongest} == {owner} else None


def stable_agent_id(event_id: str | None, event_title: str, name: str) -> str:
    namespace = event_id or event_title
    token = uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{identity_key(name)}").hex[:16]
    return f"agent_{token}"
