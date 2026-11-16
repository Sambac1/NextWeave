"""Shared thresholds, prompt rules, and guards for actor hierarchy and membership."""

import math
import re
from app.domain.agency import non_agent_kind


def _clean_text(value: object) -> str:
    """Collapse internal whitespace so actor names remain valid single-line lookup keys."""
    return " ".join(str(value or "").split())


def _lower_text(value: object) -> str:
    return _clean_text(value).lower()


def name_key(value: object) -> str:
    """Return a case-insensitive actor key with normalized internal whitespace."""
    return _lower_text(value)


MAX_PRINCIPALS = 8

MAX_MEMBERS_PER_PRINCIPAL = 8

HISTORICAL_WINDOW_DAYS = 730

HISTORICAL_WINDOW_SPAN_RATIO = 0.10

CANDIDATE_SLICE_PRINCIPAL = 40
CANDIDATE_SLICE_MEMBER = 60

MEMBER_ELIGIBLE_TYPES = {"person", "government", "organization"}

LEADER_EXEMPT_MARKERS = (
    "president", "premier", "prime minister", "chairman",
    "general secretary", "head of state", "head of government",
    "minister", "secretary of state", "chief negotiator",
    "vice premier", "chancellor", "king", "monarch", "emperor",
)

_LEADER_MARKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in LEADER_EXEMPT_MARKERS) + r")",
    re.IGNORECASE,
)

ENTITY_TYPE_RATIO_MULTIPLIER = {
    "person": 0.5,
    "group": 1.6,
    "company": 1.6,
}

COMPACTABLE_CATEGORIES = (
    "parent_state_subunit",
    "supply_chain_third_party_country",
    "supporting_third_party_country",
    "affected_third_party_country",
    "background_organization",
)
CATEGORY_REPRESENTATIVE_LIMITS = {
    "supply_chain_third_party_country": 1,
    "supporting_third_party_country": 1,
    "affected_third_party_country": 1,
    "background_organization": 1,
}


COUNTRY_NAMES = frozenset({
    "Afghanistan", "Albania", "Algeria", "Andorra",
    "Angola", "Apartheid South Africa", "Argentina", "Armenia",
    "Australia", "Austria", "Austria-Hungary", "Azerbaijan",
    "Bahamas", "Bahrain", "Bangladesh", "Barbados",
    "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bohemia", "Bolivia", "Bosnia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "British India",
    "British Raj", "Brunei", "Bulgaria", "Burkina Faso",
    "Burma", "Burundi", "Byzantine Empire", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Ceylon",
    "Chad", "Chile", "China", "Colombia",
    "Comoros", "Congo", "Costa Rica", "Croatia",
    "Cuba", "Cyprus", "Czech Republic", "Czechia",
    "Czechoslovakia", "Denmark", "Djibouti", "Dominica",
    "Dominican Republic", "DR Congo", "Dutch East Indies", "East Germany",
    "East Pakistan", "Ecuador", "Egypt", "El Salvador",
    "Eritrea", "Estonia", "Eswatini", "Ethiopia",
    "Fiji", "Finland", "France", "French Indochina",
    "Gabon", "Gambia", "Georgia", "German Democratic Republic",
    "Germany", "Ghana", "Gran Colombia", "Greece",
    "Grenada", "Guatemala", "Guinea", "Guyana",
    "Haiti", "Honduras", "Hungary", "Iceland",
    "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Ivory Coast",
    "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kingdom of Italy", "Kingdom of Yugoslavia", "Kiribati",
    "Kuwait", "Kyrgyzstan", "Laos", "Latvia",
    "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali",
    "Malta", "Mandatory Palestine", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco",
    "Mongolia", "Montenegro", "Morocco", "Mozambique",
    "Myanmar", "Namibia", "Nauru", "Nazi Germany",
    "Nepal", "Netherlands", "New Zealand", "Nicaragua",
    "Niger", "Nigeria", "North Korea", "North Macedonia",
    "North Vietnam", "Norway", "Oman", "Ottoman Empire",
    "Pakistan", "Palau", "Panama", "Papua New Guinea",
    "Paraguay", "Persian Empire", "Peru", "Philippines",
    "Poland", "Portugal", "Prussia", "Qatar",
    "Republic of China", "Rhodesia", "Roman Empire", "Romania",
    "Russia", "Russian Federation", "Rwanda", "Samoa",
    "San Marino", "Saudi", "Saudi Arabia", "Senegal",
    "Serbia", "Serbia and Montenegro", "Seychelles", "SFR Yugoslavia",
    "Siam", "Sierra", "Sierra Leone", "Singapore",
    "Slovakia", "Slovenia", "Socialist Federal Republic of Yugoslavia", "Somalia",
    "South Africa", "South Korea", "South Sudan", "South Vietnam",
    "Soviet Union", "Spain", "Sri", "Sri Lanka",
    "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syria", "Tajikistan", "Tanzania", "Thailand",
    "Third Reich", "Togo", "Tonga", "Trinidad",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Türkiye", "Turkmenistan",
    "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "United States of America", "Uruguay",
    "USSR", "Uzbekistan", "Vanuatu", "Vatican City",
    "Venezuela", "Vietnam", "Weimar Republic", "West Germany",
    "West Pakistan", "Yemen", "Yugoslavia", "Zaire",
    "Zambia", "Zimbabwe",
})

_COUNTRY_LOWER = frozenset(c.lower() for c in COUNTRY_NAMES)


def is_country_name(name: str) -> bool:
    """Recognize exact country names and parenthesized country aliases, not institutional qualifiers."""
    text = _clean_text(name).lower()
    if text in _COUNTRY_LOWER:
        return True
    m = re.match(r"^(.+?)\s*\(([^)]*)\)\s*$", text)
    if not m:
        return False
    inside = m.group(2).strip()
    return bool(inside) and inside in _COUNTRY_LOWER


_ORG_TERMS = (
    "court", "council", "commission", "committee", "tribunal", "association",
    "organization", "union", "alliance", "bloc", "front", "network",
    "secretariat", "bureau", "board", "institute", "foundation",
    "wto", "world trade organization", "coalition", "league", "movement",
    "party", "asean", "nato", "opec", "g7", "g20", "brics",
    "united nations", "european union",
    "assembly", "legislature", "congress", "diet", "cortes",
    "command", "headquarters", "general staff", "intelligence service",
)

_OFFICE_TERMS = (
    "president", "vice president", "prime minister", "chancellor",
    "secretary-general", "secretary", "minister", "senator", "representative",
    "speaker", "chair", "leader", "spokesman", "spokesperson", "commander",
    "chief", "director", "governor", "mayor", "head of",
)


def _office_in_name(name_l: str) -> str:
    """Recognize a leading office phrase without confusing a named office holder with an institution."""
    for term in _OFFICE_TERMS:
        if not name_l.startswith(term):
            continue
        rest = name_l[len(term):].strip()
        if not rest or rest.startswith(("of", "for", "at", ",")):
            return term
    return ""


_ARMED_TERMS = (
    "armed forces", "military", "army", "navy", "air force", "troops", "junta",
    "police", "parliament", "senate", "militia", "insurgents", "rebels",
    "national guard", "guard", "brigade", "regiment", "division", "battalion",
    "squadron", "task force", "peacekeeping",
)

_COMPANY_TERMS = (
    "inc.", "corp.", "corporation", "company", "airlines", "airways", "holdings",
    "huawei", "airbus", "nvidia", "alibaba", "tencent", "bytedance",
    "microsoft", "amazon", "apple", "tesla",
)

_GOV_TERMS = (
    "government", "administration", "congress", "ministry", "department",
    "agency", "office", "customs", "regulator", "regulators",
)

_ORG_WHOLE_WORDS = frozenset({
    "bank", "group", "fund", "board", "center", "centre",
    "service", "services", "corps", "force", "forces", "unit", "units",
    "authority", "office", "bureau", "secretariat",
})

_NAME_PARTICLES = frozenset({
    "de", "van", "von", "der", "den", "di", "da", "dos", "del", "la", "le",
    "bin", "al", "ibn", "binte", "mac", "mc", "st", "ste", "y", "e",
})

_NON_NAME_SINGLE_WORDS = frozenset({
    "government", "parliament", "senate", "congress", "cabinet", "council",
    "military", "army", "navy", "police", "troops", "forces", "junta",
    "authorities", "state", "public", "people", "citizens", "officials",
    "activists", "leaders", "ministers", "supporters", "members", "voters",
    "judges", "lawmakers", "politicians", "civilians", "demonstrators",
})

_ENTITY_WORD_SPLIT = re.compile(r"[^0-9A-Za-zÀ-ɏ\.]+")
_ENTITY_WORD_ONLY = re.compile(r"[^0-9a-z]+")


def _entity_words(name: str) -> list[str]:
    """Split an entity name while preserving periods in personal initials."""
    return [w for w in _ENTITY_WORD_SPLIT.split(name) if w.strip(".")]


def looks_like_person_name(name: str) -> bool:
    """Use name morphology as a fallback cue; capitalization alone does not establish a person."""
    words = _entity_words(name)
    if not words or len(words) > 5:
        return False
    if len(words) == 1:
        core = words[0].strip(".")
        if not core or not core[0].isupper():
            return False
        if len(core) >= 3 and core.isupper():
            return False
        low = core.lower()
        if low.endswith("s") and len(low) > 3:
            return False
        return low not in _NON_NAME_SINGLE_WORDS
    raw = _ENTITY_WORD_SPLIT.split(name)
    hyphen_continuation: set[int] = set()
    for index in range(1, len(raw)):
        if raw[index].strip(".") and name.find(raw[index - 1]) >= 0:
            between = name[name.find(raw[index - 1]) + len(raw[index - 1]):]
            if between.startswith("-"):
                hyphen_continuation.add(index)
    saw_name_word = False
    for index, w in enumerate(raw):
        core = w.strip(".")
        if not core:
            continue
        if core.lower() in _NAME_PARTICLES:
            continue
        if not core[0].isupper() and index not in hyphen_continuation:
            return False
        saw_name_word = True
    return saw_name_word


def infer_entity_type(
    name: str,
    raw_type: str = "",
    title: str = "",
) -> str:
    """Infer entity type from country, institution, office, and name cues before model hints. Affiliation is not a type discriminator."""
    raw = _lower_text(raw_type)
    name_l = _lower_text(name)
    title_l = _lower_text(title)
    non_agent = non_agent_kind({"entity_type": raw, "title": title})
    if non_agent:
        return non_agent

    if is_country_name(name):
        return "country"
    if _office_in_name(name_l):
        return "person"
    if any(term in name_l for term in _GOV_TERMS):
        return "government"
    if any(term in name_l for term in _ORG_TERMS):
        return "organization"
    if any(term in name_l for term in _ARMED_TERMS):
        return "organization"
    if any(term in name_l for term in _COMPANY_TERMS):
        return "company"
    if set(_ENTITY_WORD_ONLY.split(name_l)) & _ORG_WHOLE_WORDS:
        return "organization"
    if any(term in title_l for term in (
        "president", "vice president", "secretary", "minister", "senator",
        "representative", "chair", "leader", "leadership", "spokesman",
        "spokesperson", "commander", "chief", "director", "governor",
        "mayor", "prime minister", "chancellor", "head of",
    )):
        return "person"
    if any(term in title_l for term in (
        "campaign", "party", "coalition", "committee", "commission", "council",
        "bank", "institution", "institutions", "agency", "ministry",
        "department", "office", "court", "tribunal", "organization",
        "organisation", "movement", "bloc", "alliance", "union", "regime",
        "government", "administration", "military", "army", "police",
        "forces", "media outlet", "media outlets", "news", "outlet",
        "advocacy", "interest group", "watchdog", "regulator",
    )):
        return "organization"
    if looks_like_person_name(name):
        return "person"
    if raw in {"country", "government", "organization", "company", "group"}:
        return raw
    return "organization"


def is_leader(fig: dict) -> bool:
    """Apply leadership exemptions only to entity types eligible for the member pool."""
    if not is_member_eligible_type(fig):
        return False
    text = " ".join(
        _clean_text(fig.get(k))
        for k in ("title", "role_in_event", "role", "affiliation")
    )
    return bool(_LEADER_MARKER_RE.search(text))


def is_member_eligible_type(fig: dict) -> bool:
    """Exclude sovereign countries before checking the member-type whitelist."""
    name = _clean_text(fig.get("canonical_name") or fig.get("name") or "")
    if name and is_country_name(name):
        return False
    return _lower_text(fig.get("entity_type") or "person") in MEMBER_ELIGIBLE_TYPES


def belongs_elsewhere(fig: dict, principal_name: str, principal_keys: set) -> bool:
    """Reject a candidate whose declared parent is another principal."""
    parent = _lower_text(fig.get("parent_actor"))
    if not parent or parent == _lower_text(principal_name):
        return False
    return parent in principal_keys


def historical_window_days(span_days: float | None = None) -> int:
    """Scale the inactivity window to the timeline span while retaining its configured minimum."""
    if not span_days or span_days <= 0:
        return HISTORICAL_WINDOW_DAYS
    scaled = int(span_days * HISTORICAL_WINDOW_SPAN_RATIO)
    return max(HISTORICAL_WINDOW_DAYS, scaled)


def is_historical(fig: dict, cutoff_dt, span_days: float | None = None) -> bool:
    """Apply inactivity filtering to people only when a cutoff is available."""
    if cutoff_dt is None:
        return False
    if _lower_text(fig.get("entity_type") or "person") != "person":
        return False
    ts = _clean_text(fig.get("latest_action_ts"))
    if not ts:
        return False
    act_dt = parse_iso_date(ts[:10])
    if act_dt is None:
        return False
    return (cutoff_dt - act_dt).days > historical_window_days(span_days)


def timeline_span_days(timeline: list) -> float | None:
    """Measure the full timeline span, independently of the scored actor pool."""
    ts_all = sorted(
        _clean_text(e.get("timestamp"))[:10]
        for e in (timeline or [])
        if isinstance(e, dict) and _clean_text(e.get("timestamp"))
    )
    if len(ts_all) < 2:
        return None
    first, last = parse_iso_date(ts_all[0]), parse_iso_date(ts_all[-1])
    if first is None or last is None:
        return None
    return max(0.0, (last - first).days)


def evidence_threshold_for(
    figure: dict,
    base_ratio: float,
    total_entries: int,
    absolute_floor: int,
) -> int:
    """Scale evidence requirements by timeline share with an absolute minimum."""
    entity_type = str(figure.get("entity_type") or "").strip().lower()
    multiplier = ENTITY_TYPE_RATIO_MULTIPLIER.get(entity_type, 1.0)
    scaled = math.ceil(base_ratio * multiplier * max(int(total_entries or 0), 0))
    return max(absolute_floor, scaled)


def parse_iso_date(value: str):
    """Parse partial or complete ISO dates; return None for unsupported input."""
    from datetime import datetime

    text = _clean_text(value)
    parts = [p for p in text.split("-") if p.isdigit()]
    try:
        if len(parts) >= 3:
            return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2:
            return datetime(int(parts[0]), int(parts[1]), 1)
        if len(parts) == 1:
            return datetime(int(parts[0]), 1, 1)
    except ValueError:
        return None
    return None


def resolve_cutoff(cutoff: str, pool: list) -> object:
    """Use the explicit cutoff or the newest action in the global actor pool."""
    if cutoff:
        dt = parse_iso_date(cutoff)
        if dt is not None:
            return dt
    candidate_ts = [
        _clean_text(f.get("latest_action_ts"))
        for f in (pool or [])
        if isinstance(f, dict) and _clean_text(f.get("latest_action_ts"))
    ]
    candidate_ts.sort(reverse=True)
    if candidate_ts:
        return parse_iso_date(candidate_ts[0][:10])
    return None


def find_container(actor: str, principal_names: list, all_figures: list) -> str:
    """Find a declared principal container, falling back to an unambiguous country principal."""
    by_name = {
        _clean_text(f.get("canonical_name") or f.get("name") or ""): f
        for f in (all_figures or [])
        if isinstance(f, dict)
    }
    principal_set = {n.lower(): n for n in principal_names if n and n != actor}
    if not principal_set:
        return ""
    parent = _clean_text((by_name.get(actor) or {}).get("parent_actor"))
    if parent and parent.lower() in principal_set:
        return principal_set[parent.lower()]
    actor_type = _lower_text((by_name.get(actor) or {}).get("entity_type"))
    if actor_type in ("person", "government", "organization"):
        countries = [
            name for name in principal_names
            if name != actor
            and _lower_text((by_name.get(name) or {}).get("entity_type")) == "country"
        ]
        if len(countries) == 1:
            return countries[0]
    return ""


PRINCIPAL_AUTONOMY_RULES = """Choose the decision units required by this scenario's causal process, not a default state-versus-state or person-versus-person cast.
- Preserve a body whose own approval, veto, proposal, commitment, resource allocation or implementation choice can change the trajectory.
- Sharing a parent organization, country, party, ownership or policy preference does not establish identical authority or full delegation.
- Distinct institutions can be separate principals without an open conflict: ordinary legislative approval, executive assent, independent review, negotiation and operational execution may require different decisions.
- Distinguish a representative speaking for a body from a person exercising an independently relevant power. Prefer one seat for the same decision, but do not discard an evidenced personal or institutional approval to reduce cast size.
- Offices and institutions are valid decision subjects when authority is vested in them; a named officeholder is not automatically the sole actor.
- Rivals, counterparties and independently approving bodies keep their own decisions. An affected object or mere source mention does not establish agency.
- Historical participants remain historical evidence unless their authority and role still apply at the scenario cutoff. Do not infer present authority from old actions alone.
- Select only supported actors and preserve evidence gaps. If the needed decision center is missing, report a coverage gap rather than granting its power to a broad parent."""


SUBORDINATE_MEMBER_RULES = """A candidate is a member of the principal when it is an INTERNAL part of it:
- Its head of state, government officials, ministers, departments, agencies, military leadership, central bank, ruling party or its leaders (e.g. for the principal "China": Xi Jinping, the Commerce Ministry, the Foreign Ministry, the People's Bank...).
- For an organization: its member states, leadership organs, or prominent sub-bodies (e.g. for "NATO": its Secretary-General, or a member state only if that state genuinely acts through NATO's own organs — otherwise the member state stays separate).
- For a person principal: their inner circle, key ministers, or campaign organs.

A candidate is NOT a member when:
- Its ``country`` column differs from the principal's own ``country``. A different country means a different decision centre: a Korean president or a Korean party is not an organ of the United States, however prominent it is in the same events. Where either side reads ``?``, fall back to the affiliation text and treat another country's government / party / army / police as external anyway.
- It is an EXTERNAL actor interacting with the principal (another state, an adversary, an ally, a mediator) — it belongs to no one. This includes a RIVAL in an election / primary / leadership contest: a competing candidate runs their own campaign against the principal and is never an organ of it, even when both belong to the same party.
- It is a separate autonomous unit, even if related (a court ruling on the principal, a regulator overseeing it, a company it owns only partially).
- It is a POPULATION GROUP / the affected public — civilians, citizens, voters, protesters, "the people", "Ukrainians", "the public". A group of people is not an organ the principal acts THROUGH; it is the audience of or a bystander to the principal's actions. (Exception: an organized faction that genuinely acts on the principal's behalf, such as a ruling party's youth wing or a state-aligned militia, may qualify — a mere demographic never does.)
- It is a HISTORICAL / DEPARTED figure — its last recorded action (see "last_recorded_action" in the candidate list) is more than {days} days before the event's present, or its title/role describes a post it no longer holds (a former president, an acting president who left office years ago, a commander who was removed). A member must be someone who can plausibly act for the principal NOW, in this event.
- The material gives no sign it acts for the principal.

Constraints:
- Only list actors from the candidates above — never invent names that are not listed.
- Assign a candidate to at most one principal per call (you are only deciding for this one principal).
- Prefer candidates with real evidence of acting for / inside the principal; ignore vague background names.
- Do not select more than {max_members} members. If none fit, return an empty list."""


OVERLAP_EXAMPLES = """Check whether two proposed seats duplicate the SAME decision, not whether one organization contains the other.
- A board and its company, a legislature and an executive, a regulator and an operator, or a negotiator and an independently approving counterparty can have separate decision rights.
- Same-side or same-parent actors are not redundant merely because they cooperate.
- A representative whose only evidenced move is transmitting an already authorized instruction may be represented by the authorizing seat for that move.
- Omit an overlap when complete delegation is not evidenced. Keep genuine identity aliases separate from this decision-seat review."""


OVERLAP_EVENT_CONTEXT_RULE = """BUT before declaring an overlap, consider the EVENT CONTEXT: if the event itself is the contained actor splitting from its container (a ruling-party internal struggle, a coup, a parliament fighting its president, a military faction acting independently), then that actor has its OWN decision line in this event and must STAY a principal. Overlap is about decision space in THIS event, not ownership in general.

Rules:
- Only report pairs where one actor's decision space is FULLY contained in the other's for this event.
- Never report two actors that act on different levels (a state vs a third-party country), or external actors that merely influence each other.
- Do not invent actors — every name must come from the principal list above.
- When in doubt, do NOT report an overlap (keeping both is safer than wrongly deleting one)."""


AUDIT_PROBLEM_TYPES = """1. missing_demote — an actor that is a principal while its CONTAINER is also a principal, and in THIS event the actor does NOT act on its own line (the state and its head of state / government / agency / ruling organ are both principals, with no split, crisis, coup or revolt in this event). Report the contained one. Example: "United States" and "Donald Trump" are both principals while the event shows Trump acting as the US decision-maker — the actor whose decisions ARE the state's belongs to the state, not on the table beside it. IMPORTANT priority rule: whenever a PERSON principal and the state/government that person serves (its parent_actor / title / affiliation show it) are BOTH principals, DEFAULT to reporting missing_demote for that person UNLESS the event clearly shows the person acting against or splitting from the state — this pattern is the most common duplication and must be caught even when uncertain. This default does NOT apply when the event IS the split: a self-coup, a coup, a president dissolving or defying the legislature, an official defecting or revolting — there the person has its own decision line and must STAY a principal, however much its role text resembles the state's.

2. conflict_member — ONE principal's member list contains two actors that act AGAINST each other in this event (rivals, adversaries, an opposition figure assigned under the government it opposes). A principal's members must be organs it acts THROUGH — mutually opposed actors can never both be its members. Report the principal and the LESS fitting member (the lower-scored / less clearly internal one).

3. foreign_member — a member belongs to a DIFFERENT country than the principal, and did not act for this principal in this event. Report the principal and that member. This is the most common cross-country mistake: an event centred on one country draws that country's whole cast into the pool, and the per-principal assignment call attaches them to whichever principal is being staffed — so a Korean president or an opposition leader lands under the United States simply because the United States was the only country principal on the table. A person's prominence in the event is NOT a reason to make them someone else's organ: a head of state, an opposition leader, a party chief or an army commander of country A is never an internal part of country B's government, its departments, or its officials."""


AUDIT_EVIDENCE_RULES = """Rules:
- Base every judgment on THIS event's politics, not general ownership.
- Do not report an actor that genuinely has its own decision line in this event (a rebel faction, an opposition winning the election, a legislature in constitutional conflict with its executive).
- Only use names that appear above. Never invent actors.
- For missing_demote: when in doubt about an official+state pair, REPORT it (a wrongly-demoted official stays in the simulation as the state's member, so the cost of over-reporting is small; a missed duplication costs a whole extra agent line).
- For conflict_member and everything else: when in doubt, report NOTHING (keeping the hierarchy is safer than wrongly dismantling it)."""
