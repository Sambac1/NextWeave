"""Evidence-based boundary between participating objects and decision subjects."""

import re

from app.domain.identity import identity_key

VERSION = "decision-subject-1"
_HUMAN_OR_ORG = re.compile(
    r"\b(?:captain|pilot|owner|operator|crew|officer|engineer|minister|director|company|corporation|authority|agency|association|union|committee|government)\b|船长|船东|运营商|管理局|公司|委员会",
    re.I,
)
_ASSET = re.compile(
    r"^(?:(?:a|an|the|large|giant|Italian|Egyptian|Dutch|Russian|Panamanian|Japanese|Taiwanese|grounded|stranded)\s+)*(?:(?:container|cargo|bulk|oil|commercial|cruise|naval)\s+)?(?:ship|vessel|tug(?:boat)?|tanker|freighter|aircraft|airplane|train|vehicle|pipeline|bridge|equipment|machine|software|product)\b|^(?:集装箱船|货轮|船只|拖船|飞机|设备|管道|桥梁|产品|软件系统)",
    re.I,
)
_PLACE = re.compile(
    r"^(?:(?:a|an|the|major|international)\s+)*(?:canal|port|waterway|airport|city|geographic location)\b|^(?:运河|港口|机场|地理位置)",
    re.I,
)


def non_agent_kind(record: dict) -> str | None:
    raw = str(record.get("entity_type") or record.get("agent_type") or "").lower()
    if raw in {"asset", "object", "vessel", "product", "artifact"}:
        return "asset"
    if raw in {"location", "place", "event"}:
        return raw
    for field in ("title", "description", "role", "role_in_event"):
        identity = str(record.get(field) or "").strip()
        # Being a captain/owner/operator of a vessel does not make the actor a vessel.
        for pattern, kind in ((_ASSET, "asset"), (_PLACE, "location")):
            match = pattern.search(identity)
            if not match:
                continue
            # 'ship owner' is an actor; 'ship owned by a company' remains a ship.
            rest = identity[match.end() :].lstrip()
            if _HUMAN_OR_ORG.match(rest):
                continue
            return kind
    return None


def is_decision_subject(record: dict) -> bool:
    return non_agent_kind(record) is None


def resolve_entity_type(record: dict) -> tuple[str, dict]:
    """Use explicit identity descriptors, never a list of scenario entity names."""
    raw = str(record.get('agent_type') or record.get('entity_type') or 'entity').lower().strip()
    canonical = {'individual': 'person', 'organisation': 'organization', 'institution': 'organization',
                 'nation': 'country', 'state': 'country', 'corporation': 'company'}.get(raw, raw)
    seat = record.get('decision_seat') or (record.get('decision_core') or {}).get('seat') or {}
    owner = record.get('canonical_name') or record.get('source_figure') or record.get('name') or record.get('figure')
    seat_type = seat.get('entity_type')
    if (seat.get('seat_id') and seat.get('source_refs') and seat_type in
            {'person', 'country', 'government', 'organization', 'company', 'group'}
            and owner and identity_key(owner) == identity_key(seat.get('owner_name'))):
        return seat_type, {'source_type': raw, 'resolved_type': seat_type, 'status': 'registered_seat', 'descriptor': ''}
    person_role = re.compile(r'^(?:a |an |the )?(?:person|individual|leader|head|chief|president|minister|commander|founder|director|officer|chair|politician|executive|船长|部长|总统|董事长)\b', re.I)
    collective = re.compile(r'^(?:(?:an?|the)\s+)?(?:[\w-]+\s+){0,4}(organization|organisation|company|corporation|committee|agency|government|movement|group|party|union)\b|^(?:武装组织|政治组织|政党|公司|企业|政府|委员会|机构|团体)', re.I)
    descriptor = ''
    if canonical == 'person':
        for field in ('title', 'description', 'role_in_event'):
            text = str(record.get(field) or '').strip()
            if not person_role.match(text) and collective.match(text):
                descriptor = text
                canonical = 'company' if re.search(r'\b(?:company|corporation)\b|公司|企业', text, re.I) else 'organization'
                break
    return canonical, {'source_type': raw, 'resolved_type': canonical,
                       'status': 'identity_descriptor_correction' if descriptor else 'declared_type',
                       'descriptor': descriptor}


def normalize_entity_record(record: dict) -> dict:
    from copy import deepcopy
    value = deepcopy(record)
    kind, resolution = resolve_entity_type(value)
    field = 'agent_type' if 'agent_type' in value else 'entity_type'
    value[field] = kind
    if resolution['status'] in {'identity_descriptor_correction', 'registered_seat'}:
        value['entity_type_resolution'] = resolution
        if 'public_state' in value:
            value['public_state'] = dict(value.get('public_state') or {}) | {'entity_type_resolution': resolution}
    return value
