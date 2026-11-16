"""Public membership projection; inferred grouping is never command authority."""

import re

from app.domain.identity import stable_agent_id
from app.domain.behavior import behavior_from_record
from app.domain.agency import non_agent_kind, normalize_entity_record


def public_profiles(agents, case_id, title):
    agents = [normalize_entity_record(a) for a in agents]
    fields = (
        "agent_id",
        "name",
        "aliases",
        "agent_type",
        "role",
        "title",
        "affiliation",
        "description",
        "relevant_timeline_refs",
    )
    profiles = [{k: a.get(k) for k in fields if a.get(k) is not None} for a in agents]
    principals = {
        str(n).casefold(): a["agent_id"]
        for a in profiles if not non_agent_kind(a)
        for n in [a.get("name", ""), *(a.get("aliases") or [])]
    }
    known = {a["agent_id"] for a in profiles if not non_agent_kind(a)}
    for a in profiles:
        a.update(
            decision_role="principal",
            parent_actor_id=None,
            membership_status="not_applicable",
        )
    for profile, agent in zip(profiles, agents):
        if agent.get('entity_type_resolution'):
            profile.update(entity_type_resolution=agent['entity_type_resolution'],
                           entity_type_status=agent['entity_type_resolution']['status'],
                           source_agent_type=agent['entity_type_resolution']['source_type'])
        non_agent = non_agent_kind(agent)
        if non_agent:
            profile.update(agent_type=non_agent, decision_role="mentioned", profile_scope="evidence_brief", entity_type_status="non_decision_object", source_agent_type=agent.get("agent_type"))
            continue
        descriptor = str(agent.get("description") or agent.get("title") or "")
        if (agent.get("entity_type_resolution") or {}).get("status") != "registered_seat" and profile.get("agent_type") == "person" and re.match(
            r"^(?:a |the )?(?:company|manufacturer|corporation|organization)(?:\s|$)|^(?:公司|企业|机构)",
            descriptor,
            re.I,
        ):
            profile.update(
                agent_type="entity",
                entity_type_status="conflict_requires_review",
                source_agent_type="person",
            )
        profile["profile_scope"] = "decision_subject"
        from app.domain.decision_core import legacy_core, public_core, applicable_core
        profile["decision_core"] = public_core(applicable_core(legacy_core(agent), (agent.get("decision_core") or {}).get("as_of", "")))
        behavior = behavior_from_record(agent).model_dump()
        for key in (
            "contextual_roles",
            "capabilities",
            "decision_patterns",
            "objectives",
        ):
            behavior[key] = [v for v in behavior[key] if v["timeline_refs"]]
        profile["behavioral_model"] = behavior
        state = agent.get("public_state") or {}
        semantic = state.get("semantic") if isinstance(state, dict) else {}
        strength = semantic.get("strength") if isinstance(semantic, dict) else {}
        profile["capability_notes"] = (
            [
                {
                    "name": str(k),
                    "description": str(v.get("description") or ""),
                    "status": "model_inferred",
                }
                for k, v in strength.items()
                if k != "as_of" and isinstance(v, dict) and v.get("description")
            ][:10]
            if isinstance(strength, dict)
            else []
        )
    for profile in profiles:
        if profile.get("decision_core"):
            profile.pop("behavioral_model", None)
            profile.pop("capability_notes", None)
    seen = set(principals)
    member_by_name = {}
    used_ids = {a["agent_id"] for a in profiles}
    for agent in agents:
        state = agent.get("public_state") or {}
        if not isinstance(state, dict):
            continue
        for member in state.get("subordinates") or []:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            if not name or name.casefold() in principals:
                continue
            if any(str(p.get("name", "")).casefold() == name.casefold() for p in profiles[:len(agents)]):
                continue
            if name.casefold() in member_by_name:
                previous = member_by_name[name.casefold()]
                parents = list(
                    dict.fromkeys(
                        [
                            *(
                                previous.get("candidate_parent_ids")
                                or [previous.get("parent_actor_id")]
                            ),
                            agent["agent_id"],
                        ]
                    )
                )
                if len(parents) > 1:
                    previous.update(
                        parent_actor_id=None,
                        candidate_parent_ids=parents,
                        membership_status="ambiguous",
                    )
                continue
            seen.add(name.casefold())
            member_id = str(member.get("agent_id") or "")
            if not member_id or member_id in used_ids:
                member_id = stable_agent_id(case_id, title, name)
            used_ids.add(member_id)
            profiles.append(
                {
                    "agent_id": member_id,
                    "name": name,
                    "aliases": [],
                    "agent_type": str(member.get("entity_type") or "person"),
                    "title": str(member.get("title") or ""),
                    "affiliation": str(
                        member.get("affiliation") or agent.get("name") or ""
                    ),
                    "description": str(
                        member.get("role_in_entity")
                        or "Assigned participant; affiliation requires source review"
                    ),
                    "decision_role": "member",
                    "profile_scope": "member_identity",
                    "parent_actor_id": agent["agent_id"],
                    "membership_status": "model_inferred",
                    "membership_evidence": [],
                    "relevant_timeline_refs": [],
                }
            )
            member_by_name[name.casefold()] = profiles[-1]
    return profiles, known


def actor_graph(profiles, case_id, title, *, actor_id="", match_ids=None, limit=80):
    root = "case:" + case_id
    nodes = [{"id": root, "label": title, "type": "case", "detail": {"title": title}}]
    edges = []
    by_id = {p["id"]: p for p in profiles}
    selected = profiles
    if match_ids is not None:
        include = set(match_ids)
        for p in profiles:
            parent = p["metadata"].get("profile", {}).get("parent_actor_id")
            if parent in include:
                include.add(p["id"])
        for identifier in list(include):
            parent = (
                by_id.get(identifier, {})
                .get("metadata", {})
                .get("profile", {})
                .get("parent_actor_id")
            )
            if parent:
                include.add(parent)
        selected = [p for p in profiles if p["id"] in include]
    if actor_id:
        selected = [
            p
            for p in profiles
            if p["id"] == actor_id
            or p["metadata"].get("profile", {}).get("parent_actor_id") == actor_id
        ]
        parent = (by_id.get(actor_id, {}).get("metadata", {}).get("profile") or {}).get(
            "parent_actor_id"
        )
        if parent in by_id:
            selected = [by_id[parent], *selected]
    matched_count = len(selected)
    selected = sorted(
        selected,
        key=lambda p: (
            p["metadata"].get("profile", {}).get("decision_role") != "principal",
            p["title"],
        ),
    )[: max(0, limit - 2)]
    ids = {p["id"] for p in selected}
    other = "group:" + case_id + ":mentioned"
    for p in selected:
        profile = p["metadata"].get("profile", {})
        role = profile.get("decision_role") or (
            "principal" if p["metadata"].get("simulation_agent") else "mentioned"
        )
        parent = profile.get("parent_actor_id")
        nodes.append(
            {
                "id": p["id"],
                "label": p["title"],
                "type": "actor",
                "decision_role": role,
                "parent_actor_id": parent,
                "entity_type": profile.get("agent_type", "entity"),
                "detail": p["metadata"],
            }
        )
        if parent in ids:
            edges.append(
                {
                    "source": parent,
                    "target": p["id"],
                    "label": "Member affiliation (unverified)",
                    "kind": "membership",
                    "status": profile.get("membership_status", "model_inferred"),
                }
            )
        elif role == "principal":
            edges.append(
                {
                    "source": root,
                    "target": p["id"],
                    "label": "Decision actor",
                    "kind": "topic",
                }
            )
        else:
            if not any(n["id"] == other for n in nodes):
                nodes.append(
                    {
                        "id": other,
                        "label": "Other participant with unverified affiliation",
                        "type": "group",
                        "detail": {},
                    }
                )
                edges.append(
                    {
                        "source": root,
                        "target": other,
                        "label": "Related participant",
                        "kind": "topic",
                    }
                )
            edges.append(
                {
                    "source": other,
                    "target": p["id"],
                    "label": "Participant",
                    "kind": "topic",
                }
            )
    return {
        "nodes": nodes,
        "edges": edges,
        "total_actors": len(profiles),
        "truncated": len(selected) < matched_count,
        "expanded_parent_ids": [
            p["metadata"].get("profile", {}).get("parent_actor_id") or other
            for p in selected
            if match_ids
            and p["id"] in match_ids
            and p["metadata"].get("profile", {}).get("decision_role") != "principal"
        ],
        "hierarchy_note": "Member affiliations are inferred from profiles. Command relationships have not been verified.",
    }
