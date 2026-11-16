from __future__ import annotations

import asyncio
from app.llm.concurrency import gather_complete
import json
import logging

from pydantic import BaseModel, Field

from app.domain import AgentBDIPatch, WorldSnapshot
from app.llm.client import LLMClient
from app.simulation.protocol import public_digest
from app.knowledge.okf import digest
from app.simulation.context import information_context, compact_json
from app.simulation.decision_profile import current_constraint_beliefs

logger = logging.getLogger(__name__)


class InformationReview(BaseModel):
    agent_id: str
    round_id: int
    snapshot_version: int
    processed_observation_ids: list[str] = Field(default_factory=list)
    bdi_patch: AgentBDIPatch | None = None
    capability_constraint_claims: list[str] = Field(default_factory=list)
    assessment: str


def apply_goal_statuses(agent, desires, issues, round_id):
    """Only exact IDs or unique, exact descriptions identify an existing goal."""
    import re
    key = lambda text: re.sub(r"\s+", " ", str(text)).strip().casefold()
    original = {g.desire_id: g for g in agent.desires}
    proposals = {}
    for goal in desires:
        identifier = goal.desire_id
        if identifier in original and key(goal.description) != key(original[identifier].description):
            issues.append({"code": "goal_update_rejected", "agent_id": agent.agent_id,
                           "round_id": round_id, "goal_id": identifier, "reason": "goal_description_changed"})
            continue
        if identifier not in original:
            matches = [g.desire_id for g in agent.desires if key(g.description) == key(goal.description)]
            if len(matches) == 1:
                identifier = matches[0]
                issues.append({"code": "goal_id_repaired", "agent_id": agent.agent_id,
                               "round_id": round_id, "goal_id": identifier,
                               "reason": "unique_exact_description_match"})
            else:
                issues.append({"code": "goal_update_rejected", "agent_id": agent.agent_id,
                               "round_id": round_id, "reason": "unknown_goal_id_or_ambiguous_description"})
                continue
        proposals.setdefault(identifier, set()).add(goal.status)
    for identifier, statuses in proposals.items():
        if len(statuses) != 1:
            issues.append({"code": "goal_update_rejected", "agent_id": agent.agent_id,
                           "round_id": round_id, "reason": "conflicting_status_updates", "goal_id": identifier})
            continue
        original[identifier].status = next(iter(statuses))


async def process_agent_information(snapshot: WorldSnapshot, client: LLMClient, call_prefix: str) -> WorldSnapshot:
    """Let each ordinary Agent update its own BDI from public facts and its private inbox."""
    working = snapshot.model_copy(deep=True)
    if working.world_state.get('simulation_clock', {}).get('sequential_harness'):
        # Observer summaries and repeated "unknown" prose are not new facts
        # for an actor. Time windows are still reviewed by activation; private
        # inboxes and actual public events independently trigger BDI review.
        factual_state = {key: value for key, value in working.world_state.items() if key not in {
            'simulation_clock', 'contract_progress', 'contract_outlook', 'forecast_status',
            'program_observations', 'decision_queue',
        }}
        public = {'state': factual_state, 'process_issues': [i.model_dump() for i in working.process_issues],
                  'relationships': [r.model_dump() for r in working.relationships]}
    else:
        public = public_digest(working)
    fingerprint=digest({"state":public,"events":[e.timeline_id for e in working.timeline if e.origin=="simulated"],"program_observations":working.world_state.get("program_observations",[])})
    affected=[]
    for agent in working.agents:
        pending=any(not item.processed for item in agent.private_observations)
        initial_stimulus=(any(o.get("step")==working.round_id+1 for o in working.world_state.get("program_observations",[]))
                          or bool(working.world_state.get("benchmark_contract", {}).get("evidence_packet")
                                  and not agent.last_information_fingerprint))
        changed=(working.round_id>agent.last_processed_world_round or initial_stimulus) and fingerprint!=agent.last_information_fingerprint
        if pending or changed: affected.append(agent)
        else:
            agent.last_information_fingerprint=fingerprint
            agent.last_processed_world_round=working.round_id
    if not affected:
        return working

    allowed_by_agent = {}
    async def review(agent):
        bundle = information_context(working, agent)
        provided_ids = set(bundle['allowed_timeline_refs'])
        allowed_by_agent[agent.agent_id] = provided_ids
        call = dict(
            call_name=f"{call_prefix}_agent_{agent.agent_id}_information_review",
            system=(
                "You are the named scenario decision subject. Interpret new public facts and only your own private "
                "information. Update your beliefs or relatively stable desires only when those facts materially change "
                "them. Capabilities and constraints cannot be changed here; report claims for later public verification."
                " Reconcile earlier information-absence judgments with the shared public source packet. A dossier's "
                "coverage gap is not proof of absence. Read table columns, groups, units and periods before updating "
                "beliefs. Rechecking an old judgment against present evidence does not require a new historical event. "
                "Source evidence IDs are not timeline IDs; leave timeline_refs empty for a source-only judgment."
                " A historical restriction is context; do not assert that it is current or applies to this actor "
                "unless the actor's supplied limits establish its scope and validity. For a current restriction, "
                "set constraint_ref to that supplied limit ID. If scope or validity is unclear, state the uncertainty. "
                "A background record is a source report and cannot establish a dated event or current rule."
            ),
            prompt=(
                f"YOUR EVIDENCE AND PRIVATE INFORMATION BUNDLE\n{compact_json(bundle)}\n\n"
                "Echo agent_id, next round_id, and snapshot_version exactly.\n\n"
                "bdi_patch holds two SIBLING fields -- put them side by side, not one inside the other:\n"
                '  "bdi_patch": {"beliefs": {"situation_views": [...], "actor_models": [...]}, "desires": [...]}\n'
                "Leave either one out when this round changes nothing in it.\n\n"
                "For desires, copy existing desire_id values EXACTLY and preserve the original descriptions. "
                "Only status may change. Do not create, rename or renumber long-term goals. "
                "Put a revised action strategy in beliefs for the next decision, not in a replacement goal.\n"
                f"Existing goal IDs and descriptions: {compact_json([{'desire_id':g.desire_id,'description':g.description} for g in agent.desires])}\n"
                "processed_observation_ids lists ONLY observation_id values from your_unprocessed_private_information. "
                "Timeline IDs are public event references, never observation IDs. "
                "Copy the ids exactly; leave the list empty when you processed nothing."
                f"\nBelief timeline_refs may ONLY copy IDs from this allowlist: {json.dumps(sorted(provided_ids))}. Never invent, abbreviate or renumber an ID."
            ),
            schema=InformationReview,
            temperature=0.15,
        )
        for attempt in range(2):
            result = await client.json_call(**call)
            if (result.agent_id == agent.agent_id and result.round_id == working.round_id + 1
                    and result.snapshot_version == working.snapshot_version):
                return result
            if attempt:
                raise ValueError("Agent information review returned stale or mismatched identifiers after targeted repair")
            call = {**call, "call_name": call['call_name'] + '_identifier_repair', "temperature": 0,
                    "prompt": call['prompt'] + f"\nREPAIR IDENTIFIERS: agent_id={agent.agent_id}; round_id={working.round_id+1}; snapshot_version={working.snapshot_version}. Return the complete corrected review; do not change evidence IDs."}

    reviews = await gather_complete(*(review(agent) for agent in affected))
    known_timeline_ids = {item.timeline_id for item in working.timeline}
    for agent, review in zip(affected, reviews):
        if (
            review.agent_id != agent.agent_id
            or review.round_id != working.round_id + 1
            or review.snapshot_version != working.snapshot_version
        ):
            raise ValueError("Agent information review returned stale or mismatched identifiers")
        allowed = {item.observation_id for item in agent.private_observations if not item.processed}
        processed = set(review.processed_observation_ids)
        unknown = processed - allowed
        if unknown:
            logger.warning(
                "Agent %s listed %d observation id(s) it does not hold (%s); ignoring them",
                agent.agent_id, len(unknown), ", ".join(sorted(unknown)[:3]),
            )
            processed &= allowed
        if review.bdi_patch:
            patch = review.bdi_patch
            refs = {r for b in [*(patch.beliefs.situation_views if patch.beliefs else []), *(patch.beliefs.actor_models if patch.beliefs else [])] for r in b.timeline_refs}
            unknown_refs = refs - allowed_by_agent[agent.agent_id]
            if unknown_refs:
                # Reject the whole dependent update, rather than stripping citations
                # and promoting its unsupported assertion to a valid belief.
                working.validation_issues.append({"code": "bdi_patch_rejected", "agent_id": agent.agent_id,
                    "round_id": working.round_id + 1, "unknown_timeline_refs": sorted(unknown_refs),
                    "reason": "reference_not_in_supplied_evidence",
                    "fallback": "previous_bdi_preserved"})
                logger.warning("Rejected BDI update for %s: %d unknown references; previous state preserved", agent.agent_id, len(unknown_refs))
                continue
            # ``desires`` arrived nested under beliefs. Both spellings mean the
            # same update, so normalise before applying rather than dropping
            # whichever one the model happened to pick.
            desires = patch.desires
            if desires is None and patch.beliefs is not None:
                desires = patch.beliefs.desires
            if patch.beliefs is not None:
                for belief in [*patch.beliefs.situation_views, *patch.beliefs.actor_models]:
                    if not set(belief.timeline_refs) <= known_timeline_ids:
                        raise ValueError("Agent BDI patch contains an unknown timeline reference")
                # Scrub the tolerated nesting before it lands on the profile:
                # the stored beliefs must carry beliefs only, or the next round
                # re-sends a block the schema does not describe that way.
                clock = working.world_state.get("simulation_clock", {})
                on_date = clock.get("interval_start") or clock.get("start_date") or ""
                updated, rejected = current_constraint_beliefs(patch.beliefs, agent.decision_core, on_date)
                if rejected:
                    prior, _ = current_constraint_beliefs(agent.beliefs, agent.decision_core, on_date)
                    # Retain unrelated accepted updates and prior safe views if
                    # an entire group in the new patch lacked usable support.
                    for field in ("situation_views", "actor_models"):
                        if getattr(patch.beliefs, field) and not getattr(updated, field):
                            setattr(updated, field, getattr(prior, field))
                    working.validation_issues.append({"code": "bdi_constraint_claim_rejected",
                        "agent_id": agent.agent_id, "round_id": working.round_id + 1, "reasons": rejected})
                agent.beliefs = updated.model_copy(update={"desires": None})
            if desires is not None:
                if agent.decision_core is not None:
                    apply_goal_statuses(agent, desires, working.validation_issues, working.round_id + 1)
                else:
                    agent.desires = desires
        for observation in agent.private_observations:
            if observation.observation_id in processed:
                observation.processed = True
                observation.processed_round = working.round_id + 1
        agent.capability_constraint_claims.extend(
            {"claim": claim, "reported_round": working.round_id + 1}
            for claim in review.capability_constraint_claims
        )
        agent.activation.last_information_update_round = working.round_id + 1
        agent.last_processed_world_round = working.round_id
        agent.last_information_fingerprint=fingerprint
    return working
