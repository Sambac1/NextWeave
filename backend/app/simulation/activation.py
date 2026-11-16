from __future__ import annotations

import asyncio
from app.llm.concurrency import gather_complete
import json
from typing import Any

from pydantic import BaseModel, Field

from app.domain import ActivationDecision, AgentProfile, SelectedAction, WorldSnapshot
from app.llm.client import LLMClient
from app.simulation.context import public_context, actor_context, activation_cards, compact_json
from app.domain.identity import identity_key


class MasterActivationReview(BaseModel):
    agent_id: str
    master_score: float = Field(ge=0, le=1)
    decision_point_now: bool
    inaction_consistency: str
    feasible_action_exists: bool
    decision_window: str
    cross_agent_dependency: bool
    reasons: list[str]
    evidence_refs: list[str] = Field(default_factory=list)


class MasterActivationBatch(BaseModel):
    round_id: int
    snapshot_version: int
    reviews: list[MasterActivationReview]


class SelfActivationReview(BaseModel):
    agent_id: str
    round_id: int
    snapshot_version: int
    self_score: float = Field(ge=0, le=1)
    decision_required: bool
    feasible_action_exists: bool
    preferred_mode: str
    reasons: list[str]


def _selected_refs(actions: list[SelectedAction]) -> set[str]:
    return {identity_key(value) for action in actions for value in (action.agent_id, action.agent_name) if identity_key(value)}


def _seed_candidates(snapshot: WorldSnapshot, selected_actions: list[SelectedAction]) -> dict[str, dict[str, Any]]:
    selected = _selected_refs(selected_actions)
    hard_types = {"direct_attack", "sanction_imposed", "formal_ultimatum", "legal_deadline", "mandatory_vote", "election_deadline", "required_response"}
    candidates: dict[str, dict[str, Any]] = {}
    for agent in snapshot.agents:
        refs = {identity_key(agent.agent_id), identity_key(agent.name), *(identity_key(x) for x in agent.aliases)}
        sources: list[str] = []
        mandatory = False
        trigger_score = 0.0
        from app.simulation.process import pending_for
        pending = pending_for(snapshot, agent.agent_id)
        if any(item.get('kind') != 'opportunity' for item in pending):
            sources.append('pending_public_response')
            mandatory = True
            trigger_score = 1.0
        elif pending:
            sources.append('public_decision_opportunity')
            trigger_score = 0.65
        # Entering deliberation does not require taking an external action.
        # An explicit wait must be decided, not inferred from a closed gate.
        last_deliberation = agent.activation.last_deliberation_round
        if last_deliberation is None or snapshot.round_id + 1 - last_deliberation >= 3:
            sources.append('initial_deliberation' if last_deliberation is None else 'periodic_deliberation')
            mandatory = True
        if agent.activation.status == "active":
            sources.append("previous_active")
        if any(not item.processed for item in agent.private_observations):
            sources.append("private_information")
        if snapshot.world_state.get("simulation_clock", {}).get("sequential_harness") and any(
            item.received_round == snapshot.round_id and item.received_round > (agent.activation.last_deliberation_round or 0)
            for item in agent.private_observations
        ):
            # BDI processing may already have marked the observation read.
            sources.append("new_private_observation")
            mandatory = True
            trigger_score = 1.0
        if agent.activation.last_information_update_round == snapshot.round_id + 1:
            sources.append("private_state_update")
        if agent.agent_id in snapshot.world_state.get("decision_queue",[]):
            sources.append("configured_decision_deadline")
            mandatory = True
            trigger_score = 1.0
        if refs & selected:
            sources.append("selected_intervention")
            mandatory = True
            trigger_score = 1.0
        new_events = [e for e in snapshot.timeline if e.origin == 'simulated' and e.round_id == snapshot.round_id]
        for event in [*new_events, *snapshot.timeline[-8:]]:
            if refs & {identity_key(value) for value in event.actors}:
                sources.append("event_affected")
                if (event.event_type in hard_types and event.origin == 'simulated'
                        and event.round_id == snapshot.round_id):
                    sources.append("hard_event_trigger")
                    mandatory = True
                    trigger_score = 1.0
                    break
        if agent.activation.last_review_round is None or snapshot.round_id + 1 - agent.activation.last_review_round >= 3:
            sources.append("periodic_review")
        if sources:
            candidates[agent.agent_id] = {"sources": list(dict.fromkeys(sources)), "mandatory": mandatory, "trigger_score": trigger_score}
    return candidates


async def choose_active_agents(
    snapshot: WorldSnapshot,
    selected_actions: list[SelectedAction],
    client: LLMClient,
    call_prefix: str,
) -> tuple[WorldSnapshot, list[AgentProfile], list[ActivationDecision]]:
    if not snapshot.agents:
        return snapshot.model_copy(deep=True), [], []
    working = snapshot.model_copy(deep=True)
    next_round = working.round_id + 1
    public_state = public_context(working, "activation")
    seeds = _seed_candidates(working, selected_actions)
    forced = {aid for aid, seed in seeds.items() if seed['mandatory']}
    if len(forced) < len(working.agents):
        expected = {a.agent_id for a in working.agents} - forced
        call = dict(
            call_name=f"{call_prefix}_activation_master_review",
            system="You are the privileged Master Activation Agent. Use the shared round_briefing, public decision cards and pending-observation counts to decide which actors need review. Identify the concrete decision window, received request, changed condition or deadline that makes review timely. Waiting is valid; state a reconsideration trigger. The tracked question describes what the observer measures, not an objective all actors must pursue. Actor-specific private information stays in its own review. Never copy one agent's private facts into another agent's feedback. Return one overall master_score per agent, not separate weighted sub-scores.",
            prompt=(f"PUBLIC STATE\n{compact_json(public_state)}\n\nACTOR DECISION CARDS\n{compact_json([c for c in activation_cards(working) if c['agent_id'] not in forced])}\n\nPROGRAM TRIGGERS\n{json.dumps(seeds, ensure_ascii=False, indent=2)}\n\nReview only the agents in ACTOR DECISION CARDS, exactly once; exclude program-mandatory actors. Consider a real decision point, whether inaction remains consistent, whether a feasible action exists, whether the window is open, and cross-agent or settlement dependency. Influence alone is not a reason to act. Echo round_id and snapshot_version exactly."),
            schema=MasterActivationBatch,
            temperature=0.1,
            max_tokens=3500,
        )
        for attempt in range(2):
            master = await client.json_call(**call)
            returned = [review.agent_id for review in master.reviews]
            if (len(returned) == len(expected) and set(returned) == expected
                    and master.round_id == next_round and master.snapshot_version == working.snapshot_version):
                break
            if attempt:
                if len(returned) != len(expected) or set(returned) != expected:
                    raise ValueError("Master activation review did not cover the non-mandatory actors exactly once")
                raise ValueError("Master activation review used a stale round or snapshot")
            call["call_name"] = f"{call_prefix}_activation_master_review_repair"
            call["temperature"] = 0
            call["prompt"] += (
                "\n\nSTRUCTURE REPAIR: Return exactly one review for each of these agent IDs and no others: "
                f"{compact_json(sorted(expected))}. Previous IDs: {compact_json(returned)}. "
                f"Use round_id={next_round} and snapshot_version={working.snapshot_version}."
            )
    else:
        master = MasterActivationBatch(round_id=next_round, snapshot_version=working.snapshot_version, reviews=[])
    master.reviews.extend(MasterActivationReview(
        agent_id=aid, master_score=0, decision_point_now=True,
        inaction_consistency='A formal decision is required; waiting remains available',
        feasible_action_exists=True, decision_window='program_trigger',
        cross_agent_dependency=False, reasons=['program_mandatory_deliberation_no_model_gate']) for aid in sorted(forced))
    if master.round_id != next_round or master.snapshot_version != working.snapshot_version:
        raise ValueError("Master activation review used a stale round or snapshot")
    master_by_id = {item.agent_id: item for item in master.reviews}
    if len(master.reviews) != len(master_by_id) or set(master_by_id) != {item.agent_id for item in working.agents}:
        raise ValueError("Master activation review did not review every ordinary agent exactly once")

    candidate_ids = set(seeds)
    candidate_ids.update(item.agent_id for item in master.reviews if item.master_score >= 0.45)

    async def self_review(agent: AgentProfile) -> SelfActivationReview:
        return await client.json_call(
            call_name=f"{call_prefix}_agent_{agent.agent_id}_self_activation",
            system="You are the specified bounded scenario actor reviewing only your own need for a formal decision. Waiting and monitoring are valid. You cannot inspect another agent's private state.",
            prompt=(f"PUBLIC STATE\n{compact_json(public_state)}\n\nYOUR COMPLETE PRIVATE PROFILE\n{compact_json(actor_context(agent, observations=True))}\n\nWHY REVIEWED\n{json.dumps(seeds.get(agent.agent_id, {}), ensure_ascii=False)}\n\nReturn one overall self_score based on beliefs, desires, capabilities, constraints, decision policy, history, and private memory. Do not propose the action yet. Echo all identifiers exactly."),
            schema=SelfActivationReview,
            temperature=0.15,
        )

    candidates = [item for item in working.agents if item.agent_id in candidate_ids]
    self_reviews = await gather_complete(*(self_review(item) for item in candidates if item.agent_id not in forced))
    self_reviews.extend(SelfActivationReview(agent_id=aid, round_id=next_round,
        snapshot_version=working.snapshot_version, self_score=0, decision_required=True,
        feasible_action_exists=True, preferred_mode='deliberate',
        reasons=['program_mandatory_deliberation_no_model_gate']) for aid in sorted(forced))
    self_by_id = {item.agent_id: item for item in self_reviews}
    for item in self_reviews:
        if item.round_id != next_round or item.snapshot_version != working.snapshot_version:
            raise ValueError("Self activation review used a stale round or snapshot")

    scored: list[tuple[float, AgentProfile, ActivationDecision]] = []
    for agent in candidates:
        external = master_by_id[agent.agent_id]
        internal = self_by_id.get(agent.agent_id)
        if not internal:
            raise ValueError(f"Missing self activation review for {agent.agent_id}")
        seed = seeds.get(agent.agent_id, {"sources": ["master_review"], "mandatory": False, "trigger_score": 0.0})
        cooldown_penalty = min(0.20, max(0, agent.activation.consecutive_active_rounds - 2) * 0.05)
        score = max(0.0, min(1.0, 0.45 * external.master_score + 0.40 * internal.self_score + 0.15 * float(seed["trigger_score"]) - cooldown_penalty))
        previous_active = agent.activation.status == "active"
        eligible = bool(seed["mandatory"]) or (
            score >= (0.45 if previous_active else 0.60)
            and external.feasible_action_exists
            and (external.decision_point_now or external.cross_agent_dependency or internal.decision_required)
        )
        scored.append((score, agent, ActivationDecision(
            agent_id=agent.agent_id,
            selected=eligible,
            score=round(score, 4),
            master_score=external.master_score,
            self_score=internal.self_score,
            trigger_score=float(seed["trigger_score"]),
            mandatory=bool(seed["mandatory"]),
            reasons=list(dict.fromkeys([*seed["sources"], *external.reasons, *internal.reasons])),
        )))

    critical = any(row[2].mandatory for row in scored) or bool(working.world_state.get("critical_round"))
    budget = min(len(working.agents), 8 if critical else 5)
    ranked = sorted(scored, key=lambda row: (-int(row[2].mandatory), -int(row[1].activation.status == "active"), -row[0], row[1].agent_id))
    selected_ids = {row[1].agent_id for row in [item for item in ranked if item[2].selected][:budget]}
    decisions: list[ActivationDecision] = []
    for _, _, decision in scored:
        decision.selected = decision.agent_id in selected_ids
        decisions.append(decision)

    decision_by_id = {item.agent_id: item for item in decisions}
    for agent in working.agents:
        decision = decision_by_id.get(agent.agent_id)
        previous_active = agent.activation.status == "active"
        selected = agent.agent_id in selected_ids
        if selected:
            agent.activation.last_deliberation_round = next_round
            agent.activation.status = "active"
            agent.activation.last_active_round = next_round
            agent.activation.consecutive_active_rounds += 1
            agent.activation.inactive_rounds = 0
            agent.activation.cooldown_rounds = 0
        else:
            agent.activation.status = "cooldown" if previous_active else "inactive"
            agent.activation.consecutive_active_rounds = 0
            agent.activation.inactive_rounds += 1
            agent.activation.cooldown_rounds = 1 if previous_active else max(0, agent.activation.cooldown_rounds - 1)
        if decision:
            agent.activation.last_review_round = next_round
            agent.activation.score = decision.score
            agent.activation.master_score = decision.master_score
            agent.activation.self_score = decision.self_score
            agent.activation.trigger_score = decision.trigger_score
            agent.activation.reasons = decision.reasons
            agent.activation.low_score_streak = 0 if decision.score >= 0.45 else agent.activation.low_score_streak + 1
    active = [item for item in working.agents if item.agent_id in selected_ids]
    return working, active, decisions
