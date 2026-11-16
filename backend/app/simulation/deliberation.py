from __future__ import annotations

import json
from typing import Any

from app.domain import ActionProposal, AgentProfile, SelectedAction, WorldSnapshot
from app.llm.client import LLMClient
from app.simulation.context import public_context, actor_context, compact_json
from app.domain.identity import identity_key


def _public_profile(agent: AgentProfile) -> dict[str, Any]:
    return {
        "agent_id": agent.agent_id, "name": agent.name, "aliases": agent.aliases,
        "agent_type": agent.agent_type, "decision_subject": agent.decision_subject,
        "role": agent.role, "title": agent.title, "affiliation": agent.affiliation,
        "description": agent.description, "historical_actions": agent.historical_actions[-8:],
    }


def _visible_interventions(actions: list[SelectedAction], agent: AgentProfile) -> list[dict[str, Any]]:
    refs = {identity_key(agent.agent_id), identity_key(agent.name), *(identity_key(item) for item in agent.aliases)}
    visible = []
    for action in actions:
        owner = identity_key(action.agent_id) in refs or identity_key(action.agent_name) in refs
        recipient = identity_key(action.target) in refs
        if action.visibility == "public" or owner or (action.visibility == "direct" and recipient):
            visible.append(action.model_dump())
    return visible


def _relevant_timeline(snapshot: WorldSnapshot, agent: AgentProfile) -> list[dict[str, Any]]:
    refs = {identity_key(agent.agent_id), identity_key(agent.name), *(identity_key(item) for item in agent.aliases)}
    relevant = [item for item in snapshot.timeline if refs & {identity_key(actor) for actor in item.actors}]
    unique = {item.timeline_id: item for item in [*relevant[-15:], *snapshot.timeline[-5:]]}
    return [item.model_dump() for item in unique.values()]


async def generate_agent_action(
    snapshot: WorldSnapshot,
    agent: AgentProfile,
    selected_actions: list[SelectedAction],
    client: LLMClient,
    call_prefix: str,
    previous_proposal: ActionProposal | None = None,
    review_feedback: dict[str, Any] | None = None,
) -> ActionProposal:
    expected_action_id = f"action_{snapshot.round_id + 1}_{agent.agent_id}"
    retry_context = ""
    if previous_proposal is not None or review_feedback:
        retry_context = (
            f"\n\nYOUR REJECTED PROPOSAL\n{json.dumps(previous_proposal.model_dump() if previous_proposal else {}, ensure_ascii=False, indent=2)}"
            f"\n\nREVIEW FEEDBACK\n{json.dumps(review_feedback or {}, ensure_ascii=False, indent=2)}"
        )
    return await client.json_call(
        call_name=f"{call_prefix}_agent_{agent.agent_id}_decision",
        system=(
            "You are the specified bounded scenario decision subject, not an omniscient narrator. Decide only your "
            "own next move from your beliefs, desires, capabilities, constraints, policy, history, and private memory. "
            "You cannot see other agents' private state or their unsettled actions this round. Acting, waiting, monitoring, "
            "and no_action are all valid. Never claim capabilities you do not have. "
            "For a compact dossier, use canonical means, resources and limits rather than numerical personality priors. "
            "Review your current_plan before choosing: continue it unless completed, blocked or new evidence justifies change. "
            "Inspect public_process_issues for requests addressed to you. Respond, decline or explicitly wait with a concrete review condition; do not merely rephrase the previous proposal. "
            "Check elapsed time, validity dates and unmet conditions. A dated arrangement is not renewed by repeating its old description. An expired claim is not a continuing hard constraint. "
            "Optionally return plan_update with the pursued goal_ids, status, review_when and change_reason. Keep plan_update.description concise (aim for 500 characters), preserving essential conditions; avoid repeating the entire negotiation history. "
            "A plan is revisable, not a forced script. Plan goal_ids must refer to your current goals, citations to supplied events. "
            "Use contextual behavioral hypotheses with their evidence and uncertainty; do not derive hostility, loyalty, risk appetite or fixed strategy from a legacy stance label. Active initiative is distinct from risk tolerance or response speed. Roles can vary by counterpart and issue."
        ),
        prompt=(
            f"EVENT: {snapshot.event_title}\nROUND: {snapshot.round_id + 1}\nSNAPSHOT VERSION: {snapshot.snapshot_version}\n"
            f"PUBLIC DECISION CONTEXT\n{compact_json(public_context(snapshot, 'decision', agent))}\n\n"
            f"YOUR PRIVATE DECISION STATE\n{compact_json(actor_context(agent, observations=True))}\n\n"
            f"VISIBLE USER INTERVENTIONS\n{json.dumps(_visible_interventions(selected_actions, agent), ensure_ascii=False, indent=2)}"
            f"{retry_context}\n\nReturn agent_id={agent.agent_id}, agent_name={agent.name}, round_id={snapshot.round_id + 1}, "
            f"snapshot_version={snapshot.snapshot_version}, action_id={expected_action_id} exactly. Target must be an "
            "existing agent_id or empty. timeline_refs must be known timeline IDs. For wait/monitor/no_action, use "
            "action_type=no_action and do not fabricate an external act."
        ),
        schema=ActionProposal,
        temperature=0.7,
        max_tokens=2500,
    )
