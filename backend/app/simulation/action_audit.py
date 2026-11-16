from __future__ import annotations

import asyncio
from app.llm.concurrency import gather_complete
import json
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from app.domain import ActionProposal, AgentProfile, SelectedAction, WorldSnapshot
from app.layers.simulation.executor import annotate_executors
from app.llm.client import LLMClient
from app.simulation.context import audit_context, compact_json, public_context
from app.simulation.protocol import validate_action
from app.simulation.deliberation import generate_agent_action


ProposalFactory = Callable[..., Awaitable[ActionProposal]]


class AuditReview(BaseModel):
    agent_id: str
    accepted: bool
    violation_codes: list[str] = Field(default_factory=list)
    repair_instruction: str = ""


class AuditBatch(BaseModel):
    round_id: int
    snapshot_version: int
    reviews: list[AuditReview]


def _program_violations(proposal: ActionProposal, agent: AgentProfile, snapshot: WorldSnapshot) -> list[str]:
    issues: list[str] = []
    if proposal.agent_id != agent.agent_id or proposal.agent_name != agent.name:
        issues.append("IDENTITY_MISMATCH")
    if proposal.round_id != snapshot.round_id + 1:
        issues.append("ROUND_MISMATCH")
    if proposal.snapshot_version != snapshot.snapshot_version:
        issues.append("SNAPSHOT_MISMATCH")
    if proposal.action_id != f"action_{snapshot.round_id + 1}_{agent.agent_id}":
        issues.append("ACTION_ID_MISMATCH")
    if proposal.target and proposal.target not in {item.agent_id for item in snapshot.agents}:
        issues.append("UNKNOWN_TARGET")
    if not set(proposal.timeline_refs) <= {item.timeline_id for item in snapshot.timeline}:
        issues.append("UNKNOWN_TIMELINE_REFERENCE")
    if agent.retrieval_context:
        supplied = {e["timeline_id"] for e in public_context(snapshot, "decision", agent)["timeline"]}
        if not set(proposal.timeline_refs) <= supplied:
            issues.append("REFERENCE_NOT_IN_SUPPLIED_EVIDENCE")
    if proposal.visibility == "direct" and not proposal.target:
        issues.append("DIRECT_ACTION_REQUIRES_TARGET")
    if proposal.decision in {"wait", "monitor", "no_action"} and proposal.action_type != "no_action":
        issues.append("NO_ACTION_TYPE_REQUIRED")
    if proposal.decision == "act" and proposal.action_type == "no_action":
        issues.append("ACT_REQUIRES_ACTION_TYPE")
    if proposal.plan_update:
        if not set(proposal.plan_update.goal_ids) <= {g.desire_id for g in agent.desires}:
            issues.append("UNKNOWN_PLAN_GOAL")
        supplied = {e["timeline_id"] for e in public_context(snapshot,"decision",agent)["timeline"]}
        if not set(proposal.plan_update.timeline_refs) <= supplied:
            issues.append("UNKNOWN_PLAN_EVIDENCE")
    issues.extend(validate_action(snapshot,proposal))
    return issues


async def _master_audit(snapshot: WorldSnapshot, proposals: list[ActionProposal], client: LLMClient, call_prefix: str) -> dict[str, AuditReview]:
    if not proposals:
        return {}
    result = await client.json_call(
        call_name=f"{call_prefix}_master_action_audit",
        system="You are the permissive Master Action Audit Agent, not the author of actions. Reject only severe identity violations, unavailable private information, impossible timing, capability excess, hard-constraint violations, or direct contradiction of frozen facts. Conservative, surprising, suboptimal, or no_action proposals are valid. Do not rewrite actions and never reveal one ordinary agent's private state to another.",
        prompt=(f"ROUND: {snapshot.round_id + 1}; SNAPSHOT VERSION: {snapshot.snapshot_version}\n\nFROZEN STATE\n{compact_json(audit_context(snapshot, proposals))}\n\nPROGRAM-VALID PROPOSALS\n{json.dumps([item.model_dump() for item in proposals], ensure_ascii=False, indent=2)}\n\nReview every proposal exactly once. Accepted proposals need no violation codes or repair instruction."),
        schema=AuditBatch,
        temperature=0.1,
        max_tokens=3000,
    )
    if result.round_id != snapshot.round_id + 1 or result.snapshot_version != snapshot.snapshot_version:
        raise ValueError("Master action audit used a stale round or snapshot")
    by_id = {item.agent_id: item for item in result.reviews}
    if len(result.reviews) != len(by_id) or set(by_id) != {item.agent_id for item in proposals}:
        raise ValueError("Master action audit did not review every proposal exactly once")
    return by_id


async def produce_audited_actions(
    snapshot: WorldSnapshot,
    active_agents: list[AgentProfile],
    selected_actions: list[SelectedAction],
    client: LLMClient,
    call_prefix: str,
    max_retries: int = 2,
    proposal_factory: ProposalFactory = generate_agent_action,
) -> list[ActionProposal]:
    if not active_agents:
        return []
    agent_by_id = {item.agent_id: item for item in active_agents}
    initial = await gather_complete(*(
        proposal_factory(snapshot, agent, selected_actions, client, call_prefix)
        for agent in active_agents
    ))
    # Ground each proposal in the member that would carry it out, BEFORE the audit.
    initial = await annotate_executors(snapshot, list(initial), client, call_prefix)
    proposals = {item.agent_id: item for item in initial}
    accepted: dict[str, ActionProposal] = {}
    feedback: dict[str, dict[str, Any]] = {}
    for attempt in range(max_retries + 1):
        pending = [agent_id for agent_id in agent_by_id if agent_id not in accepted]
        valid: list[ActionProposal] = []
        for agent_id in pending:
            proposal = proposals.get(agent_id)
            issues = _program_violations(proposal, agent_by_id[agent_id], snapshot) if proposal else ["MISSING_PROPOSAL"]
            if issues:
                feedback[agent_id] = {"violation_codes": issues, "repair_instruction": "Correct the listed identity, schema, visibility, target, or evidence-reference violations."}
            else:
                valid.append(proposal)
        reviews = await _master_audit(snapshot, valid, client, f"{call_prefix}_attempt_{attempt + 1}")
        for proposal in valid:
            review = reviews[proposal.agent_id]
            if review.accepted:
                accepted[proposal.agent_id] = proposal
            else:
                feedback[proposal.agent_id] = review.model_dump(exclude={"agent_id", "accepted"})
        pending = [agent_id for agent_id in agent_by_id if agent_id not in accepted]
        if not pending:
            return [accepted[item.agent_id] for item in active_agents]
        if attempt >= max_retries:
            details = {agent_id: feedback.get(agent_id, {}) for agent_id in pending}
            raise ValueError(f"Ordinary-agent action review failed after {max_retries + 1} attempts: {json.dumps(details, ensure_ascii=False)}")

        async def retry(agent_id: str) -> ActionProposal:
            return await proposal_factory(
                snapshot,
                agent_by_id[agent_id],
                selected_actions,
                client,
                f"{call_prefix}_retry_{attempt + 1}",
                previous_proposal=proposals.get(agent_id),
                review_feedback=feedback.get(agent_id),
            )
        retried = list(await gather_complete(*(retry(agent_id) for agent_id in pending)))
        # A repaired proposal is re-voiced too.
        for proposal in await annotate_executors(snapshot, retried, client, call_prefix):
            proposals[proposal.agent_id] = proposal
    raise AssertionError("unreachable")
