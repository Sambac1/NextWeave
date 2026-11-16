"""Functional Claude Agent SDK roles used by one simulation round."""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError, ValidationInfo, field_validator

from app.domain import (
    ActionProposal,
    GameMasterSettlement,
    SelectedAction,
    WorldSnapshot,
)
from app.domain.forecast import ActionResult, SequentialGameMasterSettlement
from app.domain.process import IssueUpdate
from app.llm.client import LLMClient, LLMError, StructuredOutputError
from app.simulation.context import public_context, compact_json, cited_public_context
from app.simulation.protocol import program_projection, validate_program_projection
from app.simulation.action_audit import produce_audited_actions
from app.simulation.activation import choose_active_agents
from app.simulation.deliberation import generate_agent_action
from app.simulation.bdi_updates import process_agent_information
from app.simulation.settlement_repair import state_change_repair_context
from app.simulation.transitions import project_world_state

logger = logging.getLogger(__name__)


class _LegacySettlementFormatRepair(GameMasterSettlement):
    @field_validator("issue_updates", "action_results", mode="before")
    @classmethod
    def retain_well_formed_annotations(cls, updates, info: ValidationInfo):
        if not isinstance(updates, list):
            return updates
        schema = IssueUpdate if info.field_name == "issue_updates" else ActionResult
        retained = []
        for update in updates:
            try:
                retained.append(schema.model_validate(update))
            except ValidationError:
                continue
        return retained


class ActivationMasterAgent:
    """Select ordinary Agents that face a real decision point this round."""

    def __init__(self, runtime: LLMClient) -> None:
        self.runtime = runtime

    async def select(self, snapshot: WorldSnapshot, interventions: list[SelectedAction], call_prefix: str):
        return await choose_active_agents(snapshot, interventions, self.runtime, call_prefix)


class OrdinaryBDIAgent:
    """Run one ordinary Agent's private BDI update and bounded action decision."""

    def __init__(self, runtime: LLMClient) -> None:
        self.runtime = runtime

    async def process_information(self, snapshot: WorldSnapshot, call_prefix: str) -> WorldSnapshot:
        return await process_agent_information(snapshot, self.runtime, call_prefix)

    async def decide(
        self,
        snapshot: WorldSnapshot,
        agent,
        interventions: list[SelectedAction],
        call_prefix: str,
        previous_proposal: ActionProposal | None = None,
        review_feedback: dict | None = None,
    ) -> ActionProposal:
        return await generate_agent_action(
            snapshot,
            agent,
            interventions,
            self.runtime,
            call_prefix,
            previous_proposal,
            review_feedback,
        )


class ActionAuditMasterAgent:
    """Validate proposals and retry only the ordinary Agents that violate hard rules."""

    def __init__(self, runtime: LLMClient, ordinary_agent: OrdinaryBDIAgent) -> None:
        self.runtime = runtime
        self.ordinary_agent = ordinary_agent

    async def produce(
        self,
        snapshot: WorldSnapshot,
        active_agents,
        interventions: list[SelectedAction],
        call_prefix: str,
    ) -> list[ActionProposal]:
        async def propose(
            proposal_snapshot,
            agent,
            proposal_interventions,
            _runtime,
            proposal_prefix,
            previous_proposal=None,
            review_feedback=None,
        ):
            async def generate():
                return await self.ordinary_agent.decide(
                    proposal_snapshot,
                    agent,
                    proposal_interventions,
                    proposal_prefix,
                    previous_proposal,
                    review_feedback,
                )
            cache = getattr(self, 'proposal_cache', None)
            if cache is None:
                return await generate()
            return await cache.model('proposal', {'snapshot': proposal_snapshot.model_dump(),
                'actor_id': agent.agent_id, 'interventions': [i.model_dump() for i in proposal_interventions],
                'previous': previous_proposal.model_dump() if previous_proposal else None,
                'feedback': review_feedback}, generate, ActionProposal)


        return await produce_audited_actions(
            snapshot,
            active_agents,
            interventions,
            self.runtime,
            call_prefix,
            proposal_factory=propose,
        )


class GameMasterAgent:
    """Resolve audited actions and own the semantic update of World State."""

    def __init__(self, runtime: LLMClient) -> None:
        self.runtime = runtime
        self.weighted_decisions = False
        self.sequential_outcome = False
        self.main_forecast = False
        self.terminal_dimensions = []
        self.prior_measurement_states = {}

    async def settle(
        self,
        snapshot: WorldSnapshot,
        actions: list[ActionProposal],
        interventions: list[SelectedAction],
        call_prefix: str,
    ) -> GameMasterSettlement:
        next_round = snapshot.round_id + 1
        user_actions = [
            {
                **item.model_dump(),
                "action_id": f"intervention_{next_round}_{index:02d}",
            }
            for index, item in enumerate(interventions, 1)
        ]
        public_snapshot = cited_public_context(snapshot, "settlement", actions)
        program_owned = program_projection(snapshot, actions)
        public_snapshot["program_owned_next_values"] = program_owned
        state_field = "state_changes" if self.sequential_outcome else "next_world_state"
        state_instructions = (
            "Return state_changes as a list of changes to durable public state; return [] if none. Omit next_world_state. "
            "For each change include the path as an array of object keys, operation, exact before and after values, "
            "Paths are relative to the supplied snapshot.world_state object: do not add an extra world_state wrapper. "
            "Read the actual parent and leaf types from that object. If a stored field is a string or null, do not "
            "invent child paths beneath it; an evidenced replacement must target that existing field and copy its exact before-value. "
            "brief reason, event_ref and a verbatim evidence_quote from a new public event in this round. Use add only "
            "when the key does not yet exist (before=null), replace for a changed value, and remove for withdrawal or "
            "cancellation (after=null). A path parent must already exist unless the joint protocol explicitly permits "
            "creating absent object containers for an evidenced add. Consolidate each path; never write both a parent "
            "and one of its descendants. For a dated object whose existing expires_on is at or before interval_end, "
            "expire with after=null and omit event_ref and evidence_quote. For a registered measurement whose current value "
            "is no longer known, replace with after=null and state why; the missing observation does not require a new "
            "event. Unmentioned paths carry forward. Omit all program-owned paths and metadata; the engine applies those. "
        ) if self.sequential_outcome else (
            "In next_world_state provide only changed durable fields; omission preserves existing values. "
            "Program-owned next values are authoritative; copy them into the indicated paths of next_world_state exactly. "
        )
        # The event's shared strength dimensions. Every principal is rated on the
        # same set, so a change is comparable across actors. They live on the
        # snapshot so a round cannot drop them.
        strength_dimensions = list(snapshot.strength_dimensions)
        if not strength_dimensions and any(a.decision_core is None for a in snapshot.agents):
            # An empty list is not a no-op: the prompt tells the model to use exactly
            # these names and no others, so with nothing to name, every verdict it
            # returns is discarded on the way in.
            logger.warning(
                "Settlement for %s has no strength dimensions: no agent profile "
                "declared a strength block, so every power_impact this round will "
                "be discarded. Capability levels will not move.",
                snapshot.event_title,
            )
        process_instructions = (
                "Track public issue_updates using only process_topics IDs/titles when supplied. Each update states the exact prior tracked before-state (empty for a new issue), "
                "a concrete after-state of at most 500 characters, change kind, actor_ids, event_refs, an exact evidence_quote from those public events, and pending_actor_ids who still owe a response. "
                "For a changed after-state, use a concise status plus the material terms and pending conditions. Put detailed narrative in public_events. "
                "Use cause_action_ids only from PUBLIC_CAUSE_ACTION_ID_ALLOWLIST. Private/direct actions may cause public consequences, but omit their IDs from public events and issue updates; cite the public event via event_refs and evidence_quote instead. A proposal, conditional acceptance and implementation are distinct states. For kind=unchanged, before and after must exactly match an existing tracked state; omit an unknown or untracked unchanged issue. "
                "Consolidate each issue to at most one update per round. Preserve unresolved issues; do not force progress or a particular outcome. "
                "Do not invent deadlines, resource rates or external shocks. Inspect validity dates and the clock before perpetuating temporary arrangements. "
                "New public events must reflect a new action or concrete consequence. Mark change_kind and cause_action_ids when available. "
                "Repeated positions or an unchanged blockade are continuing_state_notes, not newly occurring events. "
                "With no new actions, a new event needs an actual configured observation or a recorded prior action causing a delayed consequence. "
        ) if snapshot.world_state.get('process_tracking_version') == 'issue-process-1' else ''
        selected_action_instructions = (
            "AUDITED ACTIONS contains only the selected action for each deliberating actor. Resolve those actions as given. "
            "When an action produces a concrete observable effect, record the effect where it can inform the next round: "
            "public_events for public facts, a tracked issue_update when the public issue state or outstanding response changes, "
            f"{state_field} for durable scenario state, and agent_knowledge_updates for new recipient-specific private observations. "
            "Preserve decision-relevant recipients, terms, confirmed conditions and outstanding responses where supported. "
            "For example, an invitation that was sent leaves a request with a recipient and a response status; it does not make the meeting occur. "
            "A deliberate wait preserves existing open requests unless a separate observable development resolves one. "
            "Use the structured decision field: wait, monitor and no_action do not authorize a new outward act. "
            "A public audience named only in the action text is not a registered recipient; do not assign it private knowledge or generate its response. "
            "Distinguish attempts, partial execution, blocked execution and completed effects in the applicable records. "
            "An action can legitimately produce no new public effect. Do not invent a state change to distinguish trajectories "
            "or publish details that only a private recipient would know. "
        ) if self.weighted_decisions else ''
        contract_instructions = (
            "For this forecast, use benchmark_contract to track the causal drivers that can change "
            "the answer. Historical excerpts, including numeric tables, are fixed pre-cutoff inputs. "
            "Keep a short public contract_progress string "
            + ("as a separate top-level field, " if self.sequential_outcome else "in next_world_state, ")
            + "including pending prerequisites "
            "and any concrete branch-specific change to the measured outcome. Do not repeat campaign or negotiating "
            "statements as changes in the outcome. When an explicitly scheduled milestone falls inside this round's "
            "clock window, resolve its public consequences even if no actor was activated. "
            "The benchmark_contract object is maintained by the engine. Omit it from your output. "
        ) if snapshot.world_state.get("benchmark_contract") else ""
        if snapshot.world_state.get("benchmark_contract", {}).get("forecast_criteria"):
            contract_instructions += (
                "Mark target_event or negative_outcome as met only when a settled, public observation explicitly "
                "satisfies the corresponding frozen Yes or No criterion. Cite the event and quote its decisive finding. "
                "A rejected attempt, no activity, "
                "an expired date with no observation, or missing evidence alone cannot establish No. "
                "Keep the condition unresolved when the required outcome or window is uncertain. "
            )
        if snapshot.world_state.get("benchmark_contract", {}).get("decision_dependencies"):
            contract_instructions += (
                "Use decision_dependencies to track independent prerequisites. A represented decision is a chance to act, "
                "not evidence it was made. Conditional participants may communicate, request or review within their cited role; "
                "they cannot finalize a different actor's authorization or make binding commitments on its behalf. "
                "Keep unbound or unobserved prerequisites pending unless valid current-round evidence establishes them. "
            )
        if contract_instructions and not self.sequential_outcome:
            contract_instructions += (
                "For a scheduled election, follow contract_milestone_phase. Never announce a count during pre_count. "
                "During the count window, report a simulated public count only when the rules, dated support and "
                "developments in this trajectory support it. Never present a simulated value as historical evidence. "
            )
        if snapshot.world_state.get("benchmark_contract", {}).get("spec", {}).get("stages"):
            contract_instructions += (
                ("Return contract_updates as a top-level field " if self.sequential_outcome else "In next_world_state, add contract_updates ")
                + "only for changes supported by a public event this round. "
                "Each item has stage (from benchmark_contract.spec.stages), event_ref and verbatim quote. "
                "Use the actual action for the contract's exact subject and scope. Plans and cancellations do not show actual occurrence. "
                "For duration, provide covered_from and covered_through (ISO dates) only if the public evidence explicitly "
                "establishes continuous coverage of the interval. Silence does not establish coverage. "
                "Use exact dates for contract events when known. Preserve unresolved prerequisites. "
            )
        if self.sequential_outcome:
            contract_instructions += (
                "For the frozen question, identify the exact subject, required event or observable, and its "
                "outstanding conditions. Settle how the selected actions affect those conditions this round. "
                "Record any supported public effect in public_events and durable evidence-backed state_changes. "
                "If no relevant condition moved, preserve the existing state and describe the limitation briefly. "
                "For a seat ranking, coalition rhetoric is not a change in support, turnout, or seat allocation. "
                "Omit contract_outlook. A separate readout uses only accepted public facts. "
                "The engine records the opening of a scheduled count window. Do not restate it as an event. "
                "Without an explicitly configured count mechanism, do not create an observed vote total, "
                "seat allocation or rank. Record only supported actions and their observable effects. "
            )
        if self.main_forecast:
            contract_instructions += (
                "In this forecast, apply only the actual selected step. An attempt or request can produce its "
                "own effect, such as submitting an application, but cannot supply its own authorization, response, "
                "or downstream outcome. An action waiting for an unverified prerequisite remains waiting. "
                "Do not attribute its contemplated effect to an unrelated action or to an earlier wait. "
                "Record only observed execution or explicitly configured observations. Keep the event date at "
                "its known precision; do not invent a time of day to pass an outcome boundary. "
                "A quiet or unchanged round cannot close a full-window negative criterion."
            )
        from app.simulation.joint_settlement import enabled, recorded_causal_actions
        if enabled(snapshot):
            contract_instructions += (
                "Use the shared round_briefing. First resolve each selected action against the same round-start state. "
                "Return action_results covering every selected action_id exactly once, including user interventions. "
                "For each give status (executed, partial, blocked, pending, waiting, unknown), a short summary, and "
                "event_refs for public consequences. For a private consequence, private_observation_quote must copy "
                "information actually delivered in agent_knowledge_updates to the actor or its registered target. "
                "Copy the private observation verbatim from the delivered text. If execution has neither a public "
                "consequence nor a delivered private observation, record status=unknown. "
                "Every model-generated public event must be referenced by an action_result or identify recorded "
                "selected/prior cause_action_ids. Put executed_decisions INSIDE each public_events item, never at the "
                "response root or inside action_results. For each public event it names each "
                "actor_id that made a decision, its corresponding recorded action_id, and an exact quote from that "
                "event beginning with the registered actor name followed by its action. Targets or affected parties "
                "are not automatically executors. "
                "Copy this quote from the public event description you wrote, never from the selected action or "
                "an imperative paraphrase. A leading 'The' may be outside the quoted span. "
                "Never attribute a fresh decision to a waiting/unselected recipient. "
                "Program-scheduled observations are supplied by the engine. "
                "A wait, monitor, or no_action has status=waiting and creates no outward action. "
                "Then return interactions for material support, conflict, dependency, competition for resources, or "
                "delayed effects. Name 2-6 distinct recorded action_ids including at least one selected action, the "
                "precondition, when it was available (round_start, precommitted, new_this_round), outcome "
                "(realized, pending, unresolved), and the effect. Realized effects need event_refs and evidence_quote, "
                "or a private_observation_quote copied from information delivered to a participating actor. "
                "Actions sharing a resource or issue need joint resolution. Independent actions require no interaction row. "
                "A newly delivered request cannot create an unchosen response from its recipient. Leave that decision "
                "pending for the next round. Use precommitted only when an actual selected action includes its condition. "
                "Known authority, resources, deadlines and program-owned rules constrain the result. Do not invent "
                "new resource quantities or response decisions. Treat expected_effect as the actor's expectation. "
                "Finally return question_updates for EVERY round_briefing.question_conditions ID. Copy its current "
                "status into before, set after (unknown, pending, met, unmet, blocked), summarize the public condition, "
                "and list pending_actor_ids. Changed conditions need event_refs and a verbatim evidence_quote. "
                "For each NEW pending actor return a pending_decision with the registered actor_id, event_ref, "
                "a verbatim public clause naming the recipient, and choice describing what they can decide. "
                "Use response for an actually delivered request, decision_window for a verified actionable deadline, "
                "and opportunity for a named counterpart's optional decision following a conditional public signal. "
                "Supply review_by only if the exact calendar date is in the quoted clause. "
                "Merely lacking a poll, count or outcome observation "
                "does not make an actor owe a response; keep pending_actor_ids empty for observer uncertainty. "
                "Unknown observations stay unknown; a probability estimate does not establish a fact. For unchanged "
                "conditions without new evidence, use event_refs=[], evidence_quote='' and preserve pending_actor_ids. "
                "Keep all summaries concise. Use interactions=[] when none are material. "
                "Keep each action_results summary within 300 characters. "
                "For state_changes operation=add, absent intermediate object containers can be created for the "
                "new leaf. Existing scalar or null parents cannot be replaced implicitly. Keep before=null for "
                "a genuinely new leaf and cite its public evidence. "
            )
        call = dict(
            call_name=f"{call_prefix}_game_master_settlement",
            system=(
                "You are the Game Master and authoritative environment resolver. Audited ordinary-Agent actions are "
                "fixed inputs: do not reject, replace, or rewrite them. Resolve simultaneous conflicts, feasibility, "
                "success, consequences, and the resulting public state changes. Generate multiple atomic behavioral events "
                "when multiple behaviors occur: one concrete behavior or outcome per public event, never a bundled "
                "summary of unrelated actions. Add activity_annotations to public events using domain-neutral family and stage labels, "
                "with at most two annotations per event and a short supporting clause copied verbatim from that event description. Distinguish proposals, discussions, conditional acceptance, "
                "unconditional agreement, announcements, execution and completed outcomes. A promise to do something is not its "
                "execution; conditional or in-principle acceptance is not a completed agreement. Annotate only settled public "
                "facts, never private plans. Empty annotations are allowed when the stage is unclear. "
                f"{process_instructions}"
                f"{selected_action_instructions}"
                f"{contract_instructions}"
                "Generate public events and, for every known "
                "ordinary Agent, list only the private information that Agent should know after settlement. "
                f"{state_instructions}"
                "For compact dossiers, preserve actor_resources resource IDs and units; value may be a finite number or null when unknown. An unknown quantity must stay null unless an explicit program mechanism supplies it. "
                "Explain resource changes through this round's consequences, never infer depletion from waiting. Do not invent capability scores. "
                "Preserve every path listed in world_state.measurement_paths; use an explicit null when its new value is unknown. "
                "Also judge whether any Agent's standing in the event's strength dimensions changed this round, stating each "
                "change directly. Never update beliefs, desires, profiles, relationships, or the Agent Registry."
            ),
            prompt=(
                f"EVENT: {snapshot.event_title}\nROUND: {next_round}\nSNAPSHOT VERSION: {snapshot.snapshot_version}\n\n"
                f"FROZEN PUBLIC SNAPSHOT\n{compact_json(public_snapshot)}\n\n"
                f"USER INTERVENTIONS\n{json.dumps(user_actions, ensure_ascii=False, indent=2)}\n\n"
                f"AUDITED ACTIONS\n{json.dumps([item.model_dump(exclude={"plan_update"}) for item in actions], ensure_ascii=False, indent=2)}\n\n"
                f"STRENGTH DIMENSIONS\n{json.dumps(strength_dimensions, ensure_ascii=False)}\n\n"
                f"REGISTERED AGENT IDS\n{compact_json([actor.agent_id for actor in snapshot.agents])}\n\n"
                f"Public event IDs must be unique and start with sim_{next_round}_. Use only known agent_ids as event "
                "actors. Use the simulation_clock interval: an exact event date must be inside (interval_start, interval_end]; "
                "Use interval_start_utc and interval_end_utc when present as the exact observation window. "
                "When the exact day is unknown, use a month label and let the engine retain the interval, never invent a day. "
                "Return exactly one agent_knowledge_updates entry for each REGISTERED AGENT ID and no other IDs. "
                "Historical participant IDs in evidence are not part of the registry. Use an empty information "
                f"list when the Agent learns nothing private. Public facts belong only in {state_field} and public_events.\n\n"
                "For power_impacts: judge only changes in scenario-relevant capacity supported by this round's actions, "
                "such as depleted resources, changed access, production limits or improved coordination. Do not infer a loss from waiting alone. Use "
                "EXACTLY the dimension names listed under STRENGTH DIMENSIONS, no others; an impact in an unlisted "
                "dimension is discarded. State each change directly as strengthened or weakened with the reason; give "
                "at most one impact per Agent per dimension this round. Leave the list empty when nothing changed."
            ),
            schema=SequentialGameMasterSettlement if self.sequential_outcome else GameMasterSettlement,
            temperature=0.4,
            max_tokens=(6500 if enabled(snapshot) else 4600) if getattr(self, "compact_benchmark", False) else 8000,
        )
        if enabled(snapshot):
            output_schema = SequentialGameMasterSettlement if self.sequential_outcome else GameMasterSettlement
            call['prompt'] += '\nALLOWED RESPONSE ROOT FIELDS\n' + json.dumps(list(output_schema.model_fields))
            call["prompt"] += (
                "\nRECORDED CAUSAL ACTIONS (Game Master only)\n"
                + compact_json(recorded_causal_actions(snapshot))
                + "\nThese are prior attempts, not guaranteed outcomes. Use them only for supported delayed "
                "effects or dependencies. Private records remain private; public consequences must describe only "
                "what became observable. Never infer an unchosen response from a recipient."
            )
        if self.terminal_dimensions:
            call["prompt"] += (
                "\nFROZEN OBSERVATION CONTRACT AT THIS ROUND END\n" + compact_json(self.terminal_dimensions)
                + "\nFor standing/boundary dimensions return state_observations, one per known dimension. "
                "Return dimension_id, category_id, actor_ids, activity stage and a verbatim evidence quote. "
                "Use event_ref for a retained public event; for an independently supported boundary status with "
                "no new public event, quote an explicit public continuing_state_notes statement instead. "
                "Only use public settled consequences and supported state: do not recast private negotiations as public, "
                "repeat events, or turn proposals, threats or silence into execution or into a confirmed negative. "
                "For standing states use expires_on (first invalid date) only if established, or until_superseded=true "
                "only for a standing rule/measure explicitly in force until replaced or withdrawn. A temporary "
                "measure of uncertain validity gets neither field. Boundary observations never carry forward. "
                "Do not invent measured effects, values or validity to fill gaps. Omit unobserved dimensions. "
                "Legacy dimensions may use terminal_observations with dimension_id, event_ref, stage and exact public quote."
                "\nPREVIOUS VALIDATED MEASUREMENTS (may be stale or expired; never proof of a new observation)\n"
                + compact_json(self.prior_measurement_states)
            )
        allowed_causes = sorted({a.action_id for a in actions if a.visibility == 'public'} |
                                {row["action_id"] for row in user_actions if row["visibility"] == "public"} |
                                {a['action_id'] for actor in snapshot.agents for a in actor.historical_actions if a.get('action_id')})
        call['prompt'] += '\nPUBLIC_CAUSE_ACTION_ID_ALLOWLIST\n' + json.dumps(allowed_causes)
        original_prompt = call['prompt']
        for attempt in range(2):
            try:
                result = await self.runtime.json_call(**call)
            except LLMError as exc:
                if attempt or not isinstance(exc.__cause__, (StructuredOutputError, ValidationError)):
                    raise
                call['call_name'] = f'{call_prefix}_game_master_settlement_format_repair'
                call['temperature'] = 0
                call['max_attempts'] = 1
                call['prompt'] = original_prompt + (
                    '\n\nFORMAT REPAIR: The previous attempts failed structured validation. '
                    'Settle the same audited actions from the same frozen state. Return one compact JSON object only, '
                    'from the opening brace to its matching closing brace. Preserve all required state, public consequences, '
                    'and registered recipients. Keep each changed issue after-state within 500 characters, '
                    'and each action_results summary within 300 characters. '
                    'For issue_updates.state_code use only a schema value; use unknown if unsupported. '
                    'Use empty optional lists for unsupported annotations. Include no second object or trailing content.'
                )
                if isinstance(exc.__cause__, ValidationError):
                    errors = exc.__cause__.errors(include_input=False, include_context=False, include_url=False)
                    optional_annotations = {"issue_updates", "action_results"}
                    if not self.sequential_outcome and not enabled(snapshot) and any(
                        error["loc"] and error["loc"][0] in optional_annotations for error in errors
                    ):
                        call["schema"] = _LegacySettlementFormatRepair
                    call["prompt"] += "\nFields to repair: " + compact_json([
                        {"path": list(error["loc"]), "problem": error["msg"]} for error in errors[:6]
                    ])
                continue
            try:
                self._validate(snapshot, result, next_round)
                from app.simulation.clock import align_events
                align_events(result.public_events, snapshot.world_state.get('simulation_clock', {}))
                from app.simulation.activities import annotate_events
                for event in result.public_events:
                    for annotation in event.activity_annotations:
                        annotation.method = 'settlement_annotation'
                annotate_events(result.public_events)
                from app.simulation.process import (suppress_continuations, validate_issue_updates,
                    redact_private_causes, retain_valid_issue_updates, VERSION as PROCESS_VERSION)
                from app.simulation.joint_settlement import (
                    intervention_actions, redact_private_executions,
                )
                from app.simulation.settlement_repair import normalize_joint_settlement
                causal_actions = [*actions, *intervention_actions(snapshot, interventions)]
                if snapshot.world_state.get('process_tracking_version') == PROCESS_VERSION:
                    if self.sequential_outcome:
                        from app.benchmark.dynamics import discard_unsupported_counts
                        from app.simulation.process import discard_explicitly_private_events
                        discard_explicitly_private_events(result)
                        discard_unsupported_counts(snapshot, result)
                    suppress_continuations(snapshot, result, actions, interventions)
                # Verify the internal causal account before removing private
                # identifiers from the public projection. Otherwise a legitimate
                # delayed public effect loses its recorded private cause.
                normalize_joint_settlement(snapshot, result, actions, interventions)
                redact_private_causes(snapshot, result, causal_actions)
                if snapshot.world_state.get('process_tracking_version') == PROCESS_VERSION:
                    try:
                        validate_issue_updates(snapshot, result, causal_actions)
                    except ValueError:
                        if not attempt:
                            raise
                        retain_valid_issue_updates(snapshot, result, causal_actions)
                else:
                    result.issue_updates = []
                    result.continuing_state_notes = []

                projected_state = project_world_state(snapshot, result, actions, program_owned)
                validate_program_projection(program_owned, projected_state)
                from app.domain.ensemble import StateInvariant
                from app.simulation.measurements import at_path, validate_invariants
                from app.simulation.resources import validate_resources
                validate_resources(snapshot.world_state, projected_state, program_owned)
                validate_invariants(snapshot.world_state, projected_state, [
                    StateInvariant.model_validate(value) for value in snapshot.world_state.get("state_invariants", [])
                ])
                for path in snapshot.world_state.get('measurement_paths') or ():
                    at_path(projected_state, path)

                public_events = {event.timeline_id: event for event in result.public_events}
                dimensions = {dimension["id"] for dimension in self.terminal_dimensions}
                result.terminal_observations = [
                    observation for observation in result.terminal_observations
                    if observation.dimension_id in dimensions and observation.event_ref in public_events
                    and observation.quote in public_events[observation.event_ref].description
                ]
                redact_private_executions(snapshot, result, causal_actions)
                recorder = getattr(self.runtime, 'recorder', None)
                if recorder and result._process_diagnostics:
                    recorder.write(call['call_name'], 'settlement_annotation_diagnostics',
                                   {'diagnostics': result._process_diagnostics})
                return result
            except ValueError as exc:
                diagnostic = {'message': str(exc),
                              'known_agent_ids': [a.agent_id for a in snapshot.agents],
                              'returned_agent_ids': [u.agent_id for u in result.agent_knowledge_updates],
                              'registered_actor_names': {a.agent_id: a.name for a in snapshot.agents},
                              'event_ids': [e.timeline_id for e in result.public_events],
                              'execution_records': [
                                  {'event_ref': e.timeline_id, 'description': e.description[:1000],
                                   'executed_decisions': [row.model_dump(mode="json") for row in e.executed_decisions]}
                                  for e in result.public_events[:8]],
                              'state_changes': [row.model_dump(mode="json") for row in getattr(result, 'state_changes', [])]}
                if repair := state_change_repair_context(snapshot, result, exc):
                    diagnostic['state_change_repair'] = repair
                recorder = getattr(self.runtime, 'recorder', None)
                if recorder:
                    recorder.write(call['call_name'], 'settlement_semantic_error', diagnostic)
                if attempt:
                    raise
                # One bounded semantic repair; accepted actions and the frozen
                # input remain identical. No other agent is asked to decide again.
                call['call_name'] = f'{call_prefix}_game_master_settlement_repair'
                call['temperature'] = 0
                call['prompt'] = original_prompt + '\n\nSEMANTIC VALIDATION ERROR (repair the result):\n' + json.dumps(diagnostic, ensure_ascii=False)
                if repair:
                    call['prompt'] += (
                        '\nRepair the named state path against state_change_repair. Use add with before=null only for '
                        'a genuinely absent leaf, never below an existing scalar or null. For replace or removal, '
                        'copy the exact frozen value. Reuse a literal span from a retained public event only if '
                        'it supports the proposed write. Omit unsupported writes while keeping valid outcomes and actions.'
                    )
                if str(exc).startswith("Execution quote"):
                    call["prompt"] += (
                        "\nUse the exact registered actor name at the start of each affected public execution clause. "
                        "Copy the execution quote from that same event description. Preserve the selected action, "
                        "its recorded owner, and the supported public consequence."
                    )
                previous = compact_json(result.model_dump(mode="json"))
                cap = getattr(self.runtime, 'context_max_chars', 128000)
                if len(call['system']) + len(call['prompt']) + len(previous) + 1000 < cap:
                    call['prompt'] += '\nPREVIOUS RESULT TO REPAIR (data):\n' + previous
                call['prompt'] += '\nReturn the complete corrected settlement; do not invent private information to fill missing entries. Use an empty information list when nothing private is learned.'
        raise RuntimeError('Settlement repair exhausted')

    @staticmethod
    def _validate(snapshot: WorldSnapshot, result: GameMasterSettlement, next_round: int) -> None:
        known_agents = {item.agent_id for item in snapshot.agents}
        event_ids = [item.timeline_id for item in result.public_events]
        if len(event_ids) != len(set(event_ids)) or any(
            not value.startswith(f"sim_{next_round}_") for value in event_ids
        ):
            raise ValueError("Game Master returned invalid public event IDs")
        if any(not set(item.actors) <= known_agents for item in result.public_events):
            raise ValueError("Game Master put an unknown actor into the frozen round")
        recipients = [item.agent_id for item in result.agent_knowledge_updates]
        if len(recipients) != len(set(recipients)) or set(recipients) != known_agents:
            missing = sorted(known_agents - set(recipients))
            unknown = sorted(set(recipients) - known_agents)
            duplicates = sorted({v for v in recipients if recipients.count(v) > 1})
            raise ValueError(f"Game Master must return exactly one knowledge update for every ordinary Agent; missing={missing}, duplicate={duplicates}, unknown={unknown}")

