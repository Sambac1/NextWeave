"""One-round forecast use case composed from explicit functional-Agent roles."""

from __future__ import annotations

from typing import Any, Awaitable, Callable
from pathlib import Path

from app.domain import ForecastRound, SelectedAction, WorldSnapshot, RelationshipProjection, ActivationDecision, ActionProposal
from app.domain.forecast import SequentialGameMasterSettlement
from app.functional_agents.relationships import RelationshipAgent
from app.functional_agents.simulation import (
    ActionAuditMasterAgent,
    ActivationMasterAgent,
    GameMasterAgent,
    OrdinaryBDIAgent,
)
from app.llm.client import LLMClient
from app.simulation.registry import maintain_registry
from app.simulation.transitions import apply_round_update
from app.storage.run_logger import audit_record


TerminalResolver = Callable[
    [list[list[dict]], Callable[[list[ActionProposal], str], Awaitable[tuple[WorldSnapshot, ForecastRound]]]],
    Awaitable[tuple[WorldSnapshot, ForecastRound]],
]


def _public_graph(snapshot: WorldSnapshot) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "role": agent.role,
                "public_state": agent.public_state,
                "activation": agent.activation.model_dump(),
            }
            for agent in snapshot.agents
        ],
        "edges": [item.model_dump() for item in snapshot.relationships],
    }


class ForecastRoundOrchestrator:
    """Coordinate one frozen round without taking over any Agent's decisions."""

    def __init__(
        self, runtime: LLMClient, *, update_relationships: bool = True,
        checkpoint_dir: Path | None = None, decision_seed: int | None = None, sample_id: int = 0,
        terminal_dimensions: list[dict] | None = None, prior_measurement_states: dict | None = None,
        bdi_updates: bool = True, action_selection: str = "weighted", compact_benchmark: bool = False,
        sequential_outcome: bool = False, main_forecast=None,
    ) -> None:
        self.runtime = runtime
        self.checkpoint_dir = checkpoint_dir
        self.update_relationships = update_relationships
        self.decision_seed = decision_seed
        self.sample_id = sample_id
        self.bdi_updates = bdi_updates
        self.action_selection = action_selection
        self.compact_benchmark = compact_benchmark
        self.sequential_outcome = sequential_outcome
        self.main_forecast = main_forecast
        self.ordinary_agent = OrdinaryBDIAgent(runtime)
        self.activation_master = ActivationMasterAgent(runtime)
        self.action_auditor = ActionAuditMasterAgent(runtime, self.ordinary_agent)
        self.game_master = GameMasterAgent(runtime)
        self.game_master.weighted_decisions = decision_seed is not None
        self.game_master.compact_benchmark = compact_benchmark
        self.game_master.sequential_outcome = sequential_outcome
        self.game_master.main_forecast = main_forecast is not None
        self.game_master.terminal_dimensions = terminal_dimensions or []
        self.game_master.prior_measurement_states = prior_measurement_states or {}
        self.relationship_agent = RelationshipAgent(runtime)

    def _record(
        self,
        *,
        call_prefix: str,
        round_id: int,
        content_type: str,
        data: Any,
        agent_id: str = "all",
        status: str = "info",
    ) -> None:
        audit_record(
            getattr(self.runtime, "recorder", None),
            source=call_prefix,
            round_id=round_id,
            agent_id=agent_id,
            content_type=content_type,
            status=status,
            phase=call_prefix,
            data=data,
        )

    async def run(
        self,
        snapshot: WorldSnapshot,
        interventions: list[SelectedAction],
        call_prefix: str,
        *, terminal_resolver: TerminalResolver | None = None,
    ) -> tuple[WorldSnapshot, ForecastRound]:
        next_round = snapshot.round_id + 1
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="round-input-snapshot",
            data=snapshot.model_dump(),
        )
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="timeline-before-round",
            data=[item.model_dump() for item in snapshot.timeline],
        )
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="relationship-graph-before",
            data=_public_graph(snapshot),
        )
        for agent in snapshot.agents:
            self._record(
                call_prefix=call_prefix,
                round_id=next_round,
                agent_id=agent.agent_id,
                content_type="agent-profile-before-round",
                data=agent.model_dump(),
            )

        from app.storage.round_stages import RoundStages
        from app.simulation import weighted_decisions
        request = getattr(self.runtime, 'request', None)
        scope = {'snapshot': snapshot.model_dump(),
            'interventions': [i.model_dump() for i in interventions],
            'update_relationships': self.update_relationships,
            'provider': {k: getattr(request, k, None) for k in ('model','base_url','language')}}
        if not self.bdi_updates or self.action_selection != 'weighted' or self.compact_benchmark:
            scope['benchmark_controls'] = [self.bdi_updates, self.action_selection, self.compact_benchmark]
        if self.sequential_outcome:
            from app.simulation.outlook import HARNESS_VERSION
            scope['sequential_outcome'] = HARNESS_VERSION
        if self.decision_seed is not None:
            scope['decision_sampling'] = {
                'version': weighted_decisions.VERSION, 'seed': self.decision_seed, 'sample_id': self.sample_id,
            }
        if terminal_resolver is not None:
            if self.decision_seed is None:
                raise ValueError("Terminal branching requires weighted decision pools")
            scope['terminal_branching'] = 1
        cache = RoundStages(self.checkpoint_dir, scope)
        decision_cache = cache
        if self.main_forecast is not None:
            shared_scope = {**scope, 'main_question_decisions': 1,
                            'decision_sampling': {'version': weighted_decisions.VERSION, 'seed': self.decision_seed}}
            decision_cache = RoundStages(self.main_forecast.store.directory(self.main_forecast.run_id) / "shared_round_stages",
                                         shared_scope, locks=self.main_forecast.decision_locks)
        self.action_auditor.proposal_cache = decision_cache
        async def prepare_decisions():
            boundary_state = maintain_registry(snapshot)
            self._record(
                call_prefix=call_prefix,
                round_id=next_round,
                content_type="agents-after-registry",
                data=[item.model_dump() for item in boundary_state.agents],
            )
            informed_state = boundary_state
            if self.bdi_updates:
                informed_state = await decision_cache.model('information', boundary_state.model_dump(),
                    lambda: self.ordinary_agent.process_information(boundary_state, call_prefix), WorldSnapshot)
            self._record(
                call_prefix=call_prefix,
                round_id=next_round,
                content_type="agents-after-information-processing",
                data=[item.model_dump() for item in informed_state.agents],
            )
            async def activate():
                state, active, decisions = await self.activation_master.select(informed_state, interventions, call_prefix)
                return {'state': state.model_dump(), 'active_ids': [a.agent_id for a in active],
                        'decisions': [d.model_dump() for d in decisions]}
            def decode_activation(value):
                state = WorldSnapshot.model_validate(value['state'])
                return state, [a for a in state.agents if a.agent_id in value['active_ids']], [ActivationDecision.model_validate(d) for d in value['decisions']]
            decision_state, active_agents, activation_decisions = await decision_cache.run('activation', informed_state.model_dump(), activate, decode_activation)
            for decision in activation_decisions:
                self._record(
                    call_prefix=call_prefix,
                    round_id=next_round,
                    agent_id=decision.agent_id,
                    content_type="master-self-activation-result",
                    data=decision.model_dump(),
                    status="selected" if decision.selected else "inactive",
                )
            if self.decision_seed is None:
                actions = await self.action_auditor.produce(decision_state, active_agents, interventions, call_prefix)
                sampling = {}
            else:
                actions, sampling = await weighted_decisions.produce_weighted_actions(
                    decision_state, active_agents, interventions, self.runtime, call_prefix,
                    seed=self.decision_seed, sample_id=self.sample_id, cache=decision_cache, selection=self.action_selection,
                    **({"compact_benchmark": True} if self.compact_benchmark else {}),
                    **({"export_pools": True} if terminal_resolver is not None else {}),
                    **({"record_draws": True} if self.sequential_outcome else {}),
                    **({"main_forecast": self.main_forecast} if self.main_forecast else {}),
                )
                if self.sequential_outcome and not active_agents:
                    sampling["audited_silent_round"] = 1
            for action in actions:
                self._record(
                    call_prefix=call_prefix,
                    round_id=next_round,
                    agent_id=action.agent_id,
                    content_type="audited-agent-prediction",
                    data=action.model_dump(),
                    status="success",
                )

            return {'decision_state': decision_state.model_dump(), 'active_ids': [a.agent_id for a in active_agents],
                    'activation_decisions': [d.model_dump() for d in activation_decisions],
                    'actions': [a.model_dump() for a in actions], 'sampling': sampling}
        def decode_decisions(value):
            state = WorldSnapshot.model_validate(value['decision_state'])
            active = [a for a in state.agents if a.agent_id in value['active_ids']]
            decisions = [ActivationDecision.model_validate(d) for d in value['activation_decisions']]
            actions = [ActionProposal.model_validate(a) for a in value['actions']]
            return state, active, decisions, actions, value.get('sampling', {})
        decision_state, active_agents, activation_decisions, actions, sampling = await cache.run(
            'audited_decisions', {}, prepare_decisions, decode_decisions,
        )

        if terminal_resolver is not None:
            sampling = dict(sampling)
            pools = sampling.pop("execution_pools", [])

            async def resolve(branch_actions, branch_id):
                return await self._settle(
                    snapshot, decision_state.model_copy(deep=True), active_agents, activation_decisions,
                    branch_actions, sampling, [item.model_copy(deep=True) for item in interventions],
                    f"{call_prefix}_branch_{branch_id}",
                )

            return await terminal_resolver(pools, resolve)

        return await self._settle(
            snapshot, decision_state, active_agents, activation_decisions, actions,
            sampling, interventions, call_prefix, cache=decision_cache if self.sequential_outcome else None,
        )

    async def _settle(
        self, snapshot, decision_state, active_agents, activation_decisions, actions,
        sampling, interventions, call_prefix, *, cache=None,
    ) -> tuple[WorldSnapshot, ForecastRound]:
        next_round = snapshot.round_id + 1

        async def settle_actions():
            return await self.game_master.settle(decision_state, actions, interventions, call_prefix)

        if cache is not None:
            async def save_settlement():
                result = await settle_actions()
                return {"settlement": result.model_dump(mode="json"), "diagnostics": result._process_diagnostics}

            def read_settlement(value):
                result = SequentialGameMasterSettlement.model_validate(value["settlement"])
                result._process_diagnostics = value["diagnostics"]
                return result

            settlement = await cache.run("validated_settlement", {
                "state": decision_state.model_dump(mode="json"),
                "actions": [action.model_dump(mode="json") for action in actions],
                "dimensions": self.game_master.terminal_dimensions,
                "prior_measurements": self.game_master.prior_measurement_states,
            }, save_settlement, read_settlement)
        else:
            settlement = await settle_actions()
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="game-master-settlement-result",
            data=settlement.model_dump(mode="json"),
            status="success",
        )

        # Profiles are initialised once and stay fixed: the layered design has no per-
        # round profile rewriting.
        relationship_projection = RelationshipProjection()
        if self.update_relationships:
            async def update_relationships():
                return await self.relationship_agent.update(decision_state, settlement, call_prefix)

            if cache is None:
                relationship_projection = await update_relationships()
            else:
                relationship_projection = await cache.model(
                    "relationship_projection", settlement.model_dump(mode="json"), update_relationships, RelationshipProjection,
                )
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="relationship-projection",
            data=relationship_projection.model_dump(),
            status="success",
        )

        updated = apply_round_update(
            decision_state,
            actions,
            settlement,
            relationship_projection,
        )
        from app.simulation.joint_settlement import intervention_actions
        recorded_interventions = intervention_actions(snapshot, interventions)
        actors_by_id = {actor.agent_id: actor for actor in updated.agents}
        for item in recorded_interventions:
            actor = actors_by_id.get(item.agent_id)
            if actor is not None:
                history = actor.historical_actions if item.visibility == "public" else actor.private_action_history
                history.append(item.model_dump(exclude={"reasoning", "plan_update"}))
                del history[:-(40 if item.visibility == "public" else 30)]
        if not self.bdi_updates:
            prior_agents = {agent.agent_id: agent for agent in snapshot.agents}
            for agent in updated.agents:
                if prior := prior_agents.get(agent.agent_id):
                    agent.beliefs = prior.beliefs.model_copy(deep=True)
                    agent.desires = [desire.model_copy(deep=True) for desire in prior.desires]
                    agent.current_plan = prior.current_plan.model_copy(deep=True) if prior.current_plan else None
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="timeline-generated-this-round",
            data=[item.model_dump() for item in settlement.public_events],
            status="success",
        )
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="timeline-after-round",
            data=[item.model_dump() for item in updated.timeline],
            status="success",
        )
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="relationship-graph-after",
            data=_public_graph(updated),
            status="success",
        )
        for agent in updated.agents:
            self._record(
                call_prefix=call_prefix,
                round_id=next_round,
                agent_id=agent.agent_id,
                content_type="agent-profile-after-round",
                data=agent.model_dump(),
                status="success",
            )
        self._record(
            call_prefix=call_prefix,
            round_id=next_round,
            content_type="round-output-snapshot",
            data=updated.model_dump(),
            status="success",
        )
        return updated, ForecastRound(
            step=updated.round_id,
            input_snapshot_version=snapshot.snapshot_version,
            output_snapshot_version=updated.snapshot_version,
            active_agent_ids=[item.agent_id for item in active_agents],
            activation_decisions=activation_decisions,
            agent_actions=actions,
            decision_sampling={key: value for key, value in sampling.items() if key != "sampled_choices"},
            sampled_choices=sampling.get("sampled_choices", []),
            action_results=settlement.action_results,
            intervention_actions=recorded_interventions,
            interactions=settlement.interactions,
            question_updates=settlement.question_updates,
            state_changes=settlement.state_changes if isinstance(settlement, SequentialGameMasterSettlement) else [],
            state_summary=settlement.state_summary,
            issue_updates=settlement.issue_updates,
            continuing_state_notes=settlement.continuing_state_notes,
            analysis=settlement.analysis,
            outcomes=settlement.outcomes,
            timeline=settlement.public_events,
            terminal_observations=settlement.terminal_observations,
            state_observations=settlement.state_observations,
            new_relationships=relationship_projection.relationship_updates,
            agent_updates=[],
            next_watchpoints=settlement.next_watchpoints,
            # Handed to the consensus step: it is the only place that sees all
            # samples at once, and a verdict can only be applied there once the
            # samples agree on it.
            power_impacts=list(settlement.power_impacts),
        )
