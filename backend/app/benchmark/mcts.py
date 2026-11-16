"""MCTS adapter using the normal round orchestrator and frozen market evidence."""
from __future__ import annotations

from app.application.forecasting import ForecastRoundOrchestrator
from app.domain import ActionProposal, WorldSnapshot
from app.llm.client import LLMClient
from app.simulation.clock import align_events, explicit_step_clock
from app.simulation.ensemble import ExperimentRuntime, attach_evidence, validate_step
from app.domain.protocol import ScenarioProtocol
from app.simulation.measurements import validate_invariants
from app.simulation.protocol import prepare_step
from app.simulation.evidence import DecisionEvidenceRetriever
from app.storage.compact_trace import CompactTrace
from app.storage.round_stages import RoundStages

from .branch_probability import forecast_branches, terminal_branches
from .tree_search import search_probability_tree


async def run_mcts(sim_id, request, store, iterations, fallback):
    from .runner import usage

    store.acquire(sim_id)
    try:
        config = store.load(sim_id, "config")
        baseline = WorldSnapshot.model_validate(store.load(sim_id, "initial_snapshot"))
        spec = request.benchmark
        seed = request.decision_seed
        client = LLMClient(request, CompactTrace(store.directory(sim_id), request.api_key or ""))
        client.max_attempts = 2
        client.strict_json = True
        runtime = ExperimentRuntime(client, request, store, sim_id)
        protocol = ScenarioProtocol.model_validate(request.scenario_protocol)
        retriever = DecisionEvidenceRetriever(baseline, config["knowledge"], version=config["knowledge_version"],
                                               cutoff_date=config.get("simulation_start", ""))

        async def expand(path, current):
            step = len(path) + 1
            path_id = "-".join(map(str, path)) or "root"
            prefix = f"{sim_id}_mcts_{path_id}"
            working = prepare_step(current, protocol, step)
            validate_invariants(current.world_state, working.world_state, request.state_invariants)
            attach_evidence(working, baseline, config["knowledge"], request.scenario_description, 4, retriever=retriever)
            clock = explicit_step_clock(spec.round_boundaries, step, as_of=spec.as_of, outcome_deadline=spec.outcome_deadline)
            working.world_state["simulation_clock"] = clock
            directory = store.directory(sim_id) / "mcts" / path_id
            cache = RoundStages(directory, {"snapshot": working.model_dump(), "contract": spec.model_dump(), "seed": seed})

            async def generate():
                orchestrator = ForecastRoundOrchestrator(
                    runtime, update_relationships=request.update_relationships,
                    checkpoint_dir=directory / "decisions", decision_seed=seed,
                    bdi_updates=spec.bdi_updates, action_selection=spec.action_selection, compact_benchmark=True,
                )
                transitions = []

                async def expand_decisions(pools, resolve):
                    choices = [branch for branch in terminal_branches(pools, seed=seed, sample_id=step,
                               head_count=spec.terminal_head_branches) if branch["kind"] == "enumerated"]
                    estimates = await forecast_branches(choices, working, spec, runtime, prefix, request.event_title, cache)
                    for branch in choices:
                        index = branch["branch_id"]
                        child_snapshot = None
                        if step < request.steps:
                            async def transition():
                                after, frame = await resolve([
                                    ActionProposal.model_validate(row["action"]) for row in branch["choices"]
                                ], str(index))
                                align_events(frame.timeline, clock)
                                align_events([event for event in after.timeline if event.round_id == after.round_id], clock)
                                validate_step(working, after, frame, 1)
                                validate_invariants(working.world_state, after.world_state, request.state_invariants)
                                return {"snapshot": after.model_dump(mode="json"), "frame": frame.model_dump(mode="json")}

                            saved = await cache.run("transition", branch, transition, lambda value: value)
                            child_snapshot = saved["snapshot"]
                        transitions.append([branch["joint_action_probability"], estimates[index].p_yes, child_snapshot])
                    remainder = 1 - sum(row[0] for row in transitions)
                    if remainder > 1e-12:
                        transitions.append([remainder, None, None])
                    return None

                await orchestrator.run(working, request.selected_actions if step == 1 else [], prefix,
                                       terminal_resolver=expand_decisions)
                return transitions

            rows = await cache.run("expansion", {}, generate, lambda value: value)
            return [(mass, probability, WorldSnapshot.model_validate(child) if child else None)
                    for mass, probability, child in rows]

        result = await search_probability_tree(baseline, expand, depth=request.steps, iterations=iterations, fallback=fallback)
        result.update(probability_method="mcts_puct_frontier_v1", simulation_run_id=sim_id,
                      status="failed" if result["p_yes"] is None else "partial" if result["error"] else "completed",
                      **usage(runtime.budget.report()))
        store.save(sim_id, "result", result)
        return result
    finally:
        store.release(sim_id)
