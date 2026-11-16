"""Persistent market benchmark runner using the standard simulation engine."""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import time
from typing import Literal
import uuid

from pydantic import BaseModel, Field, model_validator

from app.benchmark.branch_probability import summarize_branches
from app.benchmark.contract_mechanisms import CompiledPredicate
from app.benchmark.dynamics import contract_state
from app.domain.common import LLMOptions
from app.domain.ensemble import BenchmarkSimulation, EnsembleRequest, OutcomeDefinition
from app.domain.knowledge import FiguresRequest
from app.knowledge.brief import forecast_brief_preview
from app.knowledge.extraction import extract_figures, figures_result_from_artifacts
from app.knowledge.okf import digest
from app.knowledge.forecast_evidence import _frozen_pages, _passages, frozen_sources
from app.knowledge.public_evidence import packet_manifest, public_sources
from app.llm.budget import BudgetExceeded, ModelBudget, budget_scope
from app.llm.client import LLMClient, LLMError
from app.settings import get_settings
from app.simulation.ensemble import prepare_experiment, run_experiment
from app.simulation.outlook import HARNESS_VERSION, OUTLOOK_VERSION, SEQUENTIAL_PROTOCOL
from app.simulation.registry import NoEligibleSimulationActors, simulation_readiness
from app.simulation.question_sampling import _recorded_draws, public_replay, summarize_sequential
from app.simulation.question_schedule import MAX_ADAPTIVE_ROUNDS, round_boundaries, simulation_round_plan
from app.storage.ensemble_store import EnsembleStore
from app.storage.extraction_status import ExtractionStatusRepository
from app.storage.json_store import BENCHMARK_EVENT_ORIGIN, EventRepository

logger = logging.getLogger(__name__)
Method = Literal["direct_qa", "qa_ensemble", "one_step", "single_multistep",
                 "one_step_ensemble", "trajectory_sampling", "trajectory_frequency", "mcts_simulation"]
QA_METHODS = {"direct_qa", "qa_ensemble"}
SEQUENTIAL_METHODS = {"one_step", "one_step_ensemble", "single_multistep", "trajectory_sampling"}
DEFAULT_TRAJECTORIES = 8


def sampling_allowance(trajectories: int, steps: int | None) -> tuple[int, int]:
    """Size default ceilings for the requested work, within the API's budget bounds."""
    rounds = trajectories * (steps or MAX_ADAPTIVE_ROUNDS)
    return min(5000, 20 + 12 * rounds), min(10_000_000, 250_000 + 140_000 * rounds)


class BenchmarkRequest(LLMOptions):
    case_id: str
    confirmed_outcome_deadline: datetime | None = None
    methods: list[Method] = Field(min_length=1, max_length=8)
    steps: int | None = Field(default=None, ge=1, le=12)
    sample_count: int = Field(default=DEFAULT_TRAJECTORIES, ge=2, le=64)
    qa_samples: int = Field(default=8, ge=2, le=64)
    single_step_samples: int | None = Field(default=None, ge=2, le=64)
    mcts_iterations: int = Field(default=20, ge=1, le=100)
    max_model_calls: int = Field(default=80, ge=1, le=5000)
    max_total_tokens: int = Field(default=250000, ge=1000, le=10000000)
    max_preparation_calls: int = Field(default=80, ge=1, le=5000)
    max_preparation_tokens: int = Field(default=250000, ge=1000, le=10000000)
    max_runtime_seconds: int = Field(default=900, ge=30, le=7200)
    fallback_p_yes: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    shrinkage: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=2026, ge=0, le=(1 << 53) - 1, strict=True)
    evidence_mode: Literal["expanded", "seed_only"] = "expanded"
    round_retrieval: bool = True
    bdi_updates: bool = True
    action_selection: Literal["weighted", "greedy"] = "weighted"
    terminal_branching: bool = True
    terminal_probability: Literal["replay", "forecast"] = "forecast"
    terminal_head_branches: int = Field(default=2, ge=1, le=2)
    market_probability: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    market_observed_at: datetime | None = None

    @model_validator(mode="after")
    def unique_methods(self):
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Benchmark methods must be unique")
        if "trajectory_sampling" in self.methods:
            calls, tokens = sampling_allowance(self.sample_count, self.steps)
            if "max_model_calls" not in self.model_fields_set:
                self.max_model_calls = max(self.max_model_calls, calls)
            if "max_total_tokens" not in self.model_fields_set:
                self.max_total_tokens = max(self.max_total_tokens, tokens)
        return self


class DirectForecast(BaseModel):
    p_yes: float = Field(ge=0, le=1, allow_inf_nan=False)


def stores():
    root = get_settings().data_dir / "benchmark"
    return EnsembleStore(root / "cases"), EnsembleStore(root / "runs")


def timestamp(value: str | datetime) -> datetime:
    value = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.utcoffset() is None:
        raise ValueError("Timestamps must include a time zone")
    return value.astimezone(timezone.utc)


def forecast_inputs(case: dict, sources: list[dict]) -> dict:
    sources = public_sources(sources)
    return {
        "question": case["question"], "rules": case["rules"], "as_of": case["as_of"],
        "outcome_deadline": case["outcome_deadline"], "resolution_source": case.get("resolution_source", ""),
        **{key: case[key] for key in ("outcome_window_start", "outcome_window_basis") if key in case},
        "sources": sources,
        "evidence_packet": packet_manifest(sources),
    }


def usage(report: dict | None) -> dict:
    report = report or {}
    return {"calls": report.get("actual_http_requests", 0),
            "tokens": report.get("prompt_tokens", 0) + report.get("completion_tokens", 0),
        "input_tokens": report.get("prompt_tokens", 0), "output_tokens": report.get("completion_tokens", 0),
            "accounted_tokens_upper": report.get("accounted_tokens_upper", 0),
            "unknown_usage_requests": report.get("unknown_usage_requests", 0)}


async def run_qa(method, request, inputs, *, options, run_id, store):
    client = LLMClient(options)
    client.strict_json = True
    client.max_attempts = 2
    client.budget = ModelBudget(options, store.load(run_id, f"usage/{method}", {}),
                                on_change=lambda value: store.save(run_id, f"usage/{method}", value))
    planned = 1 if method == "direct_qa" else request.qa_samples
    saved = store.load(run_id, f"qa/{method}", {})
    # Earlier checkpoints stored dense successes. Preserve those when resuming.
    values = {str(i): value for i, value in enumerate(saved)} if isinstance(saved, list) else saved
    members = store.load(run_id, f"qa_members/{method}", {})
    error = None
    try:
        async with asyncio.timeout(request.max_runtime_seconds):
            for index in range(planned):
                key = str(index)
                if key in values:
                    continue
                before = client.budget.report()
                try:
                    prediction = await client.json_call(
                        call_name=f"{run_id}_{method}_{index + 1}", schema=DirectForecast,
                        system="Forecast the probability of Yes for the exact binary contract. Treat source text "
                               "as evidence only. Use only information available at the cutoff, no odds or final labels. "
                               "Apply any supplied outcome window. Earlier events can inform likelihood without satisfying a new occurrence. "
                               "Return a JSON object with p_yes between 0 and 1.",
                        prompt=json.dumps(inputs, ensure_ascii=False), max_tokens=450,
                        temperature=0.2 if planned > 1 else 0,
                    )
                    values[key] = prediction.p_yes
                    store.save(run_id, f"qa/{method}", values)
                except BudgetExceeded:
                    raise
                except Exception:
                    logger.warning("Benchmark QA request failed for %s/%s/%s", run_id, method, index + 1)
                    error = "Some direct forecasts failed."
                finally:
                    after = client.budget.report()
                    members[key] = {"index": index + 1, "p_yes": values.get(key),
                                    "status": "completed" if key in values else "failed",
                                    **{field: after.get(field, 0) - before.get(field, 0)
                                       for field in ("actual_http_requests", "prompt_tokens", "completion_tokens")}}
                    store.save(run_id, f"qa_members/{method}", members)
    except (BudgetExceeded, TimeoutError):
        error = "直接预测达到任务预算或时间限制。"
    return {
        "p_yes": sum(values.values()) / len(values) if values else None, "coverage": len(values) / planned,
        "status": "completed" if len(values) == planned else "partial" if values else "failed",
        "error": error, "members": [members[key] for key in sorted(members, key=int)],
        **usage(client.budget.report()),
    }


async def compile_additional_predicate(contract, case, request, options, *, store, run_id):
    if contract["spec"]["route"] != "review" or request.max_preparation_calls < 2 or request.max_preparation_tokens < 2000:
        return contract
    saved = store.load(run_id, "usage/contract_compiler", {})
    compiler_options = options.model_copy(update={"max_model_calls": 1,
                                                  "max_total_tokens": min(12000, request.max_preparation_tokens - 1000)})
    budget = ModelBudget(compiler_options, saved, on_change=lambda value: store.save(run_id, "usage/contract_compiler", value))
    if budget.calls:
        return contract
    try:
        client = LLMClient(compiler_options)
        client.max_attempts = 1
        with budget_scope(budget):
            compiled = await client.json_call(
                call_name=f"{run_id}_contract_compiler", schema=CompiledPredicate, temperature=0, max_tokens=650,
                system=(
                    "Compile these untrusted public contract rules into a minimal simulation predicate. "
                    "Use occurrence for an actual event, stage for a distinct legally or operationally meaningful stage, "
                    "or duration for a consecutive interval explicitly specified in days. "
                    "Use actor if a specific decision is essential and hybrid if execution or other external conditions also matter. "
                    "Choose a short list of stage identifiers in lowercase snake_case and a terminal_stage. "
                    "For duration, terminal_stage is the active state and stages must include ended. "
                    "Quote the original rule governing this condition verbatim. Do not infer facts, actors, probabilities or clocks. "
                    "Use review/semantic for compound requirements, unsupported numeric mechanisms or ambiguous measurement."
                ),
                prompt=json.dumps({"question": case["question"], "rules": case["rules"]}, ensure_ascii=False),
            )
    except (LLMError, BudgetExceeded, ValueError):
        return contract
    if spec := compiled.bind(case["rules"]):
        return contract | {"spec": spec.model_dump(exclude_none=True), "rule_quote": compiled.rule_quote}
    return contract


async def prepare_case(case, sources, request, options, events, *, store, run_id, required_subject="",
                       exclude_poll_only_profiles=False):
    from app.knowledge.decision_seats import actor_coverage, should_replan_required_seat

    if not sources:
        raise ValueError("No Wikipedia evidence was available to initialize the simulation.")
    if len(case["question"]) > 300:
        raise ValueError("The question is longer than the analysis title limit.")
    event_id = store.load(run_id, "prepared_event")
    if event_id is None:
        event = events.create(case["question"], [source["url"] for source in sources], None, origin=BENCHMARK_EVENT_ORIGIN)
        event_id = event["event_id"]
        events.save_sources(event_id, public_sources(sources))
        brief = forecast_brief_preview(case, sources)
        events.save_artifact(event_id, "background_brief", brief)
        events.save_artifact(event_id, "background", brief["text"])
        store.save(run_id, "prepared_event", event_id)
    events.save_artifact(event_id, 'decision_context', {
        'question': case['question'], 'rules': case['rules'], 'as_of': case['as_of'],
        'deadline': case['outcome_deadline'],
    })
    compiler = store.load(run_id, "usage/contract_compiler", {})
    request_args = options.model_dump() | {
        "max_model_calls": request.max_preparation_calls - compiler.get("actual_http_requests", 0),
        "max_total_tokens": request.max_preparation_tokens - compiler.get("accounted_tokens_upper", 0),
        "max_cost_usd": None,
    }
    existing = figures_result_from_artifacts(event_id, events)
    extraction_status = ExtractionStatusRepository(events).get(event_id).get("status")
    previous_budget = events.load_artifact(event_id, "extraction_budget", {})
    has_timeline = bool(events.load_artifact(event_id, "timeline_cache", {}).get("timeline"))
    profile_inputs = events.load_artifact(event_id, "profile_inputs", {})
    reviewed_subject = profile_inputs.get("quality", {}).get("required_subject_review", {}).get("subject")
    saved_seat_plan = events.load_artifact(event_id, 'decision_seat_plan', {})
    retry_seat_plan = should_replan_required_seat(saved_seat_plan, required_subject)
    supplement = bool(required_subject and existing and has_timeline and
                      (retry_seat_plan or (profile_inputs and reviewed_subject != required_subject)))
    if not existing and not has_timeline and previous_budget.get("actual_http_requests", 0):
        return event_id, {"status": "failed", "error": "前次准备在时间线保存前中断，无法按原账本续跑准备。",
                          **usage(previous_budget)}
    extraction_request = FiguresRequest(
        **request_args, event_id=event_id, event_title=case["question"],
        history_before=timestamp(case["as_of"]).date(), planning_mode="rules",
        required_decision_subject=required_subject,
        exclude_poll_only_profiles=exclude_poll_only_profiles,
        extraction_deadline_seconds=request.max_runtime_seconds,
        resume_profiles=has_timeline and (supplement or extraction_status in {"partial", "failed", "running"}),
    )
    try:
        if not existing or supplement:
            await extract_figures(extraction_request, events)
    except Exception as exc:
        logger.exception("Benchmark case preparation failed for %s", event_id)
        state = ExtractionStatusRepository(events).get(event_id)
        spent = usage(events.load_artifact(event_id, "extraction_budget", {}))
        reason = state.get("error") or (str(exc) if isinstance(exc, (BudgetExceeded, ValueError)) else "")
        return event_id, {
            "status": "partial" if state.get("resume_available") else "failed",
            "error": reason[:300] or "时间线或主体档案初始化失败。",
            "stage": state.get("stage"), "resume_available": bool(state.get("resume_available")), **spent,
        }
    readiness = Counter(simulation_readiness(actor, require_grounded=True, on_date=extraction_request.history_before.isoformat())
                        for actor in events.load_artifact(event_id, "agents", []))
    seat_plan = events.load_artifact(event_id, 'decision_seat_plan', {})
    coverage = actor_coverage(seat_plan,
                              events.load_artifact(event_id, 'agents', []), case['as_of'], rules=case['rules'])
    events.save_artifact(event_id, 'actor_coverage', coverage)
    if coverage['status'] == 'observation_only':
        return event_id, {'status':'unsupported', 'actor_coverage':coverage, 'readiness_counts':dict(readiness),
                         'error':'No independent actor decisions are established. This question needs a supported observation or environment mechanism.',
                         **usage(events.load_artifact(event_id,'extraction_budget',{}))}
    if missing := coverage.get('missing_essential'):
        return event_id, {'status': 'unsupported', 'actor_coverage': coverage, 'readiness_counts': dict(readiness),
                          'error': f'Essential decision lacks a verified actor: {missing[0]}',
                          **usage(events.load_artifact(event_id, 'extraction_budget', {}))}
    return event_id, {"status": "completed", "readiness_counts": dict(readiness), 'actor_coverage': coverage,
                      'actor_plan_retryable': should_replan_required_seat(seat_plan, required_subject),
                      **usage(events.load_artifact(event_id, "extraction_budget", {}))}


async def run_simulation(method, request, case, event_id, *, options, events, store, run_id, contract, round_plan):
    sequential = method in SEQUENTIAL_METHODS
    steps, dates = round_plan["steps"], round_plan["boundaries"]
    count = 1 if method in {"one_step", "single_multistep", "mcts_simulation"} else request.sample_count
    if method == "one_step_ensemble" and request.single_step_samples is not None:
        count = request.single_step_samples
    start = timestamp(case["as_of"])
    end = timestamp(case["outcome_deadline"])
    rules = case["rules"]
    if source := case.get("resolution_source"):
        rules += f"\nSpecified resolution source: {source}"
    simulation = EnsembleRequest(
        **options.model_dump(), event_id=event_id, event_title=case["question"],
        scenario_description=f"Contract question: {case['question']}\nEvidence cutoff: {case['as_of']}\n"
                             f"Simulation horizon: {case['outcome_deadline']}\n"
                             f"Horizon basis: {case.get('deadline_basis', 'unspecified')}\n"
                             "Apply the actual outcome timing and criteria in the contract rules.",
        start_date=dates[0], steps=steps, sample_count=count, step_unit="day", planning_mode="rules",
        max_runtime_seconds=request.max_runtime_seconds,
        decision_policy="weighted", decision_seed=request.seed,
        initial_world_state={"benchmark_contract": contract},
        outcome_definitions=[
            OutcomeDefinition(id="yes", label="Yes", criterion="The frozen contract's Yes rule is satisfied."),
            OutcomeDefinition(id="no", label="No", criterion="The frozen contract's No rule is satisfied."),
        ],
        benchmark=BenchmarkSimulation(
            round_boundaries=dates, resolution_rules=rules, outcome_deadline=end.isoformat(),
            as_of=start.isoformat(), prompt_profile="compact",
            round_retrieval=request.round_retrieval, bdi_updates=request.bdi_updates,
            action_selection="weighted" if sequential else request.action_selection,
            sequential_outcome=sequential,
            terminal_branching=not sequential and request.terminal_branching and method != "trajectory_frequency",
            terminal_probability=request.terminal_probability,
            terminal_head_branches=request.terminal_head_branches,
        ),
    )
    sim_store = EnsembleStore(store.directory(run_id) / "simulations")
    previous_id = store.load(run_id, f"simulation_ids/{method}")
    if previous_id:
        simulation = simulation.model_copy(update={"resume_run_id": previous_id})
    sim_id, resolved = await asyncio.to_thread(prepare_experiment, simulation, events=events, store=sim_store)
    store.save(run_id, f"simulation_ids/{method}", sim_id)
    if method == "mcts_simulation":
        from app.benchmark.mcts import run_mcts
        return await run_mcts(sim_id, resolved, sim_store, request.mcts_iterations, request.fallback_p_yes)
    try:
        result = await run_experiment(sim_id, resolved, store=sim_store)
    except Exception:
        logger.exception("Benchmark simulation failed for %s/%s", run_id, method)
        result = {}
    categories = {row["id"]: row["count"] for row in result.get("categories", [])}
    yes, no, unknown = (categories.get(key, 0) for key in ("yes", "no", "other"))
    samples = result.get("samples", [])
    failed = sum(sample.get("status") == "failed" for sample in samples)
    unfinished = count - yes - no - unknown - failed
    errors = list(dict.fromkeys(sample.get("error") for sample in samples if sample.get("error")))
    if not errors:
        stopped = sim_store.load(sim_id, "status", {}).get("error")
        errors = [stopped] if stopped else []
    if resolved.benchmark.sequential_outcome:
        summary = summarize_sequential([sim_store.sample(sim_id, index) for index in range(1, count + 1)],
                                       expected_steps=steps)
        return summary | {"planned_samples": count, "simulation_run_id": sim_id,
                          "decision_coverage": sim_store.load(sim_id, "config", {}).get("decision_coverage", []),
                          "error": errors[0] if errors else summary["error"],
                          **usage(result.get("budget") or sim_store.load(sim_id, "budget", {}))}
    if resolved.benchmark.terminal_branching:
        if not samples:
            samples = [sim_store.sample(sim_id, index) | {"sample_id": index} for index in range(1, count + 1)]
        programmatic = bool(contract.get("projection"))
        summary = summarize_branches(samples, count, request.fallback_p_yes,
                                     mode="replay" if programmatic else resolved.benchmark.terminal_probability,
                                     settled_context=True)
        if programmatic:
            summary["probability_method"] = contract["projection"]["model"]
            summary["assumption"] = contract["projection"]["assumption"]
        for trajectory in summary["trajectories"]:
            trajectory["rounds"] = public_replay(sim_store.sample(sim_id, trajectory["sample_id"]))
        return summary | {
            "planned_samples": count,
            "simulation_run_id": sim_id,
            "error": errors[0] if errors else None,
            **usage(result.get("budget") or sim_store.load(sim_id, "budget", {})),
        }
    probability = None
    if yes + no:
        estimate = (yes + (unknown + failed + unfinished) * request.fallback_p_yes) / count
        probability = (1 - request.shrinkage) * estimate + request.shrinkage * request.fallback_p_yes
    return {"p_yes": probability, "coverage": (yes + no) / count,
            "raw_yes_frequency": yes / (yes + no) if yes + no else None,
            "status": "completed" if yes + no == count else "partial" if yes + no else "failed",
            "planned_samples": count, "yes_samples": yes, "no_samples": no,
            "unknown_samples": unknown, "failed_samples": failed, "unfinished_samples": unfinished,
            "simulation_run_id": sim_id,
            "unknown_bounds": [yes / count, 1 - no / count],
            "error": errors[0][:300] if errors else "部分模拟未完成或无法根据合约判定。" if yes + no < count else None,
            **usage(result.get("budget") or sim_store.load(sim_id, "budget", {}))}


async def run_benchmark(run_id, request, case, *, store, events=None):
    store.acquire(run_id)
    status = store.load(run_id, "status", {})
    try:
        status.update(status="running", stage="freezing_evidence", outcome_deadline=case["outcome_deadline"],
                      deadline_basis=case.get("deadline_basis"),
                      nominal_market_end=case.get("nominal_market_end"))
        store.save(run_id, "status", status)
        previous_input = store.load(run_id, 'input')
        if previous_input is not None:
            _frozen_pages(case, request.evidence_mode)
            sources = previous_input.get('sources', [])
        else:
            sources = frozen_sources(case, request.evidence_mode)
        inputs = forecast_inputs(case, sources)
        options = LLMOptions.model_validate(request.model_dump())
        settings = get_settings()
        options = options.model_copy(update={"model": options.model or settings.llm_model,
                                             "base_url": options.base_url or settings.llm_base_url})
        if previous_input is not None and digest(previous_input) != digest(inputs):
            raise ValueError("Frozen benchmark input changed; start a new run to use the updated evidence packet. Existing results are preserved.")
        store.save(run_id, "input", inputs)
        status["evidence_sha256"] = digest(inputs)
        events = events or EventRepository()
        event_id = None
        preparation = None
        contract = store.load(run_id, "contract_spec")
        sequential = bool(set(request.methods) & SEQUENTIAL_METHODS)
        if sequential and contract and contract.get("trajectory_protocol") != HARNESS_VERSION:
            status.update(status="failed", stage="stopped",
                          error="This checkpoint uses an earlier simulation protocol. Start a new run.")
            return
        if not contract and any(store.load(run_id, f"simulation_ids/{method}") for method in set(request.methods) - QA_METHODS):
            raise ValueError("This checkpoint belongs to an earlier simulation protocol. Start a new run.")
        for method in request.methods:
            previous = next((row for row in status["results"] if row["method"] == method), None)
            if previous and previous["status"] == "completed":
                continue
            status.update(stage=method)
            store.save(run_id, "status", status)
            started = time.perf_counter()
            try:
                if method in QA_METHODS:
                    result = await run_qa(method, request, inputs, options=options, run_id=run_id, store=store)
                else:
                    if contract is None:
                        contract = await compile_additional_predicate(contract_state(case, sources, include_projection=not sequential), case, request,
                                                                       options, store=store, run_id=run_id)
                        if sequential:
                            contract["trajectory_protocol"] = HARNESS_VERSION
                        store.save(run_id, "contract_spec", contract)
                        status["simulation_route"] = contract["spec"]["route"]
                        status["contract_compiler_usage"] = usage(store.load(run_id, "usage/contract_compiler", {}))
                    plans = status.setdefault("round_plans", {})
                    if method not in plans:
                        plans[method] = simulation_round_plan(case, method, request.steps, contract)
                        store.save(run_id, "status", status)
                    round_plan = plans[method]
                    environment = contract["spec"]["route"] == "environment"
                    count_opens = contract.get("scheduled_milestone", {}).get("counting_from")
                    count_too_late = bool(count_opens and timestamp(case["outcome_deadline"]).date().isoformat() < count_opens)
                    if environment and (not contract.get("projection") or count_too_late or method == "mcts_simulation"):
                        result = {"status": "unsupported", "p_yes": None, "coverage": 0,
                                  "error": contract["spec"]["reason"] or ("The outcome window ends before counting can start."
                                      if count_too_late else "Actor-tree search does not apply to this count model.")}
                    elif environment:
                        result = await run_simulation(method, request, case, None, options=options,
                                                      events=events, store=store, run_id=run_id, contract=contract,
                                                      round_plan=round_plan)
                    else:
                        if preparation is None:
                            required_subject = contract["spec"].get("subject", "") if contract["spec"]["route"] == "actor" else ""
                            event_id, preparation = await prepare_case(case, sources, request, options, events,
                                store=store, run_id=run_id, required_subject=required_subject,
                                exclude_poll_only_profiles=bool(contract.get("scheduled_milestone")
                                    and contract["spec"].get("predicate") in {"rank", "threshold"}))
                            status["preparation"] = preparation
                            store.save(run_id, "status", status)
                        if preparation["status"] != "completed":
                            result = {"status": "unsupported" if preparation["status"] == "unsupported" else "failed",
                                      "p_yes": None, "coverage": 0,
                                      "error": preparation["error"], "error_code": "actor_preparation_error",
                                      "resume_available": preparation.get("resume_available", False)}
                        else:
                            dependencies = preparation.get('actor_coverage', {}).get('decision_dependencies', [])
                            simulation_contract = contract | {'decision_dependencies': dependencies} if dependencies and method in SEQUENTIAL_METHODS else contract
                            result = await run_simulation(method, request, case, event_id, options=options, events=events,
                                                          store=store, run_id=run_id, contract=simulation_contract, round_plan=round_plan)
                    result["simulation_route"] = ("semantic_tree_search" if method == "mcts_simulation" and not environment
                                                  else "semantic_replay" if contract["spec"]["route"] == "review"
                                                  else contract["spec"]["route"])
            except NoEligibleSimulationActors as exc:
                retryable = exc.retryable or bool(preparation and preparation.get('actor_plan_retryable'))
                result = {"status": "failed" if retryable else "unsupported", "p_yes": None, "coverage": 0,
                          "error_code": "retryable_actor_profile" if retryable else "needs_actor_evidence",
                          "readiness_counts": exc.readiness_counts, "resume_available": retryable, "error": str(exc)[:300],
                          "simulation_route": contract["spec"]["route"]}
            except ValueError as exc:
                result = {"status": "failed", "p_yes": None, "coverage": 0, "error": str(exc)[:300]}
            except Exception:
                logger.exception("Benchmark method %s failed for %s", method, run_id)
                result = {"status": "failed", "p_yes": None, "coverage": 0, "error": "The forecasting method failed."}
            status["results"] = [row for row in status["results"] if row["method"] != method]
            status["results"].append({
                **usage(store.load(run_id, f"usage/{method}", {})), **result, "method": method,
                "forecast_created_at": datetime.now(timezone.utc).isoformat(),
                "contract_id": case.get("condition_id") or case["case_id"], "forecast_cutoff": case["as_of"],
                "model": options.model, "evidence_snapshot_id": status["evidence_sha256"],
                "method_version": OUTLOOK_VERSION if method in SEQUENTIAL_METHODS else result.get("probability_method") or (
                    SEQUENTIAL_PROTOCOL if method in SEQUENTIAL_METHODS else
                    "direct_probability_v1" if method in QA_METHODS else "hard_replay_v1"),
                "fallback_p_yes": request.fallback_p_yes, "elapsed_seconds": round((previous or {}).get("elapsed_seconds", 0) + time.perf_counter() - started, 3),
            })
            store.save(run_id, "status", status)
        status["results"].sort(key=lambda row: request.methods.index(row["method"]))
        count = sum(result["p_yes"] is not None for result in status["results"])
        status.update(status="completed" if all(result["status"] == "completed" for result in status["results"])
                      else "partial" if count or any(row["status"] == "partial" for row in status["results"])
                      else "failed", stage="finished")
    except asyncio.CancelledError:
        status.update(status="interrupted", stage="stopped")
        raise
    except Exception:
        logger.exception("Benchmark runner failed for %s", run_id)
        status.update(status="failed", stage="stopped", error="The benchmark could not freeze its inputs or start the forecast.")
    finally:
        store.save(run_id, "status", status)
        store.release(run_id)


def new_run(request, store):
    run_id = str(uuid.uuid4())
    store.save(run_id, "config", request.model_dump(exclude={"api_key"}))
    status = {"run_id": run_id, "case_id": request.case_id, "status": "queued", "results": []}
    store.save(run_id, "status", status)
    return status
