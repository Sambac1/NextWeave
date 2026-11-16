import { apiRequest } from "./experimentService";
import type { LLMConfig } from "./llmService";

export type BenchmarkMethod =
  | "direct_qa"
  | "qa_ensemble"
  | "one_step"
  | "single_multistep"
  | "one_step_ensemble"
  | "trajectory_sampling"
  | "trajectory_frequency"
  | "mcts_simulation";

export interface BenchmarkRoundPlan {
  mode: "adaptive" | "fixed" | "single_step";
  steps: number;
  boundaries: string[];
  milestones: string[];
}

export interface BenchmarkCase {
  case_id: string;
  question: string;
  rules: string;
  as_of: string;
  outcome_deadline: string;
  deadline_basis?: string;
  evidence: {
    url: string;
    title: string;
    revision_id: number | string;
    revision_at: string;
    revision_url?: string;
  }[];
}

export interface MarketTrajectoryProbability {
  sample_id: number;
  p_yes: number | null;
  coverage: number;
  yes_mass: number;
  no_mass: number;
  unknown_mass: number;
  status?: string;
  error?: string | null;
  action_path_probability?: number | null;
  rounds?: { step: number; interval?: { interval_start?: string; interval_end?: string; interval_end_utc?: string; sequential_harness?: string } | null;
    deliberation_count?: number; pending_trigger_rejected?: boolean;
    sampled_choices?: { actor_id: string; actor_name: string; candidate_id: string; p_selected: number; action?: string | null; waited?: boolean;
      public_options?: { candidate_id: string; action: string; probability: number; weight: number; selected: boolean }[];
      public_rejections?: { candidate_id: string; action: string; code: string }[] }[];
    action_results?: { action_id: string; actor_name: string; status: string; summary: string }[];
    interactions?: { kind: string; action_ids: string[]; outcome: string; effect: string }[];
    question_updates?: { condition_id: string; label: string; before: string; after: string; summary: string; pending_actor_ids: string[] }[];
    contract_outlook?: { p_yes: number; basis: string; evidence_ref?: string; evidence_quote?: string;
      estimate_kind?: "terminal_estimate" | "terminal_carried_forward" | "verified_simulation" | null };
    outlook_issue?: string;
    state_changes?: { path: string[]; reason: string }[];
    events: { id: string; text: string; date: string }[] }[];
  branches: {
    branch_id: number;
    kind: "enumerated" | "tail_draw";
    mass: number;
    joint_action_probability: number;
    answer: "yes" | "no" | "unknown";
    reason?: string;
    conditional_p_yes?: number | null;
    yes_path?: string;
    no_path?: string;
    forecast_error?: string;
    error?: string;
    evidence?: { ref: string; text: string }[];
    actions?: { candidate_id: string; actor: string; text: string | null; probability: number }[];
  }[];
}

export interface BenchmarkResult {
  method: BenchmarkMethod;
  p_yes: number | null;
  fallback_p_yes?: number | null;
  coverage?: number | null;
  settled_replay_coverage?: number | null;
  unresolved_replay_mass?: number | null;
  failed_or_interrupted_mass?: number | null;
  unrun_mass?: number | null;
  simulation_route?: "environment" | "actor" | "hybrid" | "semantic_replay" | "semantic_tree_search" | "review";
  assumption?: string;
  calls?: number | null;
  tokens?: number | null;
  elapsed_seconds?: number | null;
  planned_samples?: number;
  yes_samples?: number;
  no_samples?: number;
  unknown_samples?: number;
  invalid_trace_samples?: number;
  failed_samples?: number;
  interrupted_samples?: number;
  unfinished_samples?: number;
  covered_p_yes?: number | null;
  unknown_bounds?: [number, number];
  provenance?: string;
  probability_method?: "conditional_terminal_v1" | "conditional_terminal_v2" | "conditional_terminal_v3" | "mcts_puct_frontier_v1" | "poll_bootstrap_national_hamilton_v1" | "sequential_sampled_outlook_v1" | "sequential_sampled_outlook_v2";
  members?: { index: number; p_yes: number | null; status: string; actual_http_requests: number; prompt_tokens: number; completion_tokens: number }[];
  unexplored_mass?: number;
  search_iterations?: number;
  tree?: { path: number[]; depth: number; mass: number; visits: number; search_value: number | null;
    conditional_p_yes: number | null; contribution: number | null; frontier: boolean }[];
  sampling_standard_error?: number | null;
  trajectories?: MarketTrajectoryProbability[];
  status?: string;
  error_code?: string;
  resume_available?: boolean;
  readiness_counts?: ActorReadinessCounts;
  decision_coverage?: { kind: "decision_subject" | "observation"; label: string; ready: boolean }[];
  error?: string | null;
}

type ActorReadinessCounts = Partial<Record<"ready" | "restricted" | "needs_evidence" | "retryable_error", number>>;

export interface BenchmarkRun {
  run_id: string;
  case_id: string;
  forecast_key?: string;
  status: string;
  stage?: string;
  methods?: BenchmarkMethod[];
  progress?: { stage?: string; sample_count?: number; completed_samples?: number; failed_samples?: number;
    completed_rounds?: number; model_calls?: number };
  as_of?: string;
  model?: string;
  outcome_deadline?: string;
  deadline_basis?: string;
  nominal_market_end?: string | null;
  source_run_id?: string;
  settings?: Record<string, string | number | boolean | null>;
  round_plans?: Partial<Record<BenchmarkMethod, BenchmarkRoundPlan>>;
  case_snapshot?: BenchmarkCase;
  resolution_status?: "awaiting_resolution";
  market_reference?: { probability: number; label: string; observed_on: string; url?: string };
  rescore_usage?: { calls: number; tokens: number };
  confirmed_outcome_deadline?: string | null;
  resume_available?: boolean;
  results?: BenchmarkResult[];
  error?: string | null;
  preparation?: { calls?: number; tokens?: number; status?: string; error?: string | null; readiness_counts?: ActorReadinessCounts;
    actor_coverage?: { status: "ready" | "limited" | "unverified" | "observation_only"; missing_required: string[];
      unresolved_requirements: string[]; missing_essential?: string[]; gaps: string[];
      seats: { seat_id: string; name: string; role: string; required: boolean;
        decision_ids: string[]; readiness: string }[] } };
}

export function importBenchmarkCase(
  base: string,
  options: {
    market_url: string;
    as_of?: string;
    seed_wiki_urls: string[];
    archived_market?: object;
  },
): Promise<BenchmarkCase> {
  return apiRequest(base, "/benchmark/cases", options);
}

export function startBenchmarkRun(
  config: LLMConfig,
  options: {
    case_id: string;
    methods: BenchmarkMethod[];
    steps?: number | null;
    terminal_head_branches: number;
    market_probability?: number;
    market_observed_at?: string;
    sample_count: number;
    qa_samples: number;
    max_model_calls: number;
    max_total_tokens: number;
    max_preparation_calls: number;
    max_preparation_tokens: number;
    confirmed_outcome_deadline: string;
    evidence_mode: "expanded" | "seed_only";
    round_retrieval: boolean;
    bdi_updates: boolean;
    action_selection: "weighted" | "greedy";
  },
): Promise<Pick<BenchmarkRun, "run_id" | "status">> {
  return apiRequest(config.apiBase || "", "/benchmark/runs", {
    ...options,
    api_key: config.apiKey || null,
    model: config.model || null,
    base_url: config.llmBaseUrl || null,
    language: "en-US",
  });
}

export type RecentBenchmarkRun = Pick<BenchmarkRun, "run_id" | "status" | "stage" | "model">;

export function getRecentBenchmarkRuns(base: string): Promise<RecentBenchmarkRun[]> {
  return apiRequest(base, "/benchmark/runs");
}

export function getBenchmarkRun(base: string, id: string): Promise<BenchmarkRun> {
  return apiRequest(base, `/benchmark/runs/${encodeURIComponent(id)}`);
}

export function getBenchmarkCase(base: string, id: string): Promise<BenchmarkCase> {
  return apiRequest(base, `/benchmark/cases/${encodeURIComponent(id)}`);
}

export interface BenchmarkResumeBudget {
  max_model_calls?: number;
  max_total_tokens?: number;
  max_preparation_calls?: number;
  max_preparation_tokens?: number;
}

export function resumeBenchmarkRun(
  config: LLMConfig, id: string, budget: BenchmarkResumeBudget,
): Promise<BenchmarkRun> {
  return apiRequest(config.apiBase || "", `/benchmark/runs/${encodeURIComponent(id)}/resume`, {
    ...budget,
    api_key: config.apiKey || null,
  });
}
