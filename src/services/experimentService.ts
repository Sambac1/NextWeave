import type { ForecastReportData } from "./forecastTaskService";
import type { LLMConfig } from "./llmService";

export async function apiRequest(
  base: string,
  path: string,
  body?: unknown,
  method?: "GET" | "POST" | "DELETE",
) {
  if (!base) throw new Error("Connect the local backend first.");
  const response = await fetch(`${base}${path}`, {
    method: method || (body === undefined ? "GET" : "POST"),
    ...(body === undefined
      ? {}
      : {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
  if (!response.ok) {
    let detail: any;
    try {
      detail = (await response.json()).detail;
    } catch {
      /* use HTTP status */
    }
    throw new Error(
      typeof detail === "string"
        ? detail
        : JSON.stringify(detail || `HTTP ${response.status}`),
    );
  }
  return response.json();
}

export function startExperiment(
  config: LLMConfig,
  options: Record<string, unknown>,
) {
  return apiRequest(config.apiBase || "", "/experiments", {
    ...options,
    api_key: config.apiKey,
    model: config.model || null,
    base_url: config.llmBaseUrl || null,
    language: config.language || "en-US",
  });
}

export interface Frequency {
  id: string;
  label: string;
  criterion: string;
  count: number;
  frequency: number | null;
  sampling_interval_95: [number, number] | null;
  sample_ids: number[];
  unclear_count?: number;
  planned_sample_bounds?: [number, number];
}

export interface EventFrequency extends Frequency {
  topic?: string;
  milestone?: string;
  trace_metrics?: {
    basis: string; matched_count: number; not_matched_count: number; unclear_count: number;
    reviewed_count: number; unfinished_count: number; planned_count: number; match_frequency: number | null;
  };
  no_count: number;
  unfinished_count: number;
  planned_samples: number;
  assessable_samples: number;
  known_sample_frequency: number | null;
  interpretation_bounds: [number, number] | null;
  measurement_status: "measured" | "partial" | "unassessable";
  answers_by_sample: {
    sample_id: number;
    answer: "yes" | "no" | "unclear" | "unfinished";
    reason: string;
    trace_answer?: "matched" | "not_matched" | "unclear" | null;
    trace_reason?: string;
    event_refs: string[];
    evidence: { timeline_id: string; description: string }[];
  }[];
}

export interface ObservedActivitySummary {
  version:string; basis:string; planned_samples:number; completed_samples:number; unfinished_samples:number;
  public_events:number; unlabelled_events:number;
  groups:{id:string; family:string; family_label:string; stage:string; label:string; count:number;
    completed_samples:number; frequency:number|null; sample_ids:number[]; evidence_count:number;
    evidence:{sample_id:number; step:number; timeline_id:string; quote:string; method:string; actors:string[]}[];
  }[];
}

export type DecisionPolicy = "single" | "weighted";

export interface DecisionSamplingSummary {
  decisions: number;
  multi_candidate_decisions: number;
  single_candidate_decisions: number;
  concentrated_decisions: number;
  rejected_candidates: number;
  duplicate_candidates: number;
  supplemented_decisions: number;
  not_selected_for_deliberation: number;
  public_issue_divergence_pairs?: number;
  public_issue_reconvergence_pairs?: number;
}

export interface TerminalDimension {
  id: string;
  label: string;
  observation_mode?: "legacy" | "standing" | "boundary" | "mechanism" | "unavailable";
  observation_scope?: string;
  assessable_samples: number;
  unclear_count: number;
  unfinished_count: number;
  unknown_sample_ids: number[];
  categories: Frequency[];
}

export interface TerminalJudgment {
  dimension_id: string;
  category_id: string;
  event_refs: string[];
  evidence_quote: string;
  reason: string;
}

export interface Reassessment {
  created_at: string;
  result: EnsembleResult;
}

export interface EnsembleResult {
  run_id: string;
  sample_count: number;
  completed_samples: number;
  failed_samples: number;
  uncompleted_samples: number;
  horizon: number;
  step_unit: string;
  step_size?: number;
  decision_policy?: DecisionPolicy;
  decision_sampling?: DecisionSamplingSummary;
  categories: Frequency[];
  outcome_dimensions?: TerminalDimension[];
  measurement_preflight?: { passed: boolean; pilot_samples: number; attempted_samples: number; missing_dimensions: string[] };
  event_frequencies: EventFrequency[];
  observed_activities?: ObservedActivitySummary;
  process_summary?: ProcessSummary;
  legacy_measurement_view?: boolean;
  partition_validation?: string;
  measurement_warnings?: string[];
  simulation_start?: string;
  knowledge_version: string | null;
  baseline_sha256: string;
  samples: {
    sample_id: number;
    status: string;
    rounds_completed: number;
    classification?: { terminal_judgments?: TerminalJudgment[] } | null;
    classification_warning?: string | null;
    error?: string;
  }[];
  reliability: {
    complete_coverage: boolean;
    classification_fallbacks: number;
    real_world_validation: string;
    degraded_samples?: number;
    bdi_updates_rejected?: number;
    goal_updates_rejected?: number;
    goal_ids_repaired?: number;
    issue_updates_rejected?: number;
    deliberation_rounds?: number;
    rounds_without_deliberation?: number;
    external_actions?: number;
    explicit_waits?: number;
  };
}

export interface IssueChange {
  issue_id: string; title: string; before: string; after: string; kind: string;
  state_code?: string; actor_ids: string[]; event_refs: string[]; evidence_quote: string;
  pending_actor_ids: string[];
}
export interface ReplayRound {
  step: number;
  simulation_interval?: { interval_start?: string; interval_end?: string } | null;
  state_summary?: string | null;
  timeline?: { timeline_id: string; description: string; actors?: string[] }[] | null;
  issue_updates?: IssueChange[] | null;
  continuing_state_notes?: string[] | null;
  measurement_states?: Record<string, {
    category_id: string; observed_on: string; quote?: string; event_ref?: string | null;
    expires_on?: string | null; carry?: boolean;
    source: "public_event" | "settled_public_state" | "public_issue_update" | "configured_mechanism" | "conflicting_observations";
  }> | null;
  public_action_count?: number;
  public_wait_count?: number;
  agent_actions?: {
    agent_id: string; agent_name?: string; decision?: string;
    action_type?: string; action?: string; target?: string;
  }[];
}
export interface SampleReplay {
  sample_id: number;
  status: string;
  rounds: ReplayRound[];
  error?: string | null;
  classification?: { category_id?: string } | null;
  validation_issues?: { code?: string; message?: string }[];
}
export interface ProcessSummary {
  version: string; completed_samples: number; planned_samples: number; mode: string;
  comparison_basis: string; path_count: number; topics: {id:string;title:string;count:number}[];
  paths: {id:string; representative_sample_id:number; sample_ids:number[]; count:number; frequency:number;
    terminal_category_id:string; steps: {step:number; interval?:{interval_start?:string;interval_end?:string};
      changes:IssueChange[]; events:{timeline_id:string;description:string}[]; public_event_count:number;
      action_count:number; wait_count:number; structured:boolean; continuing_count:number}[]}[];
}

export interface BudgetSummary {
  actual_http_requests: number;
  accounted_tokens_upper: number;
  accounted_cost_upper_usd: number;
  unknown_usage_requests: number;
  currency_estimate_available: boolean;
  limits: {
    max_total_tokens: number | null;
    max_calls: number | null;
    max_cost_usd: number | null;
  };
  pricing: {
    input: number | null;
    output: number | null;
    cached: number | null;
  };
}
export interface RuntimePlan {
  settings: {
    step_unit: string;
    retrieval_k: number;
    context_max_chars: number;
    reason: string;
  };
  research_basis: {
    sources: { title: string; url: string; version_date: string }[];
  };
}

export interface ExperimentStatus {
  event_id?: string;
  question_forecast?: boolean;
  report?: ForecastReportData | null;
  budget?: BudgetSummary;
  runtime_plan?: RuntimePlan;
  resume_supported?: boolean;
  run_id: string;
  status: string;
  stage: string;
  sample_count: number;
  steps: number;
  completed_samples: number;
  completed_rounds: number;
  failed_samples: number;
  decision_policy?: DecisionPolicy;
  model_calls?: number;
  elapsed_seconds?: number;
  error?: string;
  result?: EnsembleResult;
  reassessment?: Reassessment | null;
  plan?: {
    categories: { id: string; label: string; criterion: string }[];
    outcome_dimensions?: { id: string; label: string; categories: { id: string; label: string; criterion: string }[] }[];
  };
}
