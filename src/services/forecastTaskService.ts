import type { ScrapedArticle } from "@/domain/types";
import type { LLMConfig } from "./llmService";
import { apiRequest } from "./experimentService";

export interface ForecastSpec {
  kind: "question" | "market";
  question: string;
  as_of: string;
  outcome_deadline?: string;
  deadline_basis: string;
  nominal_market_end?: string;
  rules: string;
  yes_criterion: string;
  no_criterion: string;
}

export type ForecastPreview =
  | { status: "confirm"; preview_id: string; spec: ForecastSpec }
  | { status: "select_contract"; choices: { slug: string; question: string; url: string }[] };

export interface ConfirmForecast {
  preview_id: string;
  outcome_deadline?: string;
  yes_criterion?: string;
  no_criterion?: string;
  seed_wiki_urls: string[];
}

export interface ForecastCase {
  event_id: string;
  title: string;
  forecast_spec: ForecastSpec;
  scraped_articles: ScrapedArticle[];
  background: string;
}

export function previewForecast(api: string, query: string): Promise<ForecastPreview> {
  return apiRequest(api, "/events/forecast-preview", { query });
}

export function confirmForecast(api: string, input: ConfirmForecast): Promise<ForecastCase> {
  return apiRequest(api, "/events/forecast", input);
}

export interface ForecastReportData {
  status: "complete" | "preliminary" | "unavailable";
  algorithm?: string;
  aggregation_method?: string;
  planned_rounds?: number;
  probability_mass?: { yes: number; no: number; unresolved: number };
  question: string;
  as_of: string;
  outcome_deadline: string;
  criteria: { yes: string; no: string };
  contract_rules?: string;
  market_url?: string;
  warnings: string[];
  verified_outcomes: number;
  final_bases: string[];
  scheduled_milestones?: { date: string; quote: string; url: string }[];
  p_yes: number | null;
  conditional_estimate: number | null;
  bounds: [number, number] | null;
  estimated_trajectories: number;
  planned_trajectories: number;
  knowledge_version?: string;
  error?: string;
  resume_available: boolean;
  drivers: { basis: string; quote: string; reference?: string; sample_ids: number[] }[];
  sources: { title: string; url: string; revision_id?: number; revision_at?: string }[];
  trajectories: {
    sample_id: number;
    status: string;
    p_yes: number | null;
    action_path_probability: number | null;
    represented_mass?: number;
    terminal_status?: "yes" | "no" | "unresolved" | "estimated";
    planned_rounds?: number;
    recorded_rounds?: number;
    error?: string | null;
    rounds: {
      step: number;
      interval?: { interval_end_utc?: string; interval_end?: string; horizon_steps?: number } | null;
      events: { id: string; text: string }[];
      action_results?: { action_id: string; actor_name: string; status: string; summary: string }[];
      question_updates?: { condition_id: string; label: string; before: string; after: string; summary: string }[];
      sampled_choices?: {
        actor_id: string; actor_name?: string; candidate_id?: string; action?: string | null; p_selected?: number | null; waited?: boolean;
        execution_status?: "awaiting_prerequisite" | "attempt" | "ready";
        public_options?: { candidate_id: string; action: string; probability: number; selected: boolean }[];
      }[];
      contract_outlook?: { p_yes: number; basis: string; estimate_kind?: string; readout_policy?: string | null };
    }[];
  }[];
}

export function startQuestionForecast(config: LLMConfig, eventId: string, count: number, budget: {
  calls: number | null; tokens: number | null; seconds: number | null;
}, steps: number | null = null): Promise<{run_id: string}> {
  return apiRequest(config.apiBase || "", "/experiments/from-question", {
    event_id: eventId,
    sample_count: count,
    ...(steps === null ? {} : { steps }),
    api_key: config.apiKey,
    model: config.model || null,
    base_url: config.llmBaseUrl || null,
    language: "en-US",
    max_model_calls: budget.calls,
    max_total_tokens: budget.tokens,
    max_runtime_seconds: budget.seconds,
  });
}
