/**
 * Frontend LLM Service
 * Thin wrappers around backend task-specific LLM endpoints.
 * All prompts, JSON parsing, and OpenRouter calls live in the backend.
 */

export interface LLMConfig {
  apiKey: string;
  model: string;
  apiBase?: string;
  llmBaseUrl?: string;
  language?: string;
}

export interface ExtractionStatus {
  event_id: string;
  run_id: string;
  status: "idle" | "running" | "completed" | "partial" | "failed";
  resume_available?: boolean;
  budget?: { actual_http_requests: number; accounted_tokens_upper: number; limits: { max_calls: number; max_total_tokens: number } };
  stage: string;
  progress: number;
  message: string;
  completed_items: number;
  total_items: number;
  started_at?: string | null;
  updated_at?: string | null;
  error?: string;
}

function resolveApiBase(config: LLMConfig): string {
  if (!config.apiBase) throw new Error("Connect the backend in Settings first.");
  return config.apiBase;
}

async function post(config: LLMConfig, path: string, body: Record<string, unknown>) {
  const apiBase = resolveApiBase(config);
  const url = `${apiBase}/llm${path}`;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_key: config.apiKey,
      model: config.model,
      base_url: config.llmBaseUrl || null,
      language: config.language || "en-US",
      ...body,
    }),
  });

  if (!res.ok) {
    let msg = `LLM error: ${res.status}`;
    try {
      const err = await res.json();
      const detail = err.detail;
      if (typeof detail === "string") msg = detail;
      else if (detail?.message) msg = detail.message;
      else if (detail) msg = JSON.stringify(detail);
    } catch { /* ignore */ }
    throw new Error(msg);
  }

  return res.json();
}


export async function generateCombinedSummary(
  config: LLMConfig,
  {
    eventTitle,
    eventId,
    articleSummaries,
    additionalContext,
  }: { eventTitle: string; eventId?: string; articleSummaries: { title: string; summary: string }[]; additionalContext?: string },
): Promise<string> {
  const data = await post(config, "/combined-summary", {
    event_title: eventTitle,
    event_id: eventId || null,
    article_summaries: articleSummaries,
    additional_context: additionalContext || null,
  });
  return data.text;
}

export type FiguresResult = {
  figures: Array<{
    agent_id: string;
    name: string;
    aliases: string[];
    agent_type: string;
    decision_subject: string;
    role: string;
    title: string;
    affiliation: string;
    description: string;
    beliefs: any;
    desires: any[];
    capabilities: any[];
    constraints: string[];
    decision_policy: any;
    relevant_timeline_refs: string[];
    historical_actions: any[];
  }>;
  timeline: Array<{
    timestamp: string;
    description: string;
    timeline_id: string;
    actors?: string[];
    actions?: any[];
  }>;
  name_mapping: Record<string, string>;
};

export type FiguresPipelineStart = Partial<FiguresResult> & {
  accepted: boolean;
  reused: boolean;
  already_running: boolean;
  status: ExtractionStatus;
};

/**
 * Start extraction, or reuse saved figures for this event.
 * Heavy work runs in the background; poll status and GET /result.
 */
export async function startFiguresPipeline(
  config: LLMConfig,
  {
    eventId,
    eventTitle,
    timeWindow = "all",
    recentYears,
    maxKeyFigures = 8,
    force = false,
    resumeProfiles = false,
    maxModelCalls = null,
    maxTotalTokens = null,
    deadlineSeconds = null,
  }: {
    eventId?: string | null;
    eventTitle: string;
    timeWindow?: string;
    recentYears?: number;
    maxKeyFigures?: number;
    force?: boolean;
    resumeProfiles?: boolean;
    maxModelCalls?: number | null;
    maxTotalTokens?: number | null;
    deadlineSeconds?: number | null;
  },
): Promise<FiguresPipelineStart> {
  return post(config, "/figures-pipeline", {
    event_id: eventId || null,
    event_title: eventTitle,
    time_window: timeWindow,
    recent_years: recentYears || null,
    max_key_figures: maxKeyFigures,
    force,
    resume_profiles: resumeProfiles,
    max_model_calls: maxModelCalls,
    max_total_tokens: maxTotalTokens,
    extraction_deadline_seconds: deadlineSeconds,
  });
}

export async function getExtractionStatus(
  config: LLMConfig,
  eventId: string,
): Promise<ExtractionStatus> {
  const apiBase = resolveApiBase(config);
  const response = await fetch(
    `${apiBase}/llm/figures-pipeline/${encodeURIComponent(eventId)}/status`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw new Error(`Extraction status error: ${response.status}`);
  }
  return response.json();
}

export async function getExtractionTimeline(
  config: LLMConfig,
  eventId: string,
): Promise<any[]> {
  const response = await fetch(
    `${resolveApiBase(config)}/llm/figures-pipeline/${encodeURIComponent(eventId)}/timeline`,
    { cache: "no-store" },
  );
  if (!response.ok) throw new Error(`Timeline is not available yet: ${response.status}`);
  const data = await response.json();
  return Array.isArray(data?.timeline) ? data.timeline : [];
}

export async function getExtractionResult(
  config: LLMConfig,
  eventId: string,
): Promise<FiguresResult | null> {
  const apiBase = resolveApiBase(config);
  const response = await fetch(
    `${apiBase}/llm/figures-pipeline/${encodeURIComponent(eventId)}/result`,
    { cache: "no-store" },
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`Extraction result error: ${response.status}`);
  }
  return response.json();
}
