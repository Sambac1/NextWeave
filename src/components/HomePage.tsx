"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import axios from "axios";
import toast from "react-hot-toast";
import EventInput from "@/components/EventInput";
import AgentProfilePanel from "@/components/AgentProfilePanel";
import BackgroundBriefPanel from "@/components/BackgroundBriefPanel";
import QuestionForecast from "@/features/forecast/QuestionForecast";
import { previewForecast, confirmForecast, type ConfirmForecast, type ForecastSpec } from "@/services/forecastTaskService";
import EnsemblePanel from "@/features/forecast/EnsemblePanel";
import MarketBenchmark from "@/features/benchmark/MarketBenchmark";
import KnowledgePanel from "@/features/KnowledgePanel";
import SavedCases from "@/features/SavedCases";
import { apiRequest } from "@/services/experimentService";
import BackendConnection from "@/features/settings/BackendConnection";
import { readBackendUrl, BACKEND_STORAGE_KEY } from "@/config/backend";
import { marketEvaluationPath } from "@/config/evaluationRoutes";
import ProviderSettings from "@/features/settings/ProviderSettings";
import { LLM_PROVIDERS, detectProviderId } from "@/config/llmProviders";
import type {
  Agent,
  GraphData,
  ScrapedArticle,
} from "@/domain/types";
import {
  LLMConfig,
  getExtractionStatus,
  getExtractionTimeline,
  getExtractionResult,
  generateCombinedSummary,
  startFiguresPipeline,
  type ExtractionStatus,
  type FiguresResult,
} from "@/services/llmService";
import {
  Loader2,
  FlaskConical,
  ExternalLink,
  BookOpen,
  Settings,
  Zap,
  Clock,
  X,
} from "lucide-react";


function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function mapFiguresToAgents(figureData: FiguresResult["figures"]): Agent[] {
  return figureData.map((a: any, i: number) => ({
    agent_id: a.agent_id != null ? String(a.agent_id) : String(a.figure ?? a.name ?? `agent_${i}`),
    name: String(a.figure ?? a.name ?? a.source_figure ?? a.canonical_name ?? ""),
    aliases: Array.isArray(a.aliases) ? a.aliases.map(String) : [],
    agent_type: String(a.agent_type ?? "person"),
    decision_subject: String(a.decision_subject ?? a.figure ?? a.name ?? ""),
    role: a.role ? String(a.role) : "",
    title: a.title ? String(a.title) : undefined,
    affiliation: a.affiliation ? String(a.affiliation) : undefined,
    description: String(a.local_view ?? a.description ?? a.summary ?? ""),
    relationships: Array.isArray(a.relationships) ? a.relationships : [],
    historical_actions: Array.isArray(a.history ?? a.historical_actions) ? (a.history ?? a.historical_actions) : [],
    beliefs: a.beliefs,
    desires: a.desires,
    capabilities: a.capabilities,
    constraints: a.constraints,
    decision_policy: a.decision_policy,
    public_state: a.public_state,
    decision_core: a.decision_core,
    forecast_role: a.forecast_role,
    current_plan: a.current_plan,
    behavioral_model: a.behavioral_model,
    relevant_timeline_refs: a.relevant_timeline_refs,
    activation: a.activation,
  }));
}

function formatExtractionStatus(status: ExtractionStatus) {
  const itemProgress = status.total_items > 0
    ? ` (${status.completed_items}/${status.total_items})`
    : "";
  return `${status.message || "Extracting..."}${itemProgress} · ${status.progress}%`;
}

export default function Home({ showBenchmark, onReturnToBriefing, initialCaseId }: {
  showBenchmark: boolean;
  initialCaseId?: string;
  onReturnToBriefing: () => void;
}) {
  const [API_BASE, setBackendUrl] = useState("");
  const API_DOCS_URL = API_BASE ? API_BASE.replace(/\/api\/?$/, "") + "/docs" : null;
  const saveBackendUrl = (url: string) => {
    if (url) localStorage.setItem(BACKEND_STORAGE_KEY, url);
    else localStorage.removeItem(BACKEND_STORAGE_KEY);
    setBackendUrl(url);
  };
  const [step, setStep] = useState<"input" | "analysis">("input");
  const [eventId, setEventId] = useState<string | null>(null);
  const [caseError, setCaseError] = useState("");
  const [eventTitle, setEventTitle] = useState("");
  const [forecastSpec, setForecastSpec] = useState<ForecastSpec | null>(null);
  const [scrapedArticles, setScrapedArticles] = useState<ScrapedArticle[]>([]);
  const [generatedBackground, setGeneratedBackground] = useState("");
  const backgroundCase = useRef(eventId);
  backgroundCase.current = eventId;
  const [agents, setAgents] = useState<Agent[]>([]);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [profileAgent, setProfileAgent] = useState<Agent | null>(null);
  const [extractionStatus, setExtractionStatus] = useState<ExtractionStatus | null>(null);
  const [budgetCalls, setBudgetCalls] = useState("");
  const [budgetTokens, setBudgetTokens] = useState("");
  const [budgetMinutes, setBudgetMinutes] = useState("");
  const [overlays, setOverlays] = useState<Set<"timeline" | "forecast">>(new Set());
  const [historyFocus, setHistoryFocus] = useState<{actorId?:string;eventId?:string}>({});
  const [wikiTimeline, setWikiTimeline] = useState<{ timestamp: string; description: string; figures?: any[]; actions?: any[]; origin?: string; round_id?: number | null; retrieval_actors?: { name?: string }[] }[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");

  const [apiKey, setApiKey] = useState("");
  const [modelName, setModelName] = useState("google/gemini-2.5-flash");
  const [llmBaseUrl, setLlmBaseUrl] = useState("");
  const [providerId, setProviderId] = useState<string>("default");
  const [showApiKey, setShowApiKey] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  useEffect(() => {
    setBackendUrl(readBackendUrl());
    setBudgetCalls(localStorage.getItem("task_budget_calls") || "");
    setBudgetTokens(localStorage.getItem("task_budget_tokens") || "");
    setBudgetMinutes(localStorage.getItem("task_budget_minutes") || "");
    const savedKey = sessionStorage.getItem("llm_api_key") || "";
    localStorage.removeItem("llm_api_key");
    localStorage.removeItem("openrouter_api_key");
    const savedModel = localStorage.getItem("llm_model") || localStorage.getItem("openrouter_model") || "google/gemini-2.5-flash";
    const savedBaseUrl = localStorage.getItem("llm_base_url") || "";
    const savedProvider = localStorage.getItem("llm_provider_id") || detectProviderId(savedBaseUrl);
    if (savedKey) setApiKey(savedKey);
    if (savedModel) setModelName(savedModel);
    if (savedBaseUrl) setLlmBaseUrl(savedBaseUrl);
    if (savedProvider) setProviderId(savedProvider);
  }, []);

  const getLLMConfig = useCallback((): LLMConfig => {
    if (providerId === "default") {
      return { apiKey: "", model: "", apiBase: API_BASE, language: "en-US" };
    }
    return { apiKey, model: modelName, apiBase: API_BASE, llmBaseUrl: llmBaseUrl || undefined, language: "en-US" };
  }, [apiKey, modelName, llmBaseUrl, providerId, API_BASE]);

  const saveApiKey = useCallback((key: string) => {
    setApiKey(key);
    sessionStorage.setItem("llm_api_key", key);
  }, []);
  const saveModelName = useCallback((model: string) => {
    setModelName(model);
    localStorage.setItem("llm_model", model);
  }, []);
  const saveBaseUrl = useCallback((url: string) => {
    setLlmBaseUrl(url);
    localStorage.setItem("llm_base_url", url);
  }, []);
  const saveProviderId = useCallback((id: string) => {
    setProviderId(id);
    localStorage.setItem("llm_provider_id", id);
    const preset = LLM_PROVIDERS.find((p) => p.id === id);
    if (preset && preset.id !== "custom" && preset.id !== "default") {
      setLlmBaseUrl(preset.baseUrl);
      localStorage.setItem("llm_base_url", preset.baseUrl);
    } else if (preset && preset.id === "default") {
      setLlmBaseUrl("");
      localStorage.setItem("llm_base_url", "");
    }
  }, []);

  // Restore the last local analysis so long-running experiments remain reachable.
  useEffect(() => {
    if (!API_BASE || eventId) return;
    let cancelled = false;
    async function restore() {
      try {
        const session = JSON.parse(localStorage.getItem("analysis_session") || "null");
        const caseId = initialCaseId || (session?.apiBase === API_BASE ? session.id : null);
        if (!caseId) return;
        const context = await apiRequest(API_BASE, `/events/context?event_id=${encodeURIComponent(caseId)}`);
        if (cancelled) return;
        let saved: any = null;
        try { saved = await apiRequest(API_BASE, `/llm/figures-pipeline/${encodeURIComponent(caseId)}/result`); }
        catch { /* source-only cases can continue extraction */ }
        if (cancelled) return;
        const background = context.background_brief?.text ?? context.background ?? "";
        const title = context.event_title || (session?.id === caseId ? session.title : "");
        setEventId(caseId); setEventTitle(title); setGeneratedBackground(background);
        setForecastSpec(context.forecast_spec || null);
        setCaseError("");
        localStorage.setItem("analysis_session", JSON.stringify({ id: caseId, title, background, apiBase: API_BASE }));
        setGraphData(context.graph); setStep("analysis");
        if (saved) { setAgents(mapFiguresToAgents(saved.figures || [])); setWikiTimeline(saved.timeline || []); }
      } catch (failure) {
        if (!cancelled && initialCaseId) setCaseError(failure instanceof Error ? failure.message : "The saved analysis could not be opened.");
      }
    }
    restore();
    return () => { cancelled = true; };
  }, [API_BASE, eventId, initialCaseId]);

  async function handleForecastPreview(query: string) {
    if (!API_BASE) {
      setShowSettings(true);
      throw new Error("Connect the backend in Settings before reviewing a question.");
    }
    return previewForecast(API_BASE, query);
  }

  async function handleEventSubmit(input: ConfirmForecast) {
    const result = await confirmForecast(API_BASE, input);
    const background = result.background || "";
    setEventId(result.event_id);
    setEventTitle(result.title);
    setForecastSpec(result.forecast_spec);
    setScrapedArticles(result.scraped_articles);
    setGeneratedBackground(background);
    setAgents([]);
    setWikiTimeline([]);
    setGraphData({ nodes: [], edges: [] });
    setExtractionStatus(null);
    localStorage.setItem("analysis_session", JSON.stringify({
      id: result.event_id, title: result.title, background, apiBase: API_BASE,
    }));
    setStep("analysis");
    toast.success("Wikipedia evidence collected");
  }

  const applyFiguresResult = useCallback(async (
    eventIdValue: string,
    result: FiguresResult,
    options?: { silent?: boolean },
  ) => {
    const figureData = result.figures || [];
    if (!figureData.length) {
      toast.error("No actor profiles were returned.");
      return false;
    }

    if (result.timeline?.length) {
      setWikiTimeline(result.timeline);
    }

    const newAgents = mapFiguresToAgents(figureData);
    setAgents(newAgents);

    for (const agent of newAgents) {
      try {
        await axios.post(`${API_BASE}/agents/node`, {
          event_id: eventIdValue,
          node_id: agent.agent_id,
          label: agent.name,
          node_type: "agent",
          properties: {
            role: agent.role,
            title: agent.title || "",
            affiliation: agent.affiliation || "",
            description: agent.description,
            agent_type: agent.agent_type,
            decision_subject: agent.decision_subject,
            beliefs: agent.beliefs,
            desires: agent.desires,
            capabilities: agent.capabilities,
            constraints: agent.constraints,
            decision_policy: agent.decision_policy,
          },
        });
      } catch (e) {
        console.error("Failed to add agent node:", e);
      }
    }

    try {
      const graphRes = await axios.get(`${API_BASE}/agents/${eventIdValue}/graph`);
      setGraphData(graphRes.data);
    } catch (e) {
      console.error("Failed to load graph after extraction:", e);
    }
    if (!options?.silent) {
      toast.success(`Loaded ${newAgents.length} actor ${newAgents.length === 1 ? "profile" : "profiles"}`);
    }
    setExtractionStatus((prev) => (
      prev
        ? { ...prev, status: "completed", progress: 100, message: prev.message || "Extraction completed." }
        : prev
    ));
    return true;
  }, [API_BASE]);

  const pollExtractionUntilSettled = useCallback(async (config: LLMConfig, eventIdValue: string) => {
    let timelineLoaded = false;
    for (;;) {
      const status = await getExtractionStatus(config, eventIdValue);
      setExtractionStatus(status);
      if (!timelineLoaded && status.progress >= 60) {
        try {
          const timeline = await getExtractionTimeline(config, eventIdValue);
          if (timeline.length) {
            setWikiTimeline(timeline);
            timelineLoaded = true;
          }
        } catch {
          // The endpoint returns 404 until the Timeline artifact is persisted.
        }
      }
      if (status.status === "running") {
        setLoadingMessage(formatExtractionStatus(status));
        await sleep(1500);
        continue;
      }
      return status;
    }
  }, []);

  const handleGenerateAgents = useCallback(async (force = false, resumeProfiles = false) => {
    if (!eventId) return;
    setLoading(true);
    setLoadingMessage("Checking saved extraction...");
    const config = getLLMConfig();

    try {
      const started = await startFiguresPipeline(config, { eventId, eventTitle, force, resumeProfiles,
        maxModelCalls: budgetCalls ? Number(budgetCalls) : null, maxTotalTokens: budgetTokens ? Number(budgetTokens) : null, deadlineSeconds: budgetMinutes ? Number(budgetMinutes)*60 : null });
      if (started.status) {
        setExtractionStatus(started.status);
        if (started.status.status === "running") {
          setLoadingMessage(formatExtractionStatus(started.status));
        }
      }

      if ((started.reused || started.figures?.length) && started.figures?.length) {
        await applyFiguresResult(eventId, started as FiguresResult);
        return;
      }

      const status = await pollExtractionUntilSettled(config, eventId);
      if (status.status === 'partial') {
        toast.error(status.error || status.message);
        return;
      }
      if (status.status === "failed") {
        toast.error(status.error || "Extraction failed");
        return;
      }

      const saved = await getExtractionResult(config, eventId);
      if (!saved) {
        toast.error(status.message || "The timeline was saved, but no decision actors were identified. Check the sources for evidence of who can act.");
        return;
      }
      await applyFiguresResult(eventId, saved);
    } catch (err: any) {
      console.error("Failed to generate figures:", err);
      try {
        const saved = eventId ? await getExtractionResult(config, eventId) : null;
        if (saved?.figures?.length) {
          await applyFiguresResult(eventId, saved);
          return;
        }
      } catch {
        // Fall through to the original error.
      }
      toast.error(`Pipeline Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [eventId, eventTitle, getLLMConfig, applyFiguresResult, pollExtractionUntilSettled, budgetCalls, budgetTokens, budgetMinutes]);

  useEffect(() => {
    if (step !== "analysis" || !eventId || agents.length > 0) return;
    let cancelled = false;
    const config = getLLMConfig();
    void (async () => {
      try {
        const saved = await getExtractionResult(config, eventId);
        if (cancelled) return;
        if (!saved?.figures?.length) {
          const status = await getExtractionStatus(config, eventId);
          if (!cancelled) {
            setExtractionStatus(status);
            if (status.resume_available) {
              const cached = await getExtractionTimeline(config, eventId);
              if (!cancelled) setWikiTimeline(cached);
            }
          }
          return;
        }
        await applyFiguresResult(eventId, saved, { silent: true });
      } catch (error) {
        console.error("Could not restore saved figures:", error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [step, eventId, agents.length, getLLMConfig, applyFiguresResult]);

  const forecastActorReady = agents.some(agent => agent.forecast_role === "decision_actor");
  const resumeProfiles = !!extractionStatus?.resume_available;
  const profilesRunning = extractionStatus?.status === "running";
  const profileActionLabel = loading ? "Preparing actor profiles…"
    : profilesRunning ? "Check profile generation"
      : resumeProfiles ? "Resume actor profiles"
        : agents.length > 0 ? "Regenerate actor profiles" : "Generate actor profiles";
  const needsProfiles = resumeProfiles || profilesRunning || (forecastSpec ? !forecastActorReady : agents.length === 0);
  const prepareAnalysisActors = () => handleGenerateAgents(
    !!forecastSpec && agents.length > 0 && !forecastActorReady && !resumeProfiles && !profilesRunning,
    resumeProfiles,
  );

  const overlayMeta = {
    timeline: { label: "Historical Knowledge", icon: Clock },
    forecast: { label: "Global Forecast", icon: Zap },
  } as const;

  const toggleOverlay = (key: "timeline" | "forecast") => {
    setOverlays((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div className="research-workspace flex flex-col text-ink-700">
      <div className="h-[2px] shrink-0 bg-signal-blue" />

      <main className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <header className="workspace-header flex flex-wrap items-center justify-between gap-3 border-b bg-white shrink-0">
          <div className="flex min-w-0 flex-wrap items-center gap-x-2">
            <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight text-ink-900">
              <Image
                src="/brand/nextweave-icon.png"
                alt=""
                width={120}
                height={120}
                priority
                className="h-9 w-9"
              />
              NextWeave
            </h1>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {eventId && !showBenchmark && <button disabled={loading} onClick={() => {
              if (initialCaseId) onReturnToBriefing();
              localStorage.removeItem("analysis_session"); setEventId(null); setEventTitle("");
              setAgents([]); setWikiTimeline([]); setScrapedArticles([]); setGeneratedBackground(""); setForecastSpec(null);
              setOverlays(new Set()); setStep("input");
            }} className="text-xs text-ink-500 border rounded-lg px-3 py-1.5 disabled:opacity-40">New Topic</button>}
            {showBenchmark && (
              <button type="button" onClick={onReturnToBriefing}
                className="rounded-lg border border-paper-300 px-3 py-1.5 text-xs text-ink-600 hover:bg-paper-100 focus-visible:ring-2 focus-visible:ring-brand-500">
                Back to Briefing
              </button>
            )}
            <Link
              href={marketEvaluationPath}
              shallow
              aria-current={showBenchmark ? "page" : undefined}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border focus-visible:ring-2 focus-visible:ring-brand-500 ${showBenchmark ? "bg-brand-700 border-brand-700 text-white" : "border-paper-300 text-ink-600 hover:bg-paper-100"}`}
            >
              <FlaskConical aria-hidden="true" className="w-3 h-3" />
              Market Evaluation
            </Link>

            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all border ${
                showSettings
                  ? "bg-signal-blue border-signal-blue text-white"
                  : "bg-white border-paper-300 text-ink-600 hover:bg-paper-100"
              } shadow-sm hover:shadow`}
            >
              <Settings className="w-3 h-3" />
              Settings
            </button>

            {API_DOCS_URL && <a
              href={API_DOCS_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-paper-300 bg-white text-ink-600 hover:bg-paper-100 shadow-sm hover:shadow transition-all"
            >
              <BookOpen className="w-3 h-3" />
              Docs
            </a>}
          </div>
        </header>

        {!API_BASE && !showBenchmark && (
          <div className="px-6 py-3 bg-signal-blue/5 border-b border-signal-blue/15 text-sm flex flex-wrap gap-2 items-center justify-between">
            <p>Connect the backend to gather sources, build actor profiles, and run forecasts.</p>
            <button onClick={() => setShowSettings(true)} className="text-signal-blue font-semibold underline underline-offset-4">Connect Backend</button>
          </div>
        )}
        {showSettings && <div className="shrink-0 max-h-[65vh] overflow-y-auto">
        <BackendConnection value={API_BASE} onSave={saveBackendUrl} disabled={loading || !!eventId} />
        {!showBenchmark && <div className="border-b border-paper-200 bg-white px-6 py-4 text-xs space-y-2">
          <h3 className="font-semibold text-ink-800">Task Budget</h3>
          <p className="text-ink-400">Applies to new extraction and simulation tasks. Leave blank for no limit. Resumed runs use cumulative usage.</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {([
              ["Total model request limit",budgetCalls,setBudgetCalls,"task_budget_calls",1,50000],
              ["Total token limit",budgetTokens,setBudgetTokens,"task_budget_tokens",1000,100000000],
              ["Runtime limit per session (minutes)",budgetMinutes,setBudgetMinutes,"task_budget_minutes",1,120],
            ] as const).map(([label,value,setter,key,min,max])=><label key={key}>{label}
              <input aria-label={label} type="number" min={min} max={max} placeholder="Unlimited" value={value} disabled={loading}
                onChange={e=>{const next=e.target.value; setter(next); if(next) localStorage.setItem(key,next); else localStorage.removeItem(key);}}
                onBlur={e=>{const raw=e.target.value; if(raw){const next=String(Math.min(max,Math.max(min,Math.trunc(Number(raw)))));setter(next);localStorage.setItem(key,next);}}}
                className="mt-1 block w-full rounded-lg border border-paper-300 px-3 py-2" />
            </label>)}
          </div>
        </div>}
        <ProviderSettings
          open={showSettings}
          providerId={providerId}
          baseUrl={llmBaseUrl}
          apiKey={apiKey}
          modelName={modelName}
          showApiKey={showApiKey}
          onProviderChange={saveProviderId}
          onBaseUrlChange={saveBaseUrl}
          onApiKeyChange={saveApiKey}
          onModelNameChange={saveModelName}
          onShowApiKeyChange={setShowApiKey}
          onClose={() => setShowSettings(false)}
        />
        </div>}
        <div id="market-benchmark" className={`workspace-content flex-1 min-h-0 overflow-y-auto ${showBenchmark ? "" : "hidden"}`}>
          <MarketBenchmark
            key={API_BASE}
            config={getLLMConfig()}
            onOpenSettings={() => setShowSettings(true)}
          />
        </div>
        <div className={`workspace-content flex-1 min-h-0 overflow-y-auto flex-col ${showBenchmark ? "hidden" : "flex"}`} data-stage={step}>
          {step === "input" ? (
            <div className="flex-1 flex items-center justify-center px-4 py-10 sm:px-8 sm:py-14">
              <div className="workspace-input mx-auto">
                {caseError && <p role="alert" className="mb-3 rounded-lg border border-signal-red/30 bg-white p-3 text-sm text-signal-red">{caseError}</p>}
                <EventInput onPreview={handleForecastPreview} onConfirm={handleEventSubmit} loading={loading} />
                <SavedCases apiBase={API_BASE} onOpen={async (item) => {
                  try {
                    const context = await apiRequest(API_BASE, `/events/context?event_id=${encodeURIComponent(item.id)}&event_title=${encodeURIComponent(item.title)}`);
                    let saved: any = null;
                    try { saved = await getExtractionResult(getLLMConfig(), item.id); } catch { /* sources only */ }
                    setEventId(item.id); setEventTitle(item.title); setGeneratedBackground(context.background_brief?.text || context.background || "");
                    setForecastSpec(context.forecast_spec || null);
                    setExtractionStatus(null); setCaseError("");
                    setGraphData(context.graph); setAgents(saved ? mapFiguresToAgents(saved.figures || []) : []);
                    setWikiTimeline(saved?.timeline || []); setStep("analysis");
                    localStorage.setItem("analysis_session", JSON.stringify({ id: item.id, title: item.title, background: context.background_brief?.text || context.background || "", apiBase: API_BASE }));
                  } catch (e: any) { toast.error(e.message); }
                }} />
              </div>
            </div>
          ) : (
            <div className="workspace-reader">
              {(loading || extractionStatus?.status === "running") && (
                <div className="mb-5 px-4 py-3 rounded-xl border border-signal-blue/20 bg-signal-blue/5">
                  <div className="flex items-center gap-2 text-sm text-signal-blue">
                    <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                    <span>{loadingMessage || extractionStatus?.message || "Preparing actor profiles…"}</span>
                  </div>
                  <div className="mt-2 h-1.5 rounded-full bg-paper-200 overflow-hidden">
                    <div
                      className="h-full bg-signal-blue transition-all"
                      style={{ width: `${Math.max(4, extractionStatus?.progress || 0)}%` }}
                    />
                  </div>
                </div>
              )}
              {extractionStatus?.status === 'partial' && <div role="alert" className="mb-4 border border-amber-300 bg-amber-50 rounded-xl p-4 text-sm space-y-2">
                <p className="font-semibold">Timeline saved. Actor profiles are still incomplete.</p>
                <p>{extractionStatus.error || extractionStatus.message}</p>
                {extractionStatus.budget && <p>Used {extractionStatus.budget.actual_http_requests} model requests and approximately {extractionStatus.budget.accounted_tokens_upper.toLocaleString("en-US")} tokens. Usage carries over when resumed.</p>}
              </div>}
              {forecastSpec && <h1 className="mb-5 max-w-2xl border-b border-paper-300 pb-4 text-2xl font-semibold leading-tight tracking-tight text-ink-900 sm:text-3xl">{eventTitle}</h1>}
              <div id="background-brief" className="workspace-reader-title flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <BookOpen className="h-5 w-5 text-signal-blue" aria-hidden="true" />
                  <h2 className="text-lg font-semibold text-ink-800">Background Brief</h2>
                </div>
                {eventId && needsProfiles && <button type="button" disabled={loading || !API_BASE}
                  className="min-h-11 rounded-lg bg-brand-700 px-3 py-2 text-sm font-semibold text-white hover:bg-brand-800 disabled:opacity-50"
                  onClick={() => void prepareAnalysisActors()}>{profileActionLabel}</button>}
              </div>
              <BackgroundBriefPanel text={generatedBackground} apiBase={API_BASE} caseId={eventId} disabled={loading} forecast={!!forecastSpec}
                onRegenerate={async()=>{
                  const current=eventId;
                  if(!current) return;
                  const text=await generateCombinedSummary(getLLMConfig(), {eventTitle,eventId:current,articleSummaries:[]});
                  if(backgroundCase.current !== current) return;
                  setGeneratedBackground(text);
                  localStorage.setItem("analysis_session",JSON.stringify({id:current,title:eventTitle,background:text,apiBase:API_BASE}));
                }}/>

              {scrapedArticles.length > 0 && (
                <div className="mt-6 pt-4 border-t border-paper-200">
                  <h4 className="text-xs font-medium text-ink-400 mb-3">Sources</h4>
                  {scrapedArticles.map((article, i) => (
                    <div key={i} className="mb-1.5">
                      {article.error ? (
                        <span className="text-xs text-signal-red font-mono">ERR: {article.url}</span>
                      ) : (
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-ink-600 hover:text-signal-blue flex items-center gap-1.5 truncate transition-colors"
                        >
                          <ExternalLink className="w-3 h-3 shrink-0" />
                          <span className="truncate">{article.title || article.domain}</span>
                        </a>
                      )}
                    </div>
                  ))}
                </div>
              )}
              {eventId && forecastSpec && <footer className="mt-8 flex justify-end border-t border-paper-200 pt-5">
                <button type="button" disabled={!API_BASE}
                  className={`min-h-11 rounded-lg px-4 py-2 text-sm font-semibold disabled:opacity-50 ${needsProfiles ? "border border-paper-300 bg-white text-brand-800 hover:bg-paper-100" : "bg-brand-700 text-white hover:bg-brand-800"}`}
                  onClick={() => setOverlays(new Set(["forecast"] as const))}>Configure &amp; run forecast</button>
              </footer>}
            </div>
          )}
        </div>
      </main>
      {!showBenchmark && step !== "input" && (
        <div className="workspace-dock fixed left-1/2 -translate-x-1/2 z-[60] overflow-x-auto flex items-center bg-white border">
          {(["timeline", "forecast"] as const).map((key) => {
            const Icon = overlayMeta[key].icon;
            const isOpen = overlays.has(key);
            const label = key === "forecast" && forecastSpec ? "Forecast" : overlayMeta[key].label;
            return (
              <button
                key={key}
                aria-pressed={isOpen}
                onClick={() => toggleOverlay(key)}
                className={`shrink-0 flex items-center gap-1.5 font-medium ${
                  isOpen ? "bg-signal-blue text-white" : "text-ink-600 hover:bg-paper-100 hover:text-signal-blue"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
                {key === "timeline" && wikiTimeline.length > 0 && (
                  <span className={`min-w-[1.1rem] px-1 rounded-full text-xs ${isOpen ? "bg-white/20" : "bg-paper-200 text-ink-600"}`}>
                    {wikiTimeline.length}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
      {!showBenchmark && profileAgent && (
        <AgentProfilePanel
          agent={profileAgent}
          allAgents={agents}
          relationships={graphData.edges.map((e: any) => ({
            source: e.data.source,
            target: e.data.target,
            label: e.data.label || "",
            description: e.data.description || "",
          }))}
          onClose={() => setProfileAgent(null)}
          onOpenHistory={(eventId) => { setHistoryFocus({actorId:profileAgent.agent_id,eventId}); setProfileAgent(null); setOverlays(new Set<"timeline" | "forecast">(["timeline"])); }}
        />
      )}
      {overlays.size > 0 && (
        <div className={`fixed inset-0 z-50 ${showBenchmark ? "hidden" : ""}`}>
          <div className="workspace-scrim absolute inset-0" onClick={() => setOverlays(new Set())} />
          <div className="workspace-modal-stage relative h-full w-full flex flex-wrap items-stretch justify-center overflow-auto">
            {Array.from(overlays).map((key) => {
              const meta = overlayMeta[key];
              const label = key === "forecast" && forecastSpec ? "Forecast" : meta.label;
              const Icon = meta.icon;
              return (
                <div
                  key={key}
                  className="workspace-panel bg-white border overflow-hidden flex flex-col animate-fade-in"
                  style={{ height: "82vh" }}
                >
                  <div className="workspace-panel-heading shrink-0 flex items-center justify-between border-b">
                    <div className="flex items-center gap-2">
                      <Icon className="w-4 h-4 text-signal-blue" />
                      <h2 className="text-base font-semibold text-ink-800">{label}</h2>
                    </div>
                    <button aria-label={`Close ${label}`} onClick={() => toggleOverlay(key)} className="p-1.5 rounded-lg text-ink-400 hover:text-ink-800 hover:bg-paper-200/60 transition-colors">
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="workspace-panel-body flex-1 min-h-0 overflow-hidden flex flex-col">
                    {key === "timeline" && (
                      <div className="flex-1 min-h-0 overflow-y-auto">
                        <KnowledgePanel apiBase={API_BASE} caseId={eventId} historyFocus={historyFocus} disabled={loading} onReanalyze={() => handleGenerateAgents(true)} onOpenProfile={(id) => { const agent = agents.find((a) => a.agent_id === id); if (agent) setProfileAgent(agent); }} />
                      </div>
                    )}

                    {key === "forecast" && (
                      <div className="flex-1 min-h-0 overflow-y-auto">
                        {forecastSpec && eventId
                          ? <QuestionForecast key={eventId} caseId={eventId} spec={forecastSpec} config={getLLMConfig()}
                              ready={forecastActorReady} disabled={loading}
                              hasProfiles={agents.length > 0}
                              prepareLabel={profileActionLabel} onPrepare={prepareAnalysisActors}
                              taskBudget={{calls:budgetCalls ? Number(budgetCalls) : null,tokens:budgetTokens ? Number(budgetTokens) : null,seconds:budgetMinutes ? Number(budgetMinutes)*60 : null}} />
                          : <EnsemblePanel caseId={eventId} title={eventTitle} background={generatedBackground}
                              agents={agents} config={getLLMConfig()} disabled={loading} taskBudget={{calls:budgetCalls ? Number(budgetCalls) : null,tokens:budgetTokens ? Number(budgetTokens) : null,seconds:budgetMinutes ? Number(budgetMinutes)*60 : null}} />}

                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
