"use client";
import { useEffect, useId, useRef, useState } from "react";
import { BarChart3, Loader2, Pause, Play, RotateCcw } from "lucide-react";
import {
  apiRequest,
  startExperiment,
  type DecisionPolicy,
  type ExperimentStatus,
  type Reassessment,
} from "@/services/experimentService";
import type { LLMConfig } from "@/services/llmService";
import EnsembleResults from "./EnsembleResults";
import { MAX_CONTEXT_CHARS } from "@/config/experiment";

export default function EnsemblePanel({
  caseId,
  title,
  background,
  agents,
  config,
  disabled,
  onBusyChange,
  taskBudget,
}: {
  caseId: string | null;
  title: string;
  background: string;
  agents: any[];
  config: LLMConfig;
  disabled: boolean;
  onBusyChange?: (busy: boolean) => void;
  taskBudget: {calls:number|null;tokens:number|null;seconds:number|null};
}) {
  const [count, setCount] = useState(8),
    [steps, setSteps] = useState(3);
  const [unit, setUnit] = useState("auto");
  const [stepSizeInput, setStepSizeInput] = useState("1");
  const [decisionPolicy, setDecisionPolicy] = useState<DecisionPolicy>("single");
  const decisionPolicyHelpId = useId();
  const stepSize = Number(stepSizeInput);
  const validStepSize = unit === "auto" || (Number.isInteger(stepSize) && stepSize >= 1 && stepSize <= 365);
  const contextCap = MAX_CONTEXT_CHARS;
  const [startDate, setStartDate] = useState("");
  const [costCap, setCostCap] = useState(""),
    [inputPrice, setInputPrice] = useState(""),
    [outputPrice, setOutputPrice] = useState("");
  const [cachedPrice, setCachedPrice] = useState(""),
    [budgetReady, setBudgetReady] = useState(false);
  const hydratedRun = useRef<string | null>(null);
  const [description, setDescription] = useState("");
  const [definitions, setDefinitions] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<ExperimentStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const poll = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useRef(true);
  const latestConfig = useRef(config);
  latestConfig.current = config;


  useEffect(() => {
    mounted.current = true;
    let cancelled = false;
    const savedRun = caseId ? localStorage.getItem(`ensemble_run_${caseId}`) : null;
    setRunId(savedRun);
    if (caseId && !savedRun && config.apiBase) {
      apiRequest(config.apiBase, `/experiments?event_id=${encodeURIComponent(caseId)}`)
        .then(value => {
          const latest = value.runs?.[0]?.run_id;
          if (!cancelled && latest) {
            setRunId(latest);
            localStorage.setItem(`ensemble_run_${caseId}`, latest);
          }
        }).catch((e: Error) => { if (!cancelled) setError(`Could not load saved experiments. ${e.message}`); });
    }
    setStatus(null);
    setBudgetReady(false);
    hydratedRun.current = null;
    return () => {
      cancelled = true;
      mounted.current = false;
      if (poll.current) clearTimeout(poll.current);
    };
  }, [caseId, config.apiBase]);

  const changeBusy = (value: boolean) => {
    setBusy(value);
    onBusyChange?.(value);
  };

  useEffect(() => {
    if (!runId || !config.apiBase) return;
    let cancelled = false;
    async function refresh() {
      try {
        const value: ExperimentStatus = await apiRequest(
          config.apiBase!,
          `/experiments/${runId}`,
        );
        if (cancelled || !mounted.current) return;
        setStatus(value);
        if (value.budget && hydratedRun.current !== runId) {
          const b = value.budget;
          setCostCap(
            b.limits.max_cost_usd == null ? "" : String(b.limits.max_cost_usd),
          );
          setInputPrice(b.pricing.input == null ? "" : String(b.pricing.input));
          setOutputPrice(
            b.pricing.output == null ? "" : String(b.pricing.output),
          );
          setCachedPrice(
            b.pricing.cached == null ? "" : String(b.pricing.cached),
          );
          hydratedRun.current = runId;
          setBudgetReady(true);
        }
        const running = ["queued", "running"].includes(value.status);
        changeBusy(running);
        if (running) poll.current = setTimeout(refresh, 2000);
      } catch (e: any) {
        if (!cancelled) {
          setError(e.message);
          changeBusy(false);
        }
      }
    }
    refresh();
    return () => {
      cancelled = true;
      if (poll.current) clearTimeout(poll.current);
    };
    // Starting/resuming increments the run object even when the ID is unchanged.
  }, [runId, config.apiBase, status?.status === "queued"]);

  async function reassess(): Promise<Reassessment> {
    const review: Reassessment = await apiRequest(config.apiBase || "", `/experiments/${runId}/reassess`, {
      api_key: config.apiKey,
      model: config.model || undefined,
      base_url: config.llmBaseUrl || undefined,
      language: config.language || "en-US",
      max_model_calls: taskBudget.calls,
      max_total_tokens: taskBudget.tokens,
      max_runtime_seconds: taskBudget.seconds,
    });
    setStatus(current => current?.run_id === runId ? { ...current, reassessment: review } : current);
    return review;
  }

  async function start(resume = false) {
    setError("");
    changeBusy(true);
    try {
      if (!resume && !validStepSize) throw new Error("Enter a whole number from 1 to 365 for the step length.");
      const categories =
        !resume && definitions.trim()
          ? definitions
              .split("\n")
              .filter((s) => s.trim())
              .map((line, index) => {
                const [label, ...rest] = line.split("|");
                if (!rest.join("|").trim())
                  throw new Error(
                    "Enter at least two outcome categories. Use one line per category with its name and criterion separated by a vertical bar.",
                  );
                return {
                  id: `category_${index + 1}`,
                  label: label.trim(),
                  criterion: rest.join("|").trim(),
                };
              })
          : [];
      const value = await startExperiment(latestConfig.current, {
        event_id: caseId,
        event_title: title,
        background,
        agents,
        sample_count: count,
        steps,
        max_runtime_seconds: taskBudget.seconds,
        scenario_description: description,
        outcome_definitions: categories,
        resume_run_id: resume ? runId : null,
        update_relationships: false,
        planning_mode: "auto",
        max_model_calls: taskBudget.calls,
        max_total_tokens: taskBudget.tokens,
        context_max_chars: contextCap,
        max_cost_usd: resume && costCap ? Number(costCap) : null,
        input_price_per_million: inputPrice ? Number(inputPrice) : null,
        output_price_per_million: outputPrice ? Number(outputPrice) : null,
        cached_input_price_per_million: cachedPrice
          ? Number(cachedPrice)
          : null,
        ...(!resume
          ? {
              step_unit: unit,
              step_size: unit === "auto" ? 1 : stepSize,
              decision_policy: decisionPolicy,
              initial_world_state: {},
              scenario_protocol: {},
              state_invariants: [],
              initial_private_information: {},
              start_date: startDate || null,
              event_questions: [],
            }
          : {}),
      });
      setStatus({
        run_id: value.run_id,
        status: "queued",
        stage: "prepared",
        sample_count: value.sample_count,
        steps: value.steps,
        completed_samples: 0,
        completed_rounds: 0,
        failed_samples: 0,
        decision_policy: resume ? status?.decision_policy ?? status?.result?.decision_policy : decisionPolicy,
      });
      setRunId(value.run_id);
      if (!resume) {
        hydratedRun.current = null;
        setBudgetReady(false);
      }
      if (caseId) localStorage.setItem(`ensemble_run_${caseId}`, value.run_id);
    } catch (e: any) {
      setError(e.message);
      changeBusy(false);
    }
  }

  async function stop() {
    if (!runId) return;
    try {
      await apiRequest(config.apiBase || "", `/experiments/${runId}/stop`, {});
    } catch (e: any) {
      setError(e.message);
    }
  }

  const result = status?.result;
  return (
    <div className="forecast-panel space-y-6 text-ink-700">
      <div className="flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-signal-blue" />
        <h3 className="text-sm font-semibold">Independent Simulations</h3>
      </div>
      <p className="text-xs text-ink-400 leading-relaxed">
        Each trajectory starts from the same historical state and tracks its own decisions, memory, and subsequent events. Outcomes are summarized after the runs finish.
      </p>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <label className="space-y-1">
          <span>Number of simulations</span>
          <input
            aria-label="Number of simulations"
            type="number"
            min={2}
            max={400}
            value={count}
            disabled={busy}
            onChange={(e) =>
              setCount(Math.max(2, Math.min(400, +e.target.value || 2)))
            }
            className="w-full border rounded-lg p-2 bg-white"
          />
        </label>
        <label className="space-y-1">
          <span>Steps per simulation</span>
          <input
            aria-label="Steps per simulation"
            type="number"
            min={1}
            max={12}
            value={steps}
            disabled={busy}
            onChange={(e) =>
              setSteps(Math.max(1, Math.min(12, +e.target.value || 1)))
            }
            className="w-full border rounded-lg p-2 bg-white"
          />
        </label>
        <label className="space-y-1">
          <span>Time unit per step</span>
          <select
            aria-label="Time unit per step"
            value={unit}
            disabled={busy}
            onChange={(e) => setUnit(e.target.value)}
            className="w-full border rounded-lg p-2 bg-white"
          >
            {[
              ["auto", "Automatic planning"],
              ["day", "day"],
              ["week", "week"],
              ["month", "month"],
              ["year", "year"],
            ].map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
        </label>
        {unit !== "auto" && <label className="space-y-1">
          <span>Units per step</span>
          <input type="number" min={1} max={365} step={1} value={stepSizeInput} disabled={busy}
            aria-label="Units per step" aria-invalid={!validStepSize}
            onChange={(e) => setStepSizeInput(e.target.value)}
            className="w-full border rounded-lg p-2 bg-white" />
        </label>}
      </div>
      {!validStepSize && <p className="text-xs text-red-700" role="alert">Enter a whole number from 1 to 365 for the step length.</p>}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        <label className="space-y-1"><span>Simulation start date (optional)</span>
          <input type="date" aria-label="Simulation start date" value={startDate} disabled={busy}
            onChange={e=>setStartDate(e.target.value)} className="w-full border rounded-lg p-2 bg-white" />
        </label>
      </div>
      <p className="text-xs text-ink-400">Leave blank to start at the latest date in the historical evidence.</p>
      <div className="space-y-2 text-xs">
        <label className="block space-y-1">
          <span>Decision policy for new experiments</span>
          <select
            value={decisionPolicy}
            disabled={busy}
            aria-describedby={decisionPolicyHelpId}
            onChange={(e) => setDecisionPolicy(e.target.value === "weighted" ? "weighted" : "single")}
            className="min-h-11 w-full rounded-lg border bg-white p-2"
          >
            <option value="single">Single candidate</option>
            <option value="weighted">Weighted sampling</option>
          </select>
        </label>
        <p id={decisionPolicyHelpId} className="leading-relaxed text-ink-500">
          {decisionPolicy === "weighted"
            ? "Each active actor generates up to three actions. One of the eligible actions is sampled according to its weight."
            : "Each active actor proposes one action. The action enters settlement if it passes review."} Resuming uses the experiment's saved policy.
        </p>
      </div>
      <label className="block text-xs space-y-1">
        <span>Scenario, constraints, and questions of interest</span>
        <textarea value={description} disabled={busy} onChange={e=>setDescription(e.target.value)}
          placeholder="Describe the scenario, constraints, and questions to simulate" rows={3}
          className="w-full border rounded-lg p-2 bg-white leading-relaxed" />
      </label>
      <details className="text-xs">
        <summary className="cursor-pointer text-signal-blue">
          Custom outcome categories (optional)
        </summary>
        <p className="text-ink-400 py-2">
          Use one line per category with the name and decision criterion separated by a vertical bar.
          Provide at least two mutually exclusive categories. If left blank, categories are generated and fixed before sampling.
        </p>
        <textarea
          value={definitions}
          disabled={busy}
          onChange={(e) => setDefinitions(e.target.value)}
          rows={3}
          className="w-full border rounded-lg p-2 bg-white"
        />
      </details>
      <p className="text-xs text-ink-400">
        Estimated {count * steps}{" "}
        simulation rounds, with two trajectories running in parallel by default.
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          disabled={disabled || busy || !agents.length || !config.apiBase}
          onClick={() => start()}
          className="rounded-lg px-4 py-2 bg-signal-blue text-white text-xs font-semibold disabled:opacity-40 flex items-center gap-2"
        >
          {busy ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <Play className="w-3 h-3" />
          )}
          Start Experiment
        </button>
        {runId &&
          !busy &&
          status?.status !== "completed" &&
          status?.resume_supported !== false &&
          !(status?.stage === "observation_preflight" && (status.result?.measurement_preflight?.pilot_samples ?? 0) >= 2) && (
            <button
              disabled={!budgetReady}
              onClick={() => start(true)}
              className="rounded-lg border px-3 py-2 text-xs flex items-center gap-2 disabled:opacity-40"
            >
              <RotateCcw className="w-3 h-3" />
              {budgetReady ? "Resume Unfinished Runs" : "Loading saved budget…"}
            </button>
          )}
        {busy && (
          <button
            onClick={stop}
            className="rounded-lg border px-3 py-2 text-xs flex items-center gap-2"
          >
            <Pause className="w-3 h-3" />
            Stop and Save Progress
          </button>
        )}
      </div>
      {error && (
        <p
          role="alert"
          className="text-xs text-red-700 bg-red-50 rounded-lg p-3"
        >
          {error}
        </p>
      )}
      {status && (
        <div className="rounded-lg bg-paper-100 p-3 space-y-2 text-xs">
          <p className="font-semibold">
            {(
              {
                queued: "Preparing",
                running: "Running",
                completed: "Completed",
                partial: "Partial · can resume",
                failed: "Failed · review and resume",
                interrupted: "Interrupted · can resume",
              } as Record<string, string>
            )[status.status] || status.status}
          </p>
          <p>
            Rounds {status.completed_rounds}/{status.sample_count * status.steps}{" "}
            · Completed {status.completed_samples}/{status.sample_count} · Failed{" "}
            {status.failed_samples}
          </p>
          {!result && status.decision_policy && <p>
            Decision policy · {status.decision_policy === "weighted" ? "Weighted sampling" : "Single candidate"}
          </p>}
          <div className="h-1.5 rounded-full bg-paper-200 overflow-hidden">
            <div
              className="h-full bg-signal-blue"
              style={{
                width: `${Math.min(100, ((status.completed_rounds || 0) / (status.sample_count * status.steps)) * 100)}%`,
              }}
            />
          </div>
          {!result && status.error && <p className="text-red-700">{status.error}</p>}
          {status.resume_supported === false && (
            <p className="text-amber-700">
              This experiment used an earlier engine. Reanalyze the sources to start a new experiment.
            </p>
          )}
          {!result && status.budget && (
            <p className="rounded-lg bg-paper-100 p-3 text-xs">
              Model requests {status.budget.actual_http_requests} · Accounted or reserved
              Token {status.budget.accounted_tokens_upper.toLocaleString("en-US")} /{" "}
              {status.budget.limits.max_total_tokens?.toLocaleString("en-US") || "Unlimited"}
              {status.budget.currency_estimate_available
                ? ` · Estimated cost ceiling $${status.budget.accounted_cost_upper_usd.toFixed(4)}`
                : ""}
              {status.budget.unknown_usage_requests
                ? ` · ${status.budget.unknown_usage_requests} requests with unreported usage`
                : ""}
            </p>
          )}
          {!result && status.plan && (
            <details>
              <summary className="cursor-pointer">Outcome criteria for this experiment</summary>
              {status.plan.outcome_dimensions?.map(dimension => <div key={dimension.id} className="py-2">
                <p className="font-medium">{dimension.label}</p>
                {dimension.categories.map(c => <p key={c.id} className="py-1 text-ink-500">{c.label} · {c.criterion}</p>)}
              </div>)}
              {status.plan.categories.map((c) => (
                <p key={c.id} className="py-1 text-ink-400">
                  {c.label}：{c.criterion}
                </p>
              ))}
            </details>
          )}
        </div>
      )}
      {result && status && <EnsembleResults key={`${result.run_id}:${config.apiBase}`} result={result}
        status={status} agents={agents} apiBase={config.apiBase || ""} onReassess={reassess} />}
    </div>
  );
}
