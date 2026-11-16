"use client";

import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import { marketComparisonPath } from "@/config/evaluationRoutes";
import { apiRequest } from "@/services/experimentService";
import {
  getBenchmarkCase, getBenchmarkRun, getRecentBenchmarkRuns, resumeBenchmarkRun,
  type BenchmarkCase, type BenchmarkRun, type RecentBenchmarkRun,
} from "@/services/benchmarkService";
import type { LLMConfig } from "@/services/llmService";
import EvaluationResults, { type EvaluationMethod, type EvaluationRun } from "./EvaluationResults";
import { comparisonMethods, comparisonRunLabels } from "./evaluationComparison";
import RunReport, { displayText } from "./RunReport";

function saveDownload(text: string, name: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const methods: EvaluationMethod[] = comparisonMethods.map(([method]) => method);
const methodPreferenceKey = "nextweave-evaluation-methods-v2";
const input = "mt-1 w-full rounded-md border border-paper-300 bg-white px-3 py-2 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-700";
const primary = "min-h-11 rounded-md bg-brand-700 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-800 disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-700";
const secondary = "min-h-11 rounded-md border border-paper-300 px-4 py-2 text-sm font-semibold text-brand-800 hover:bg-brand-50 disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-700";
const busyRun = (state?: string) => ["Collecting", "Running", "Pausing", "queued", "running", "pending", "preparing"].includes(state || "");
const isFinishedLegacy = (state?: string) => ["failed", "partial", "interrupted"].includes(state || "");

const defaultParameters = { qa_samples: 5, steps: 3, sample_count: 8, mcts_iterations: 20 };
type Parameter = keyof typeof defaultParameters;

export default function MarketBenchmark({ config, onOpenSettings }: { config: LLMConfig; onOpenSettings: () => void }) {
  const router = useRouter();
  const [view, setView] = useState<"new" | "results" | "question">("new");
  const [mode, setMode] = useState<"historical" | "live">("historical");
  const [byUrl, setByUrl] = useState(false);
  const [marketUrl, setMarketUrl] = useState("");
  const [count, setCount] = useState(30);
  const [selected, setSelected] = useState<EvaluationMethod[]>(methods);
  const [parameters, setParameters] = useState(defaultParameters);
  const [roundMode, setRoundMode] = useState<"adaptive" | "fixed">("adaptive");
  const [modelChoice, setModelChoice] = useState<"configured" | "default">();
  const [id, setId] = useState("");
  const [questionInputId, setQuestionInputId] = useState("");
  const [run, setRun] = useState<EvaluationRun>();
  const [resultMode, setResultMode] = useState<EvaluationRun["mode"]>("historical");
  const [evaluations, setEvaluations] = useState<Pick<EvaluationRun, "run_id" | "status" | "mode" | "model" | "quick_test">[]>([]);
  const [question, setQuestion] = useState<{ run: BenchmarkRun; snapshot: BenchmarkCase }>();
  const [recent, setRecent] = useState<RecentBenchmarkRun>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const api = config.apiBase || "";
  const model = config.model && modelChoice !== "default" ? "configured" : "default";
  const stepBased = selected.some((method) => ["single_trajectory_simulation", "multi_trajectory_simulation", "mcts_simulation"].includes(method));

  function route(key: "evaluationRun" | "benchmarkRun", value: string) {
    const url = new URL(window.location.href);
    url.searchParams.delete(key === "evaluationRun" ? "benchmarkRun" : "evaluationRun");
    url.searchParams.set(key, value);
    if (url.href !== window.location.href) {
      void router.replace(url.pathname + url.search + url.hash, undefined, { shallow: true, scroll: false });
    }
    if (key === "evaluationRun") setId(value);
    else setQuestionInputId(value);
  }

  function message(failure: unknown) {
    return displayText(failure instanceof Error ? failure.message : "", "The request failed. Check your backend connection or run ID.");
  }

  async function getQuestion(runId: string) {
    const result = await getBenchmarkRun(api, runId);
    const snapshot = result.case_snapshot || await getBenchmarkCase(api, result.case_id);
    return { run: result, snapshot };
  }

  async function openSaved(value: string, kind?: "evaluation" | "question") {
    if (!api || !value.trim()) return;
    const runId = value.trim();
    setBusy(true);
    setError("");
    try {
      if (kind !== "question") {
        try {
          const saved = await apiRequest(api, `/benchmark/evaluations/${encodeURIComponent(runId)}`) as EvaluationRun;
          setRun(saved);
          setResultMode(saved.mode);
          setQuestion(undefined);
          setView("results");
          route("evaluationRun", saved.run_id);
          return;
        } catch (failure) {
          if (kind === "evaluation" || !/Evaluation not found|Not Found|HTTP 404/i.test(message(failure))) throw failure;
        }
      }
      setQuestion(await getQuestion(runId));
      setRun(undefined);
      setView("question");
      route("benchmarkRun", runId);
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }

  async function openQuestion(runId: string) {
    setBusy(true);
    setError("");
    try {
      setQuestion(await getQuestion(runId));
      setQuestionInputId(runId);
      setView("question");
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!api) return;
    try {
      const saved = JSON.parse(window.localStorage.getItem(methodPreferenceKey) || "null");
      if (Array.isArray(saved)) {
        const valid = methods.filter((method) => saved.includes(method));
        if (valid.length) setSelected(valid);
      }
    } catch { /* Use the default selection. */ }
    const query = new URLSearchParams(window.location.search);
    const cohort = query.get("evaluationRun");
    const legacy = query.get("benchmarkRun");
    if (cohort) void openSaved(cohort, "evaluation");
    else if (legacy) void openSaved(legacy, "question");
    let active = true;
    void getRecentBenchmarkRuns(api).then((rows) => { if (active) setRecent(rows[0]); }).catch(() => {});
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  const { marketEvaluation, evaluationRun, benchmarkRun } = router.query;
  useEffect(() => {
    if (marketEvaluation === "1" && evaluationRun === undefined && benchmarkRun === undefined) {
      setView("new");
      setError("");
    }
  }, [marketEvaluation, evaluationRun, benchmarkRun]);

  useEffect(() => {
    if (!api || view !== "results") return;
    let active = true;
    void apiRequest(api, "/benchmark/evaluations").then((rows) => { if (active) setEvaluations(rows); }).catch(() => {});
    return () => { active = false; };
  }, [api, view, run?.status]);

  const activeId = run?.run_id;
  const running = busyRun(run?.status);
  const questionId = question?.run.run_id;
  const questionRunning = busyRun(question?.run.status);
  useEffect(() => {
    if (!api || busy || (!running && !questionRunning)) return;
    let stopped = false;
    async function poll() {
      try {
        if (activeId && running) {
          const update = await apiRequest(api, `/benchmark/evaluations/${activeId}`) as EvaluationRun;
          if (!stopped) setRun((current) => current?.run_id === activeId ? update : current);
        }
        if (questionId && questionRunning) {
          const update = await getBenchmarkRun(api, questionId);
          if (!stopped) setQuestion((current) => current?.run.run_id === questionId ? { ...current, run: update } : current);
        }
        if (!stopped) setError((old) => old === "Progress is temporarily unavailable. Retrying." ? "" : old);
      } catch {
        if (!stopped) setError("Progress is temporarily unavailable. Retrying.");
      }
    }
    const timer = window.setInterval(() => void poll(), 2500);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [api, activeId, running, questionId, questionRunning, busy]);

  async function launch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const quick = !byUrl && (event.nativeEvent as SubmitEvent).submitter?.getAttribute("value") === "quick";
    setBusy(true);
    setError("");
    try {
      const configured = model === "configured";
      if (configured && !config.apiKey?.trim()) {
        throw new Error("Add an API key for the configured provider in Settings, or choose Backend default.");
      }
      const started = await apiRequest(api, "/benchmark/evaluations", {
        mode, question_count: byUrl ? 1 : mode === "historical" ? count : 30, quick_test: quick,
        ...(byUrl && { market_url: marketUrl.trim() }),
        methods: methods.filter((method) => selected.includes(method)),
        ...(selected.includes("qa_ensemble") && { qa_samples: parameters.qa_samples }),
        ...(stepBased && { steps: roundMode === "fixed" ? parameters.steps : null }),
        ...(selected.includes("multi_trajectory_simulation") && { sample_count: parameters.sample_count }),
        ...(selected.includes("mcts_simulation") && { mcts_iterations: parameters.mcts_iterations }),
        model: configured ? config.model || null : null,
        base_url: configured ? config.llmBaseUrl || null : null,
        api_key: configured ? config.apiKey || null : null,
        use_backend_credentials: !configured, language: "en-US",
      }) as EvaluationRun;
      setRun(started);
      setResultMode(started.mode);
      setQuestion(undefined);
      setView("results");
      route("evaluationRun", started.run_id);
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }

  async function evaluationAction(action: "pause" | "resume" | "settlement") {
    if (!run) return;
    setBusy(true);
    setError("");
    try {
      setRun(await apiRequest(api, `/benchmark/evaluations/${run.run_id}/${action}`, action === "resume" ? { api_key: config.apiKey || null } : {}));
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }

  async function download(report: "forecasts" | "metrics") {
    if (!run) return;
    try {
      const response = await fetch(`${api}/benchmark/evaluations/${run.run_id}/export?report=${report}`);
      if (!response.ok) throw new Error("The report could not be exported.");
      saveDownload(await response.text(), `nextweave-${report}-${run.run_id}.csv`, "text/csv");
    } catch (failure) {
      setError(message(failure));
    }
  }

  function parameter(label: string, key: Parameter, min = 2, max = 64) {
    return <label className="block text-sm font-medium">{label}
      <input className={input} type="number" min={min} max={max} required value={parameters[key] || ""}
        onChange={(event) => setParameters((old) => ({ ...old, [key]: Number(event.target.value) }))} />
    </label>;
  }

  const focused = run?.questions?.find((row) => row.run_id === question?.run.run_id);
  const detail = question && {
    ...question.run,
    ...(focused?.quote != null && focused.quote_at && { market_reference: {
      probability: focused.quote, observed_on: focused.quote_at, label: "Market at Cutoff",
    }}),
  };

  return <section aria-label="Market Evaluation" className="mx-auto w-full max-w-5xl px-4 py-7 sm:px-8">
    <header className="mb-5">
      <p className="text-xs font-semibold uppercase tracking-wide text-brand-800">Politics · Wikipedia-only evidence</p>
      <h2 className="mt-1 font-display text-3xl font-semibold text-ink-900">Market Evaluation</h2>
    </header>
    <nav className="mb-5 flex flex-wrap gap-2 border-b border-paper-300 pb-3" aria-label="Evaluation views">
      {([["new", "New Evaluation"], ["results", "Evaluation Results"], ["question", "Question Detail"]] as const).map(([key, label]) => (
        <button type="button" key={key} aria-pressed={view === key} onClick={() => { setView(key); setError(""); }}
          className={view === key ? primary : secondary}>{label}</button>
      ))}
      <Link href={marketComparisonPath} className={`${secondary} inline-flex items-center`}>Historical &amp; Live Comparison</Link>
    </nav>
    {!api && <p className="mb-4 text-sm">Connect the backend to run an evaluation. <button type="button" className="underline" onClick={onOpenSettings}>Open Settings</button></p>}
    {error && <p role="alert" className="mb-4 rounded border border-signal-red px-3 py-2 text-sm text-signal-red">{error}</p>}
    {view === "new" && <form onSubmit={launch} className="space-y-5 rounded-lg border border-paper-300 bg-white p-4 sm:p-5">
      <fieldset disabled={!api || busy || running} className="space-y-5 disabled:opacity-60">
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-sm font-medium">Question Source
            <select className={input} value={byUrl ? "url" : "automatic"} onChange={(event) => {
              const manual = event.target.value === "url";
              setByUrl(manual);
              if (manual) setMode("live");
              setError("");
            }}>
              <option value="automatic">Automatic Discovery</option>
              <option value="url">Contract URL</option>
            </select>
          </label>
          {byUrl && <label className="text-sm font-medium sm:col-span-2">Polymarket Contract URL
            <input className={input} type="url" required maxLength={1500} value={marketUrl}
              placeholder="https://polymarket.com/event/…"
              onChange={(event) => setMarketUrl(event.target.value)} />
          </label>}
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-sm font-medium">Evaluation Mode
            <select className={input} value={mode} onChange={(event) => setMode(event.target.value === "live" ? "live" : "historical")}>
              <option value="historical">Historical</option><option value="live">Live</option>
            </select>
          </label>
          <label className="text-sm font-medium">Model
            <select className={input} value={model} onChange={(event) => setModelChoice(event.target.value === "configured" ? "configured" : "default")}>
              {config.model && <option value="configured">{config.model}</option>}
              <option value="default">Backend default</option>
            </select>
          </label>
          {!byUrl && (mode === "historical" ? <label className="text-sm font-medium">Number of Questions
            <select className={input} value={count} onChange={(event) => setCount(Number(event.target.value))}>
              {[5, 10, 20, 30, 45, 60].map((size) => <option key={size}>{size}</option>)}
            </select>
          </label> : <div className="text-sm"><p className="font-medium">Number of Questions</p><p className="mt-3">Eligible contracts from the daily Top 30</p></div>)}
        </div>
        <p className="text-sm text-ink-600">{byUrl
          ? mode === "historical"
            ? "Run one resolved Yes/No contract. A verified rules snapshot and Wikipedia evidence must be available from before settlement."
            : "Run one open Yes/No contract. NextWeave automatically finds and freezes Wikipedia evidence and saves the current market quote when available."
          : mode === "historical"
            ? "Resolved political contracts are screened automatically. Only cutoff-valid contract rules and Wikipedia revisions are used."
            : "The daily Polymarket Top 30 is screened for open political contracts. Wikipedia evidence is frozen before forecasts are generated."}</p>
        <fieldset>
          <legend className="mb-2 text-sm font-semibold">Forecasting Methods</legend>
          <div className="grid gap-2 sm:grid-cols-2">
            {comparisonMethods.map(([method, name]) => <label className="flex min-h-11 items-center gap-2 rounded border border-paper-300 px-3 text-sm" key={method}>
              <input type="checkbox" className="h-4 w-4 accent-brand-700" checked={selected.includes(method)}
                onChange={(event) => {
                  const next = event.target.checked ? [...selected, method] : selected.filter((item) => item !== method);
                  setSelected(next);
                  window.localStorage.setItem(methodPreferenceKey, JSON.stringify(next));
                }} />
              {name}
            </label>)}
          </div>
        </fieldset>
        <div className="grid gap-3 sm:grid-cols-3">
          {selected.includes("qa_ensemble") && parameter("QA Ensemble Size", "qa_samples")}
          {stepBased && <>
            <div>
              <label className="block text-sm font-medium">Round Planning
                <select className={input} value={roundMode} aria-describedby="benchmark-round-planning-help"
                  onChange={(event) => setRoundMode(event.target.value === "fixed" ? "fixed" : "adaptive")}>
                  <option value="adaptive">Adaptive (recommended)</option>
                  <option value="fixed">Fixed</option>
                </select>
              </label>
              <p id="benchmark-round-planning-help" className="mt-1 text-sm text-ink-600">
                {roundMode === "adaptive" ? "Plans 1–8 rounds from the contract window and milestones in its rules." : "Uses the same number of rounds for each multi-step forecast."}
              </p>
            </div>
            {roundMode === "fixed" && parameter("Rounds per Trajectory", "steps", 2, 12)}
          </>}
          {selected.includes("multi_trajectory_simulation") && <div>
            {parameter("Trajectories per Question", "sample_count")}
            <p className="mt-1 text-sm text-ink-600">More trajectories increase run time and model usage.</p>
          </div>}
          {selected.includes("mcts_simulation") && parameter("MCTS Search Iterations", "mcts_iterations", 1, 100)}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {!byUrl && <button type="submit" value="quick" className={secondary} disabled={!selected.length}>Quick Test</button>}
          <button type="submit" value="full" className={primary} disabled={!selected.length}>{busy ? "Starting…" : byUrl ? "Run Contract" : "Run Evaluation"}</button>
          {!byUrl && <span className="text-sm text-ink-600">Quick Test uses one eligible contract with the same workflow and selected methods.</span>}
        </div>
      </fieldset>
    </form>}
    {view !== "new" && (
      <form className="mb-4 flex flex-wrap items-end gap-2" onSubmit={(event) => {
        event.preventDefault();
        if (view === "question") void openSaved(questionInputId, "question");
        else void openSaved(id);
      }}>
        <label className="min-w-52 flex-1 text-sm font-medium">{view === "question" ? "Question Run ID" : "Open by Run ID"}
          <input className={input} value={view === "question" ? questionInputId : id}
            onChange={(event) => view === "question" ? setQuestionInputId(event.target.value) : setId(event.target.value)} required />
        </label>
        <button className={secondary} disabled={!api || busy}>Open Run</button>
        {recent && <button type="button" className={secondary} disabled={busy} onClick={() => void openSaved(recent.run_id, "question")}>Open Latest Contract Run</button>}
      </form>
    )}
    {view === "results" && <div className="space-y-4">
      <div className="flex flex-wrap gap-2" aria-label="Evaluation result modes">
        {(["historical", "live"] as const).map((key) => <button type="button" key={key}
          aria-pressed={resultMode === key} onClick={() => setResultMode(key)} className={resultMode === key ? primary : secondary}>
          {key === "historical" ? "Historical Evaluation" : "Live Evaluation"}
        </button>)}
      </div>
      {evaluations.some((row) => row.mode === resultMode && row.run_id !== run?.run_id) && <details className="text-sm">
        <summary className="cursor-pointer font-medium text-brand-800">Recent Evaluations</summary>
        <ul className="mt-2 space-y-1">
          {evaluations.filter((row) => row.mode === resultMode && row.run_id !== run?.run_id).slice(0, 8).map((row) => <li key={row.run_id}>
            <button type="button" className="min-h-11 break-all text-left text-brand-800 underline" disabled={busy}
              onClick={() => void openSaved(row.run_id, "evaluation")}>{row.status} · {row.model} · {row.run_id}</button>
          </li>)}
        </ul>
      </details>}
      {run && run.mode === resultMode ? <>
        <div className="rounded-md border border-paper-300 bg-white p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-semibold text-ink-900">{run.mode === "historical" ? "Historical Evaluation" : "Live Evaluation"}{run.market_url ? " · Contract URL" : run.quick_test ? " · Quick Test" : ""}</h3>
            <strong role="status" className="text-sm text-brand-800">{run.status}</strong>
          </div>
          <p className="mt-1 break-all text-sm">Run ID: {run.run_id} · {run.model}</p>
          {running && <p className="mt-2 text-sm">{run.total ? `Question ${run.current} of ${run.total}` : run.market_url
            ? "Loading the contract and freezing Wikipedia evidence." : "Screening contracts and freezing Wikipedia evidence."}
            {run.active_stage && <> · {run.active_stage.replace(/_/g, " ")}</>}</p>}
          {run.status === "Paused" && <p className="mt-2 text-sm">Progress saved{run.current && run.total ? ` at question ${run.current} of ${run.total}` : ""}. Resume when ready.</p>}
          {run.error && <p className="mt-2 text-sm text-signal-red">{run.error}</p>}
          {!!run.market_choices?.length && (
            <ul aria-label="Contracts in this event" className="mt-3 divide-y divide-paper-200 rounded-md border border-paper-300">
              {run.market_choices.map((choice) => <li key={choice.slug}>
                <button type="button" className="min-h-11 w-full px-3 py-2 text-left text-sm font-medium text-brand-800 hover:bg-brand-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-700"
                  onClick={() => {
                    setByUrl(true);
                    setMode(run.mode);
                    setMarketUrl(choice.url);
                    setError("");
                    setView("new");
                  }}>
                  {displayText(choice.question, choice.slug)}
                </button>
              </li>)}
            </ul>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            {(run.status === "Collecting" || run.status === "Running" || run.status === "Pausing") &&
              <button type="button" className={secondary} disabled={busy || run.status === "Pausing"}
                onClick={() => void evaluationAction("pause")}>{run.status === "Pausing" ? "Pausing…" : "Pause Evaluation"}</button>}
            {run.methods.every((method) => methods.includes(method)) && (run.status === "Paused" || run.status === "Failed" || run.status === "Partial") &&
              <button type="button" className={secondary} disabled={busy} onClick={() => void evaluationAction("resume")}>Resume Evaluation</button>}
            {run.mode === "live" && !running && !!run.questions?.length && <button type="button" className={secondary} disabled={busy} onClick={() => void evaluationAction("settlement")}>Refresh Settlement</button>}
            {!!run.questions?.length && <button type="button" className={secondary} onClick={() => void download("forecasts")}>Export CSV</button>}
            {run.scores && <button type="button" className={secondary} onClick={() => void download("metrics")}>Export Metrics CSV</button>}
          </div>
        </div>
        <EvaluationResults key={run.run_id} run={run} openQuestion={(value) => void openQuestion(value)} />
      </> : <p className="text-sm text-ink-600">Start an evaluation or enter a saved Run ID.</p>}
    </div>}
    {view === "question" && !question && <p className="text-sm text-ink-600">Enter a question run ID or open a question from an evaluation report.</p>}
    {view === "question" && question && detail && <section className="space-y-4 rounded-lg border border-paper-300 bg-white p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-semibold">{displayText(question.snapshot.question, "Contract question")}</h3>
        <p role="status" className="text-sm capitalize">{question.run.status}</p>
      </div>
      <p className="break-all text-sm">Run ID: {question.run.run_id} · Cutoff: {question.snapshot.as_of}</p>
      {focused?.resolution && <p className="text-sm">{focused.resolution}</p>}
      <details className="text-sm"><summary className="cursor-pointer font-medium text-brand-800">Question &amp; Resolution Rules</summary>
        <p className="mt-2 whitespace-pre-wrap">{displayText(question.snapshot.rules, "See the original contract for its rules.")}</p>
      </details>
      <details className="text-sm"><summary className="cursor-pointer font-medium text-brand-800">Wikipedia Evidence · {question.snapshot.evidence.length} revisions</summary>
        <ul className="mt-2 space-y-1">{question.snapshot.evidence.map((page) => <li key={`${page.url}:${page.revision_id}`}>
          <a href={page.revision_url || `${new URL(page.url).origin}/w/index.php?oldid=${encodeURIComponent(page.revision_id)}`} target="_blank"
            rel="noreferrer" className="text-brand-800 underline">{displayText(page.title, "Wikipedia revision")}</a>
          {" · "}{page.revision_at} · #{page.revision_id}
        </li>)}</ul>
      </details>
      {question.run.error && <p role="alert" className="text-sm text-signal-red">{displayText(question.run.error, "This run stopped. Review Diagnostics.")}</p>}
      {questionRunning && question.run.progress && <p className="text-sm">{question.run.progress.completed_samples ?? 0} / {question.run.progress.sample_count ?? "—"} trajectories completed</p>}
      {question.run.resume_available !== false && !run && question.run.methods?.every((method) =>
        Object.prototype.hasOwnProperty.call(comparisonRunLabels, method)) && isFinishedLegacy(question.run.status) && <button type="button" className={secondary} disabled={busy}
        onClick={async () => {
          setBusy(true);
          setError("");
          try { setQuestion({ ...question, run: await resumeBenchmarkRun(config, question.run.run_id, {}) }); }
          catch (failure) { setError(message(failure)); }
          finally { setBusy(false); }
        }}>Resume Run</button>}
      {detail.results?.some(({ method }) => Object.prototype.hasOwnProperty.call(comparisonRunLabels, method)) &&
        <RunReport key={detail.run_id} run={detail} methodLabels={comparisonRunLabels} showForecastCoverage={false} />}
    </section>}
  </section>;
}
