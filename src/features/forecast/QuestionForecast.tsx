import Link from "next/link";
import { useEffect, useState } from "react";
import { apiRequest, startExperiment, type ExperimentStatus } from "@/services/experimentService";
import { startQuestionForecast, type ForecastSpec } from "@/services/forecastTaskService";
import type { LLMConfig } from "@/services/llmService";
import ForecastReport from "./ForecastReport";

const action = "min-h-10 rounded-lg bg-brand-700 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-700 disabled:opacity-50";
const secondary = "min-h-10 rounded-lg border border-paper-300 px-3 py-2 text-sm hover:bg-paper-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-700 disabled:opacity-50";

function timeStepLimit(spec: ForecastSpec): number {
  if (!spec.outcome_deadline) return 12;
  const utcDay = (value: string) => Math.floor(Date.parse(value) / 86_400_000);
  return Math.min(12, Math.max(1, utcDay(spec.outcome_deadline) - utcDay(spec.as_of)));
}

export default function QuestionForecast({ caseId, spec, config, ready, disabled, hasProfiles, prepareLabel, onPrepare, taskBudget }: {
  caseId: string; spec: ForecastSpec; config: LLMConfig; ready: boolean; disabled: boolean; hasProfiles: boolean;
  prepareLabel: string;
  onPrepare: () => Promise<void>; taskBudget: { calls: number | null; tokens: number | null; seconds: number | null };
}) {
  const base = config.apiBase || "";
  const [runId, setRunId] = useState("");
  const [runs, setRuns] = useState<{run_id: string; created_at: string}[]>([]);
  const [status, setStatus] = useState<ExperimentStatus>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [count, setCount] = useState(8);
  const [steps, setSteps] = useState<number | null>(null);
  const maxSteps = timeStepLimit(spec);

  useEffect(() => {
    let active = true;
    apiRequest(base, `/experiments?event_id=${encodeURIComponent(caseId)}`)
      .then(data => {
        if (!active) return;
        const items = data.runs.filter((run: {question_forecast?: boolean}) => run.question_forecast);
        setRuns(items);
        setRunId(current => current || items[0]?.run_id || "");
      })
      .catch(failure => { if (active) setError(failure.message); })
      .finally(() => { if (active) setLoadingHistory(false); });
    return () => { active = false; };
  }, [base, caseId]);

  useEffect(() => {
    if (!runId || busy) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const value: ExperimentStatus = await apiRequest(base, `/experiments/${runId}`);
        if (!active) return;
        setStatus(value);
        if (value.status === "queued" || value.status === "running")
          timer = setTimeout(refresh, 1500);
      } catch (failure) {
        if (active) setError(failure instanceof Error ? failure.message : "Could not load the forecast.");
      }
    }
    void refresh();
    return () => { active = false; clearTimeout(timer); };
  }, [base, runId, busy]);

  async function start(resume = false) {
    setError("");
    setBusy(true);
    try {
      const result = resume
        ? await startExperiment(config, {
            event_title: spec.question, resume_run_id: runId,
            max_model_calls: taskBudget.calls, max_total_tokens: taskBudget.tokens, max_runtime_seconds: taskBudget.seconds,
          })
        : await startQuestionForecast(config, caseId, count, taskBudget, steps);
      if (!resume) {
        setRuns(rows => [{run_id: result.run_id, created_at: new Date().toISOString()}, ...rows]);
        setRunId(result.run_id);
      }
      setStatus(undefined);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The forecast could not be started.");
    } finally { setBusy(false); }
  }

  const pending = loadingHistory || !!runId && ((!status && !error) || status?.status === "queued" || status?.status === "running");
  const locked = busy || disabled || pending || !base;
  const report = status?.report;
  const settings = (
    <div className="space-y-4">
      {!report && spec.outcome_deadline && <p className="text-sm text-ink-600">
        Forecast deadline <strong>{new Date(spec.outcome_deadline).toLocaleString("en-US", {timeZone: "UTC", timeZoneName: "short"})}</strong>
      </p>}
      {runs.length > 1 && <label className="flex flex-wrap items-center gap-2 text-sm">
        <span>Saved forecast</span>
        <select className="rounded-lg border border-paper-300 bg-white p-2" value={runId}
          onChange={event => { setError(""); setStatus(undefined); setRunId(event.target.value); }}>
          {runs.map(run => <option key={run.run_id} value={run.run_id}>{new Date(run.created_at).toLocaleString("en-US")} ({run.run_id.slice(0, 8)})</option>)}
        </select>
      </label>}
      <div className="space-y-3">
        {!ready && <p className="text-sm text-ink-600">{hasProfiles
          ? "Only evidence participants are available. Forecasts need an actor with decision authority and initial BDI."
          : "Generate actor profiles from the collected evidence to enable the forecast."}</p>}
        <div className="flex flex-wrap items-end gap-3">
          <label className="grid gap-1 text-sm text-ink-700"><span>Trajectories</span>
            <input type="number" min={2} max={64} step={1} className="min-h-10 w-24 rounded-lg border border-paper-300 p-2" value={count}
              onChange={event => setCount(event.target.valueAsNumber || 0)} disabled={locked} />
          </label>
          <label className="grid gap-1 text-sm text-ink-700"><span>Time steps</span>
            <select className="min-h-10 rounded-lg border border-paper-300 bg-white p-2" value={steps ?? ""}
              aria-describedby="forecast-step-help" onChange={event => setSteps(event.target.value ? Number(event.target.value) : null)} disabled={locked}>
              <option value="">Automatic</option>
              {Array.from({ length: maxSteps }, (_, index) => index + 1).map(value => <option key={value} value={value}>{value}</option>)}
            </select>
          </label>
          {!ready && <button type="button" className={secondary} onClick={() => void onPrepare()} disabled={locked}>{prepareLabel}</button>}
          <button type="button" className={action} disabled={locked || !ready || !Number.isInteger(count) || count < 2 || count > 64}
            onClick={() => void start()}>Run forecast</button>
          {status?.status === "running" && <button type="button" className={secondary} onClick={async () => {
            try { await apiRequest(base, `/experiments/${runId}/stop`, {}); }
            catch (failure) { setError(failure instanceof Error ? failure.message : "The forecast could not be stopped."); }
          }}>Stop forecast</button>}
          {(status?.status === "interrupted" || status?.report?.resume_available) && !pending &&
            <button type="button" className={secondary} disabled={busy || disabled} onClick={() => void start(true)}>Resume forecast</button>}
        </div>
        <p id="forecast-step-help" className="text-xs text-ink-600">Automatic uses the forecast window and known milestones. A fixed count applies to new runs and keeps the same deadline.</p>
        {status && status.steps > 0 && <p className="text-xs text-ink-600">Selected run: {status.steps} time {status.steps === 1 ? "step" : "steps"} per trajectory.</p>}
      </div>
    </div>
  );
  return (
    <section className="mx-auto max-w-4xl space-y-4 px-4 py-6 sm:px-8">
      {report && <ForecastReport report={report} />}
      {error && <p role="alert" className="rounded-lg border border-signal-red/40 p-3 text-sm text-signal-red">{error}</p>}
      {report ? <>
        <details className="rounded-xl border border-paper-300 bg-white px-4">
          <summary className="min-h-11 py-3 text-sm font-semibold text-ink-800 hover:text-brand-800">Forecast settings and saved runs</summary>
          <div className="pb-4 pt-1">{settings}</div>
        </details>
        <Link href={`/?forecastRun=${encodeURIComponent(runId)}`} className="inline-flex min-h-11 items-center text-sm font-semibold text-brand-800 underline underline-offset-4">Open standalone report</Link>
      </> : settings}
      {pending && !error && <p role="status" className="text-sm text-ink-600">
        {status?.status === "running" ? `Running forecast · ${status.completed_rounds || 0} rounds settled` : "Loading forecast…"}
      </p>}
      {!report && status?.error && <p role="alert" className="text-sm text-signal-red">{status.error}</p>}
    </section>
  );
}
