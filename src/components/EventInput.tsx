"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import type { ConfirmForecast, ForecastPreview } from "@/services/forecastTaskService";

interface EventInputProps {
  onPreview: (query: string) => Promise<ForecastPreview>;
  onConfirm: (input: ConfirmForecast) => Promise<void>;
  loading: boolean;
}

const field = "w-full rounded-lg border border-paper-300 bg-paper-50 px-3 py-2.5 text-sm text-ink-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-700";
const action = "inline-flex min-h-11 items-center gap-2 rounded-lg bg-brand-700 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-700 disabled:opacity-50";

function utcDeadline(value: string): string {
  const instant = new Date(`${value}:00Z`);
  if (Number.isNaN(instant.getTime()) || instant.toISOString().slice(0, 16) !== value) {
    throw new Error("Choose a valid UTC date and time.");
  }
  return instant.toISOString();
}

function firstDeadlineMinute(cutoff: string): string {
  const minute = Math.floor(Date.parse(cutoff) / 60_000) + 1;
  return new Date(minute * 60_000).toISOString().slice(0, 16);
}

export default function EventInput({ onPreview, onConfirm, loading }: EventInputProps) {
  const [query, setQuery] = useState("");
  const [preview, setPreview] = useState<ForecastPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deadline, setDeadline] = useState("");
  const [yes, setYes] = useState("");
  const [no, setNo] = useState("");
  const [seeds, setSeeds] = useState("");
  const disabled = loading || busy;
  const spec = preview?.status === "confirm" ? preview.spec : null;

  async function review(input: string) {
    setBusy(true);
    setError("");
    try {
      const result = await onPreview(input.trim());
      if (result.status === "confirm") {
        setDeadline("");
        setYes("");
        setNo("");
      }
      setPreview(result);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The question could not be reviewed.");
    } finally { setBusy(false); }
  }

  async function confirm() {
    if (preview?.status !== "confirm") return;
    setBusy(true);
    setError("");
    try {
      await onConfirm({
        preview_id: preview.preview_id,
        ...(deadline ? { outcome_deadline: utcDeadline(deadline) } : {}),
        ...(spec?.kind === "question" && yes.trim() ? { yes_criterion: yes.trim() } : {}),
        ...(spec?.kind === "question" && no.trim() ? { no_criterion: no.trim() } : {}),
        seed_wiki_urls: seeds.split("\n").map(url => url.trim()).filter(Boolean),
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The forecast could not be prepared.");
    } finally { setBusy(false); }
  }

  return (
    <div className="workspace-input-form space-y-4 bg-white">
      {error && <p role="alert" className="rounded-lg border border-signal-red/40 px-3 py-2 text-sm text-signal-red">{error}</p>}
      {!spec && (
        <form className="space-y-3" onSubmit={(event) => { event.preventDefault(); void review(query); }}>
          <label className="block space-y-1.5">
            <span className="text-sm font-semibold text-ink-800">Forecast question or Polymarket URL</span>
            <textarea className={field} rows={2} required minLength={3} maxLength={1500} value={query}
              placeholder="Will the trade agreement be ratified by 2027-03-31?"
              onChange={event => { setQuery(event.target.value); setPreview(null); setError(""); }} disabled={disabled} />
          </label>
          <button type="submit" className={action} disabled={disabled || query.trim().length < 3}>
            {busy && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}Review question
          </button>
        </form>
      )}
      {preview?.status === "select_contract" && (
        <section aria-label="Contracts" className="border-t border-paper-200 pt-3">
          <h2 className="font-semibold">Choose a contract</h2>
          <ul className="mt-2 max-h-48 space-y-1 overflow-y-auto">
            {preview.choices.map(choice => <li key={choice.slug}>
              <button type="button" className="min-h-10 text-left text-sm text-brand-800 underline underline-offset-2"
                disabled={disabled} onClick={() => void review(choice.url)}>{choice.question}</button>
            </li>)}
          </ul>
        </section>
      )}
      {spec && (
        <form className="space-y-4" onSubmit={(event) => { event.preventDefault(); void confirm(); }}>
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-lg font-semibold text-ink-900">Confirm the forecast</h2>
            <button type="button" className="text-sm text-brand-800 underline" disabled={disabled}
              onClick={() => { setPreview(null); setError(""); }}>Edit input</button>
          </div>
          <p className="text-sm font-medium text-ink-800">{spec.question}</p>
          {spec.kind === "market" && <details open className="rounded-lg border border-paper-200 p-3 text-sm">
            <summary className="cursor-pointer font-semibold">Official resolution rules</summary>
            <p className="mt-2 max-h-44 overflow-y-auto whitespace-pre-wrap text-ink-700">{spec.rules}</p>
          </details>}
          {spec.outcome_deadline && <p className="text-sm text-ink-700">
            {spec.deadline_basis === "market_end_date_requires_rule_review" ? "Platform end time to verify" : "Proposed forecast deadline"}{" "}
            <strong>{new Date(spec.outcome_deadline).toLocaleString("en-US", {timeZone: "UTC", timeZoneName: "short"})}</strong>
          </p>}
          {spec.deadline_basis !== "explicit_contract_deadline" && (
            <label className="block space-y-1 text-sm text-ink-800">
              <span>{spec.outcome_deadline ? "Set a different deadline (UTC)" : "Forecast deadline (UTC)"}</span>
              <input type="datetime-local" step={60} className={`${field} min-h-11 min-w-0`} value={deadline}
                min={firstDeadlineMinute(spec.as_of)}
                required={!spec.outcome_deadline} disabled={disabled} onChange={event => setDeadline(event.target.value)} />
            </label>
          )}
          {spec.kind === "question" && <div className="space-y-2">
            <p id="forecast-criteria-help" className="text-sm text-ink-600">Both conditions are optional for a yes/no question. Blank conditions are generated automatically.</p>
            <label className="block space-y-1 text-sm text-ink-800"><span>What qualifies as Yes?</span>
              <textarea className={field} rows={2} maxLength={1200} value={yes} disabled={disabled}
                aria-describedby="forecast-criteria-help" onChange={event => setYes(event.target.value)} />
            </label>
            <label className="block space-y-1 text-sm text-ink-800"><span>What qualifies as No?</span>
              <textarea className={field} rows={2} maxLength={1200} value={no} disabled={disabled}
                aria-describedby="forecast-criteria-help" onChange={event => setNo(event.target.value)} />
            </label>
          </div>}
          <details className="text-sm text-ink-700">
            <summary className="cursor-pointer font-medium">Advanced</summary>
            <label className="mt-2 block space-y-1"><span>Optional Wikipedia pages, one URL per line</span>
              <textarea className={field} rows={2} value={seeds} disabled={disabled} onChange={event => setSeeds(event.target.value)} />
            </label>
          </details>
          <button type="submit" className={action} disabled={disabled}>
            {busy && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
            {busy ? "Finding Wikipedia evidence…" : "Confirm and find evidence"}
          </button>
        </form>
      )}
    </div>
  );
}
