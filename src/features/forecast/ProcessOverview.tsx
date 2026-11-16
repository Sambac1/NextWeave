"use client";
import type { EnsembleResult, IssueChange, SampleReplay } from "@/services/experimentService";
import ReplayStepCard from "./ReplayStepCard";
import ReplayContext from "./ReplayContext";

export const sampleStatusLabel = (status: string) => ({
  completed: "Completed", failed: "Failed", interrupted: "Interrupted", pending: "Not started",
  queued: "Queued", running: "Running", partial: "Partial", budget_exceeded: "Budget exhausted",
}[status] || status || "Not started");

export default function ProcessOverview({ result, selectedSample, replay, loading, error, onSelect, onRetry,
  onDiagnostics, agents, apiBase, runId }: {
  result: EnsembleResult; selectedSample: number | null; replay: SampleReplay | null;
  loading: boolean; error: string; onSelect: (id: number) => void; onRetry: () => void;
  onDiagnostics: () => void; agents: any[]; apiBase: string; runId: string;
}) {
  const summary = result.process_summary;
  const paths = summary?.paths.slice(0, 3) || [];
  const selected = result.samples.find(s => s.sample_id === selectedSample);
  const path = summary?.paths.find(p => p.sample_ids.includes(selectedSample!));
  const terminal = result.categories.find(c => c.sample_ids?.includes(selectedSample!)
    || (replay?.classification?.category_id && c.id === replay.classification.category_id));
  const judgments = selected?.classification?.terminal_judgments || [];
  const actorNames: Record<string, string> = Object.fromEntries(agents.map(a => [a.agent_id, a.name]));
  const issueStates = new Map<string, IssueChange>();
  const steps = (replay?.rounds || []).map((round, index, rounds) => {
    const previousPending = Array.from(issueStates.values()).filter(c => c.pending_actor_ids?.length);
    for (const change of round.issue_updates || []) issueStates.set(change.issue_id, change);
    // Carry unresolved requests forward until a later saved update clears them.
    return { round, previousRound: rounds[index - 1], previousPending, pending: Array.from(issueStates.values()).filter(c => c.pending_actor_ids?.length) };
  });

  return <section aria-label="Simulation Process" className="space-y-5 border-t border-paper-300 pt-6">
    <div>
      <h4 className="font-semibold">Simulation Process</h4>
      <p className="mt-1 text-xs text-ink-500">View actor actions, state changes, and pending issues. Expand a step for the full record.</p>
    </div>
    {paths.length > 0 && <div className="flex flex-wrap gap-2" aria-label="Representative trajectories">
      {paths.map(p => <button key={p.id} type="button" aria-pressed={p.representative_sample_id === selectedSample}
        onClick={() => onSelect(p.representative_sample_id)}
        className={`min-h-11 rounded-lg border px-3 py-2 text-xs ${p.representative_sample_id === selectedSample
          ? "border-brand-700 bg-brand-50 text-brand-800" : "border-paper-300 bg-white text-ink-600 hover:border-brand-700"}`}>
        Representative trajectory #{p.representative_sample_id}{p.count > 1 ? ` · ${p.count} in this group` : ""}
      </button>)}
    </div>}
    {result.samples.length > 0 ? <label className="flex flex-wrap items-center gap-3 text-xs">
      Select trajectory
      <select aria-label="Select simulation trajectory" value={selectedSample ?? ""} onChange={e => onSelect(Number(e.target.value))}
        className="max-w-full min-w-0 rounded-lg border border-paper-300 bg-white p-2">
        {result.samples.map(s => <option key={s.sample_id} value={s.sample_id}>
          #{s.sample_id} · {sampleStatusLabel(s.status)} · {s.rounds_completed} steps
        </option>)}
      </select>
    </label> : <p className="text-sm text-ink-500">No simulation trajectories have been saved.</p>}
    {selected && <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-paper-100 px-4 py-3 text-xs">
      <p className="font-medium text-ink-700">Trajectory #{selected.sample_id} · {sampleStatusLabel(selected.status)}
        {terminal ? ` · ${terminal.label}` : ""}</p>
      {(selected.error || replay?.error || selected.status === "failed") && <button type="button" onClick={onDiagnostics}
        className="min-h-11 text-red-700 underline underline-offset-4">View failure reason</button>}
      {path && path.count > 1 && <p className="w-full text-ink-500">This trajectory belongs to a group of {path.count} with similar saved records.</p>}
    </div>}
    {selected?.classification_warning && <p className="text-xs text-amber-800">{selected.classification_warning}</p>}
    {!!judgments.length && <details className="rounded-lg border border-paper-300 px-4 text-sm">
      <summary className="min-h-11 cursor-pointer py-3">Terminal outcome and evidence</summary>
      <ul className="space-y-3 pb-4">
        {judgments.map(judgment => {
          const dimension = result.outcome_dimensions?.find(d => d.id === judgment.dimension_id);
          const category = dimension?.categories.find(c => c.id === judgment.category_id);
          return <li key={judgment.dimension_id} className="space-y-1">
            <p className="font-medium">{dimension?.label} · {category?.label || "Insufficient evidence"}</p>
            {!!judgment.reason && <p className="text-ink-600">{judgment.reason}</p>}
            {!!judgment.evidence_quote && <blockquote className="border-l border-paper-300 pl-3 text-ink-600">
              {judgment.evidence_quote}
            </blockquote>}
          </li>;
        })}
      </ul>
    </details>}
    <div aria-busy={loading}>
      {loading ? <p role="status" className="rounded-lg bg-paper-100 px-4 py-6 text-sm text-ink-500">Loading trajectory #{selectedSample}…</p>
        : error ? <div role="alert" className="rounded-lg bg-red-50 p-4 text-sm text-red-700">
          <p>Could not load the replay. {error}</p><button type="button" onClick={onRetry} className="mt-2 min-h-11 underline">Retry</button>
        </div> : replay ? <>
          {replay.rounds.length > 0 ? <ol className="space-y-4" aria-label={`Steps for trajectory ${replay.sample_id}`}>
            {steps.map(({ round, previousRound, pending, previousPending }) => <li key={round.step}>
              <ReplayStepCard round={round} previousRound={previousRound} actorNames={actorNames}
                pendingIssues={pending} previousPending={previousPending} dimensions={result.outcome_dimensions} />
            </li>)}
          </ol> : <p className="py-4 text-sm text-ink-500">This trajectory has no complete steps saved.</p>}
          <ReplayContext key={`${runId}:${replay.sample_id}:${replay.rounds.length}:${replay.status}`} replay={replay}
            agents={agents} apiBase={apiBase} runId={runId} />
        </> : selected ? <p className="py-4 text-sm text-ink-500">The replay will be available once this trajectory starts.</p> : null}
    </div>
    {summary && <details className="text-xs text-ink-500">
      <summary className="py-2">How representative trajectories are selected</summary>
      <p className="mt-2 leading-relaxed">{summary.comparison_basis}</p>
      <p className="mt-1">The saved runs form {summary.path_count} groups with {paths.length} representatives. Use the selector above to view any trajectory.</p>
      {summary.mode === "recorded_public_events" && <p className="mt-1">Legacy experiments use saved public records for the replay.</p>}
    </details>}
  </section>;
}
