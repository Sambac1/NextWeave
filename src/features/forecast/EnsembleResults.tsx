"use client";
import { useEffect, useRef, useState } from "react";
import { apiRequest, type EnsembleResult, type ExperimentStatus, type Frequency, type Reassessment, type SampleReplay } from "@/services/experimentService";
import EventFrequencyCard from "./EventFrequencyCard";
import ProcessOverview, { sampleStatusLabel } from "./ProcessOverview";

const percentage = (value: number | null | undefined) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const stepUnits: Record<string, string> = { day: "day", week: "week", month: "month", year: "year" };
function FrequencyRow({ value, total }: { value: Frequency; total: number }) {
  return <div className="frequency-row space-y-2">
    <div className="flex flex-wrap items-start justify-between gap-3 text-sm">
      <span className="font-medium">{value.label}</span>
      <span className="whitespace-nowrap font-semibold tabular-nums text-brand-800">{value.count}/{total} trajectories · {percentage(value.frequency)}</span>
    </div>
    <div className="h-2 overflow-hidden rounded-full bg-paper-200" role="img" aria-label={`${value.label} · ${value.count}/${total} trajectories · ${percentage(value.frequency)}`}>
      <div className="h-full rounded-full bg-brand-700" style={{ width: `${Math.max(0, Math.min(1, value.frequency || 0)) * 100}%` }} />
    </div>
    <details className="text-xs text-ink-500">
      <summary className="py-2">Outcome criteria and sampling range</summary>
      <div className="space-y-2 pt-2">
        <p>{value.criterion}</p>
        {value.sampling_interval_95 && <p>95% sampling interval · {value.sampling_interval_95.map(percentage).join(" – ")}</p>}
        {value.planned_sample_bounds && value.planned_sample_bounds[0] !== value.planned_sample_bounds[1] && <p>
          Bounds including unfinished runs · {value.planned_sample_bounds.map(percentage).join(" – ")}
        </p>}
        {!!value.unclear_count && <p>Another {value.unclear_count} trajectories could not be classified and are excluded from the occurrence count.</p>}
      </div>
    </details>
  </div>;
}

export default function EnsembleResults({ result: original, status, apiBase, agents, onReassess }: {
  result: EnsembleResult; status: ExperimentStatus; apiBase: string; agents: any[];
  onReassess: () => Promise<Reassessment>;
}) {
  const [showOriginal, setShowOriginal] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [reviewError, setReviewError] = useState("");
  const result = (!showOriginal && status.reassessment?.result) || original;
  const canReview = original.completed_samples > 0 && !["running", "queued"].includes(status.status);

  async function requestReview() {
    setReviewing(true);
    setReviewError("");
    try {
      await onReassess();
      setShowOriginal(false);
    } catch (error) {
      setReviewError(error instanceof Error ? error.message : "The reassessment could not be completed. Try again.");
    } finally {
      setReviewing(false);
    }
  }
  const [chosenSample, setChosenSample] = useState<number | null>(null);
  const [loaded, setLoaded] = useState<{ key: string; data: SampleReplay | null; error: string } | null>(null);
  const [retry, setRetry] = useState(0);
  const processRef = useRef<HTMLDivElement>(null);
  const diagnosticRef = useRef<HTMLDetailsElement>(null);
  const defaultSample = result.process_summary?.paths[0]?.representative_sample_id
    ?? result.samples.find(s => s.status === "completed")?.sample_id ?? result.samples[0]?.sample_id ?? null;
  const selectedSample = chosenSample != null && result.samples.some(s => s.sample_id === chosenSample) ? chosenSample : defaultSample;
  const sample = result.samples.find(s => s.sample_id === selectedSample);
  const notStarted = !!sample && ["pending", "queued", "not_started"].includes(sample.status) && !sample.rounds_completed;
  const shouldLoad = selectedSample != null && !notStarted;
  // Key by saved progress, not each poll response. A late response never replaces another sample.
  const requestKey = JSON.stringify([result.run_id, apiBase, selectedSample, sample?.status, sample?.rounds_completed, sample?.error, retry]);
  useEffect(() => {
    if (!shouldLoad) return;
    let cancelled = false;
    apiRequest(apiBase, `/experiments/${result.run_id}/samples/${selectedSample}`)
      .then((data: SampleReplay) => { if (!cancelled) setLoaded({ key: requestKey, data, error: "" }); })
      .catch((e: Error) => { if (!cancelled) setLoaded({ key: requestKey, data: null, error: e.message }); });
    return () => { cancelled = true; };
  }, [requestKey, shouldLoad, apiBase, result.run_id, selectedSample]);
  const current = shouldLoad && loaded?.key === requestKey ? loaded : null;
  const replay = current?.data || null;
  const reliability = result.reliability;
  const rejected = (reliability.bdi_updates_rejected || 0) + (reliability.goal_updates_rejected || 0) + (reliability.issue_updates_rejected || 0);
  const failures = result.samples.filter(s => s.error || ["failed", "interrupted", "budget_exceeded"].includes(s.status));
  const warningCount = rejected + result.failed_samples;
  const hasDiagnosticWarning = warningCount > 0 || !!status.error;
  const stepUnit = stepUnits[result.step_unit];
  const decisionPolicy = result.decision_policy ?? status.decision_policy;
  const sampling = result.decision_sampling;

  function inspectSample(id: number) {
    setChosenSample(id);
    // Scroll only for a user-requested cross-section jump, never on initial load.
    processRef.current?.focus({ preventScroll: true });
    processRef.current?.scrollIntoView({ behavior: "auto", block: "start" });
  }
  function showDiagnostics() {
    if (!diagnosticRef.current) return;
    diagnosticRef.current.open = true;
    diagnosticRef.current.querySelector("summary")?.focus({ preventScroll: true });
    diagnosticRef.current.scrollIntoView({ behavior: "auto", block: "start" });
  }

  return <div className="forecast-result space-y-6 border-paper-300">
    <section aria-label="Results Overview" className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><h4 className="font-semibold">Results Overview</h4>
          <p className="mt-1 text-xs text-ink-500">Completed {result.completed_samples} of {result.sample_count} simulations
            {result.simulation_start ? ` · Start date ${result.simulation_start}` : ""}
            {stepUnit && ` · Step length ${result.step_size ?? 1} ${stepUnit}`}
            {decisionPolicy && ` · ${decisionPolicy === "weighted" ? "Weighted sampling" : "Single candidate"}`}</p>
        </div>
        <button type="button" onClick={showDiagnostics} className={`min-h-11 rounded-lg border px-3 py-2 text-xs ${hasDiagnosticWarning
          ? "border-amber-200 bg-amber-50 text-amber-800" : "border-paper-300 text-ink-500 hover:bg-paper-100"}`}>
          {hasDiagnosticWarning ? "Review run diagnostics" : "Run Diagnostics"}
        </button>
      </div>
      {(canReview || status.reassessment) && <div className="space-y-2 border-t border-paper-200 pt-3">
        <div className="flex flex-wrap items-center gap-2">
          {status.reassessment && <button type="button" aria-pressed={showOriginal}
            onClick={() => setShowOriginal(!showOriginal)} className="min-h-11 rounded-lg border border-paper-300 px-3 text-sm">
            {showOriginal ? "View reassessed results" : "View original results"}
          </button>}
          {canReview && !status.reassessment && <button type="button" disabled={reviewing}
            onClick={requestReview} className="min-h-11 rounded-lg border border-paper-300 px-3 text-sm disabled:opacity-60">
            {reviewing ? "Reassessing…" : "Reassess saved trajectories"}
          </button>}
        </div>
        <p role="status" className="text-xs text-ink-600">
          {reviewing ? "Checking the evidence and recalculating the results."
            : status.reassessment ? (showOriginal ? "Showing the original results." : "Showing results reassessed under the current rules.")
              : "Reassessment uses the configured model to review completed trajectories."}
        </p>
        {reviewError && <p role="alert" className="text-sm text-red-700">{reviewError}</p>}
      </div>}
      {result.partition_validation === "unreviewed" && <p className="text-xs text-amber-800">These outcome categories have not been verified as mutually exclusive.</p>}
      {!!reliability.classification_fallbacks && <p className="text-xs text-amber-800">{reliability.classification_fallbacks} trajectories are classified as Other or Undetermined.</p>}
      {!reliability.complete_coverage && <p className="text-xs text-amber-800">
        Failed runs {result.failed_samples}{result.uncompleted_samples > result.failed_samples
          ? ` · ${result.uncompleted_samples - result.failed_samples} unfinished` : ""}.
      </p>}
      {result.measurement_preflight && !result.measurement_preflight.passed &&
        <div role="status" className="space-y-1 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <p className="font-medium">Sampling paused after the observation check failed</p>
          <p>Completed {result.measurement_preflight.pilot_samples} of {result.measurement_preflight.attempted_samples} pilot runs.
            {result.measurement_preflight.missing_dimensions.length > 0
              && ` Missing verifiable terminal observations for ${result.measurement_preflight.missing_dimensions.join(", ")}.`}</p>
          <p>The saved pilot observations are shown below for review.</p>
        </div>}
      <div className="space-y-3">
        <h5 className="text-sm font-medium">Outcome Distribution</h5>
        {result.outcome_dimensions?.length ? result.outcome_dimensions.map(dimension => (
          <section key={dimension.id} className="space-y-3 rounded-lg border border-paper-300 p-4">
            <h6 className="text-sm font-semibold">{dimension.label}</h6>
            {dimension.observation_scope && <p className="text-xs text-ink-600">{dimension.observation_scope}</p>}
            <p className="text-xs text-ink-600">Assessable trajectories {dimension.assessable_samples} of {result.completed_samples}
              {" · "}Insufficient evidence in {dimension.unclear_count}</p>
            {dimension.assessable_samples === 0
              ? <p className="text-sm font-medium">{dimension.observation_mode === "unavailable"
                ? "This scenario has no observation or mechanism for this outcome dimension"
                : "Insufficient evidence for an estimate"}</p>
              : result.measurement_preflight && !result.measurement_preflight.passed
                ? <ul className="space-y-1 text-sm">{dimension.categories.map(category => (
                  <li key={category.id}>{category.label} · {category.count} pilot runs</li>
                ))}</ul>
              : <>
                {dimension.assessable_samples < result.sample_count && <p className="text-xs text-ink-600">Percentages use the assessable trajectories.</p>}
                {dimension.categories.map(c => <FrequencyRow key={c.id} value={c} total={dimension.assessable_samples} />)}
              </>}
          </section>
        )) : result.completed_samples ? result.categories.map(c => <FrequencyRow key={c.id} value={c} total={result.completed_samples} />)
          : <p className="py-3 text-sm text-ink-500">The outcome distribution will appear when trajectories finish.</p>}
      </div>
      {result.event_frequencies.length > 0 && <div className="space-y-3 border-t border-paper-200 pt-5">
        <h5 className="text-sm font-medium">Event Metrics</h5>
        {result.event_frequencies.slice(0, 3).map(c => <EventFrequencyCard key={c.id} value={c} onInspect={inspectSample} />)}
        {result.event_frequencies.length > 3 && <details className="text-xs text-ink-500">
          <summary className="py-2">More predefined metrics · {result.event_frequencies.length - 3}</summary>
          <div className="space-y-3 pt-3">{result.event_frequencies.slice(3).map(c => <EventFrequencyCard key={c.id} value={c} onInspect={inspectSample} />)}</div>
        </details>}
      </div>}
      <p className="text-xs leading-relaxed text-ink-500">Percentages are frequencies under the selected model and scenario assumptions.</p>
    </section>

    <div ref={processRef} tabIndex={-1} className="min-w-0 scroll-mt-6 rounded-lg">
      <ProcessOverview result={result} selectedSample={selectedSample} replay={replay}
        loading={shouldLoad && !current} error={current?.error || ""} onSelect={setChosenSample}
        onRetry={() => setRetry(v => v + 1)} onDiagnostics={showDiagnostics}
        agents={agents} apiBase={apiBase} runId={result.run_id} />
    </div>

    <details ref={diagnosticRef} className="rounded-lg border border-paper-300 p-4 text-xs text-ink-500">
      <summary className="py-2 font-medium text-ink-700">Run Diagnostics{hasDiagnosticWarning ? " · issues recorded" : ""}</summary>
      <div className="space-y-5 pt-4">
        {status.error && <p className="whitespace-pre-wrap break-words text-red-700">{status.error}</p>}
        <div className="space-y-2">
          <p className="font-medium text-ink-700">Execution and validation</p>
          <p>Completed {result.completed_samples} · Failed {result.failed_samples} · Unfinished {result.uncompleted_samples}</p>
          <p>Rejected updates · Beliefs {reliability.bdi_updates_rejected || 0} · Goals {reliability.goal_updates_rejected || 0} · Issues {reliability.issue_updates_rejected || 0}</p>
          {!!rejected && <p>Rejected updates kept the preceding state. Completed steps are available in the replay.</p>}
          {!!reliability.goal_ids_repaired && <p>Goal identifiers repaired · {reliability.goal_ids_repaired}</p>}
          {reliability.deliberation_rounds != null && <p>{reliability.deliberation_rounds} rounds with actor deliberation · {reliability.rounds_without_deliberation || 0} rounds without deliberation · {reliability.external_actions || 0} external actions · {reliability.explicit_waits || 0} explicit waits</p>}
          {result.observed_activities && <p>Saved public events · {result.observed_activities.public_events}</p>}
        </div>
        {sampling && <div className="space-y-2">
          <h5 className="font-medium text-ink-700">Weighted Sampling Statistics</h5>
          <p>The saved rounds contain {sampling.decisions} actor decisions. Of those, {sampling.multi_candidate_decisions} had multiple eligible candidates.</p>
          <dl className="grid grid-cols-1 gap-x-5 gap-y-2 sm:grid-cols-2">
            {[
              ["Decisions with one eligible candidate", sampling.single_candidate_decisions, "" ],
              ["Multiple candidates with top weight ≥ 80%", sampling.concentrated_decisions, "" ],
              ["Actor rounds without deliberation", sampling.not_selected_for_deliberation, "" ],
              ["Decisions with added candidates", sampling.supplemented_decisions, "" ],
              ["Rejected candidates", sampling.rejected_candidates, ""],
              ["Merged duplicates", sampling.duplicate_candidates, ""],
            ].map(([label, count, unit]) => <div key={label} className="flex items-baseline justify-between gap-2">
              <dt>{label}</dt>
              <dd className="shrink-0 tabular-nums">{count} {unit}</dd>
            </div>)}
          </dl>
          {(sampling.public_issue_divergence_pairs ?? 0) > 0 && sampling.public_issue_reconvergence_pairs != null && (
            <p className="leading-relaxed">
              Of {sampling.public_issue_divergence_pairs} trajectory pairs with differing public issue states, {sampling.public_issue_reconvergence_pairs} later converged.
            </p>
          )}
        </div>}
        {failures.length > 0 && <div className="space-y-3">
          <p className="font-medium text-ink-700">Unfinished trajectories and failure reasons</p>
          {failures.map(s => <div key={s.sample_id} className="space-y-2 border-l-2 border-amber-200 pl-3">
            <button type="button" onClick={() => inspectSample(s.sample_id)} className="min-h-11 text-brand-800 underline underline-offset-4">
              View trajectory #{s.sample_id} · {sampleStatusLabel(s.status)} · {s.rounds_completed} steps saved
            </button>
            <p className="whitespace-pre-wrap break-words text-red-700">{s.error || (replay?.sample_id === s.sample_id ? replay.error : null) || "No failure reason was recorded."}</p>
          </div>)}
        </div>}
        {replay && !!replay.validation_issues?.length && <div className="space-y-2">
          <p className="font-medium text-ink-700">Validation log for trajectory #{replay.sample_id}</p>
          {Object.entries(replay.validation_issues.reduce<Record<string, number>>((counts, issue) => {
            const code = issue.code || "unclassified"; counts[code] = (counts[code] || 0) + 1; return counts;
          }, {})).map(([code, count]) => <p key={code}>{({ bdi_patch_rejected: "Rejected belief updates", goal_update_rejected: "Rejected goal updates",
            goal_id_repaired: "Repaired goal identifiers", issue_update_rejected: "Rejected issue updates" } as Record<string, string>)[code] || code} · {count}</p>)}
        </div>}
        {status.budget && <div className="space-y-2">
          <p className="font-medium text-ink-700">Run Usage</p>
          <p>Model requests {status.budget.actual_http_requests} · Accounted or reserved tokens {status.budget.accounted_tokens_upper.toLocaleString("en-US")} / {status.budget.limits.max_total_tokens?.toLocaleString("en-US") || "Unlimited"}</p>
          {status.budget.currency_estimate_available && <p>Estimated cost ceiling ${status.budget.accounted_cost_upper_usd.toFixed(4)}</p>}
          {!!status.budget.unknown_usage_requests && <p>{status.budget.unknown_usage_requests} requests have unreported usage.</p>}
        </div>}
      </div>
    </details>
  </div>;
}
