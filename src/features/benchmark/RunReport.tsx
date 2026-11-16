import { useState } from "react";
import type { BenchmarkMethod, BenchmarkResult, BenchmarkRun } from "@/services/benchmarkService";

export const methodNames: Record<string, string> = {
  direct_qa: "Direct QA", qa_ensemble: "QA Ensemble", one_step: "Single-step Simulation",
  single_multistep: "Single-trajectory Simulation", one_step_ensemble: "Single-step Ensemble",
  trajectory_sampling: "Multi-trajectory Simulation", trajectory_frequency: "Trajectory Count Baseline", mcts_simulation: "MCTS Simulation",
};

const percentFormat = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 2 });
export const percent = (value?: number | null) => value == null ? "—" : percentFormat.format(value);
const amount = new Intl.NumberFormat("en-US");
const routeLabels: Record<NonNullable<BenchmarkResult["simulation_route"]>, string> = {
  environment: "Environment model", actor: "Actor decisions", hybrid: "Actor and environment",
  semantic_replay: "Text-based replay", semantic_tree_search: "Text-based tree search", review: "Review required",
};

export function displayText(value: string | null | undefined, fallback: string) {
  const text = value
    ?.replace(/^(?:公开对象或受众|Public audience)[：:] */gm, "Public audience · ")
    .replace(/^(?:承诺|Commitment)[：:] */gm, "Commitment · ")
    .replace(/^(?:触发或复核条件|Trigger or review condition)[：:] */gm, "Trigger or review condition · ");
  return text && !/[一-龯]/.test(text) ? text : fallback;
}

function diagnosticError(message: string) {
  if (/BudgetExceeded/.test(message)) return "The run reached a budget or time limit. Increase the resume budget if needed.";
  return displayText(message, "This method did not finish. Check the provider and run budget before resuming.");
}

const runStates: Record<string, string> = {
  completed: "Completed", partial: "Partial", interrupted: "Interrupted", failed: "Failed",
  unsupported: "Unsupported", unresolved: "Outcome unestimated", invalid_trace: "Invalid action trace",
  not_started: "Not started", running: "Running", queued: "Queued",
};
const outlookIssues: Record<string, string> = {
  missing_outlook: "The settled state did not yield a conditional estimate.",
  unsupported_quote: "The estimate could not be linked to the quoted public evidence.",
  non_substantive_quote: "The estimate needs a verbatim source passage or a substantive data-row citation.",
  readout_deferred_for_terminal_budget: "The round was settled. The probability readout was deferred to prioritize final estimates.",
  unverified_certainty: "The observed state does not establish a certain result.",
  missing_change_driver: "No supported change in the settled state explains the updated estimate.",
  no_terminal_observation: "No public observation supports a new estimate at the deadline.",
  missing_observation_mechanism: "The count window is known, but this simulation has no supported mechanism for observing the count.",
  missing_public_evidence: "The simulation has no public evidence for this estimate.",
  outlook_call_unavailable: "The public estimate could not be generated. The settled round was saved.",
};

function methodState(result: BenchmarkResult) {
  if (result.error_code === "needs_actor_evidence") return "Needs evidence";
  if (result.error_code === "retryable_actor_profile" || (result.error_code === "actor_preparation_error" && result.resume_available)) return "Retryable error";
  return result.status ? runStates[result.status] || result.status : null;
}

function methodError(result: BenchmarkResult) {
  if (result.error_code === "needs_actor_evidence") return displayText(result.error, "The frozen evidence does not establish the required actor's decision scope. Add relevant sources before starting a new run.");
  if (result.error_code === "retryable_actor_profile") return "Actor profile generation failed. Resume from saved preparation when available.";
  if (result.status === "unsupported") return displayText(result.error, "The frozen evidence does not support a qualified decision actor.");
  return result.error ? diagnosticError(result.error) : null;
}

function ProbabilityTrack({ value, reference }: { value: number; reference?: number }) {
  return (
    <div className="relative mx-1 h-6" aria-hidden="true">
      <div className="absolute inset-x-0 top-1/2 border-t border-paper-300" />
      {reference != null && <span className="absolute inset-y-0 border-l-2 border-dashed border-signal-amber" style={{ left: percent(reference) }} />}
      <span className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand-700 ring-2 ring-white" style={{ left: percent(value) }} />
    </div>
  );
}

export default function RunReport({ run, methodLabels, showForecastCoverage = true }: {
  run: BenchmarkRun;
  methodLabels?: Partial<Record<BenchmarkMethod, string>>;
  showForecastCoverage?: boolean;
}) {
  const [tab, setTab] = useState("Forecast Comparison");
  const [selection, setSelection] = useState<string>();
  const [sample, setSample] = useState<number>();
  const actorCoverage = run.preparation?.actor_coverage;
  const otherUnresolved = actorCoverage?.unresolved_requirements.filter(
    text => !actorCoverage.missing_essential?.includes(text),
  ) ?? [];
  const savedResults = run.results || [];
  const results = methodLabels ? Object.keys(methodLabels).flatMap((method) => {
    const result = savedResults.find((row) => row.method === method);
    return result ? [result] : [];
  }) : savedResults;
  const roundPlans = Object.entries(run.round_plans ?? {}).filter(([method]) =>
    !methodLabels || Object.prototype.hasOwnProperty.call(methodLabels, method));
  const simulations = results.filter((result) => result.trajectories?.length);
  const current = results.find((result) => result.method === selection) || results[0];
  const simulation = simulations.find((result) => result.method === selection) || simulations[0];
  const trajectory = simulation?.trajectories?.find((row) => row.sample_id === sample) || simulation?.trajectories?.[0];
  const reference = run.market_reference;
  const legacy = results.some((result) => result.probability_method?.startsWith("conditional_terminal_")
    || result.probability_method === "poll_bootstrap_national_hamilton_v1"
    || result.probability_method === "sequential_sampled_outlook_v1");
  const panel = "rounded-md border border-paper-300 bg-white p-4 sm:p-5";
  const control = "rounded px-3 py-2 text-sm font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-700";
  const label = (method: string) => methodLabels?.[method as BenchmarkMethod] || methodNames[method] || method;

  function diagnostics(result: BenchmarkResult) {
    return (
      <div className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
        {[
          ...(showForecastCoverage ? [["Forecast coverage", percent(result.coverage)]] : []),
          ["Model requests", result.calls == null ? "—" : amount.format(result.calls)],
          ["Tokens", result.tokens == null ? "—" : amount.format(result.tokens)],
          ["Runtime", result.elapsed_seconds == null ? "—" : `${amount.format(result.elapsed_seconds)} s`],
          ...(result.settled_replay_coverage == null ? [] : [
            ["Resolved replay weight", percent(result.settled_replay_coverage)],
            ["Unresolved replay weight", percent(result.unresolved_replay_mass)],
            ["Failed or interrupted weight", percent(result.failed_or_interrupted_mass)],
            ["Unrun weight", percent(result.unrun_mass)],
          ]),
        ].map(([label, value]) => <div key={label}><dt className="text-ink-600">{label}</dt><dd className="font-semibold text-ink-900">{value}</dd></div>)}
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-4">
      {run.preparation?.actor_coverage && <details className="rounded-md border border-paper-300 px-4 py-2 text-sm">
        <summary className="min-h-11 cursor-pointer py-3 font-medium text-ink-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600">
          Decision seats · {run.preparation.actor_coverage.seats.length} identified · {run.preparation.actor_coverage.status === "ready" ? "Coverage checked" : run.preparation.actor_coverage.status === "observation_only" ? "Observation mechanism needed" : "Coverage incomplete"}
        </summary>
        <p className="mb-3 text-ink-600">A seat represents an independent choice. Unknown intentions do not remove an actor; execution still requires its own authority.</p>
        <ul className="space-y-3">
          {run.preparation.actor_coverage.seats.map(seat => <li key={seat.seat_id} className="border-t border-paper-200 pt-3">
            <div className="flex flex-wrap justify-between gap-2 font-medium"><span>{seat.name}{seat.required ? " · Required" : ""}</span>
              <span>{({ready:"Ready", restricted:"Limited actions", needs_evidence:"Needs evidence", retryable_error:"Profile repair needed", missing:"Missing profile"} as Record<string,string>)[seat.readiness] || seat.readiness}</span>
            </div>
            <p className="mt-1 text-ink-600">{displayText(seat.role, "See the saved decision-seat evidence.")}</p>
          </li>)}
        </ul>
        {!!run.preparation.actor_coverage.missing_required.length && <p className="mt-3 font-medium text-ink-900">Missing required actors: {run.preparation.actor_coverage.missing_required.join(", ")}</p>}
        {!!run.preparation.actor_coverage.missing_essential?.length && <p className="mt-2 font-medium text-ink-900">Essential decisions without a verified actor: {run.preparation.actor_coverage.missing_essential.map(text => displayText(text, "Unresolved decision owner")).join("; ")}</p>}
        {otherUnresolved.length > 0 && <p className="mt-2 text-ink-700">Other unresolved decisions: {otherUnresolved.map(text => displayText(text, "Unresolved decision owner")).join("; ")}</p>}
      </details>}
      {run.resolution_status === "awaiting_resolution" && <p className="text-sm font-medium text-ink-700">Awaiting Resolution</p>}
      {run.outcome_deadline && <p className="text-sm text-ink-600">
        {run.deadline_basis === "scheduled_election_count_window_assumption" ? "Assumed simulation horizon" : "Simulation horizon"}
        {" · "}{run.outcome_deadline}
        {run.nominal_market_end && <> · Market metadata end {run.nominal_market_end}</>}
      </p>}
      {legacy && <p className="rounded-md border border-paper-300 px-3 py-2 text-sm text-ink-700">
        This report was generated with an earlier simulation protocol.
      </p>}
      {run.source_run_id && <p className="break-words text-sm text-ink-600">Source run: {run.source_run_id}
        {run.rescore_usage && <> · Rescore: {run.rescore_usage.calls} requests, {amount.format(run.rescore_usage.tokens)} tokens</>}
      </p>}
      <nav aria-label="Run report views" className="flex flex-wrap gap-1 border-b border-paper-300 pb-2">
        {["Forecast Comparison", "Trajectory Explorer", "Diagnostics"].map((name) => (
          <button key={name} type="button" aria-pressed={tab === name} onClick={() => setTab(name)}
            className={`${control} ${tab === name ? "bg-brand-700 text-white" : "text-brand-800 hover:bg-brand-50"}`}>{name}</button>
        ))}
      </nav>
      {tab === "Forecast Comparison" && (
        <section className={panel} aria-label="Forecast comparison">
          <h4 className="font-semibold text-ink-900">Probability of Yes</h4>
          {reference && <p className="mt-1 text-sm text-ink-600">
            {reference.label}: <strong>{percent(reference.probability)}</strong> · {reference.observed_on}
            {reference.url?.startsWith("https://polymarket.com/") && <> · <a href={reference.url} target="_blank" rel="noreferrer" className="text-brand-800 underline">View contract</a></>}
          </p>}
          <div className="mt-3 space-y-2">
            {results.map((result) => (
              <button key={result.method} type="button" onClick={() => setSelection(result.method)}
                aria-pressed={current?.method === result.method}
                className="grid min-h-11 w-full grid-cols-[minmax(0,1fr)_minmax(48px,1.5fr)_4rem] items-center gap-2 rounded px-2 text-left text-sm hover:bg-brand-50 focus-visible:outline focus-visible:outline-brand-700">
                <span>{label(result.method)}
                  {result.p_yes == null && methodState(result) && <span className="mt-0.5 block text-xs text-ink-600">{methodState(result)}</span>}
                </span>
                {result.p_yes != null ? <ProbabilityTrack value={result.p_yes} reference={reference?.probability} /> : <span />}
                <span className="text-right font-semibold tabular-nums">{percent(result.p_yes)}</span>
              </button>
            ))}
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)_minmax(48px,1.5fr)_4rem] gap-2 px-2 text-xs text-ink-600" aria-hidden="true">
            <span /><div className="flex justify-between"><span>0%</span><span>50%</span><span>100%</span></div>
          </div>
          {current && <div className="mt-4 space-y-2 border-t border-paper-200 pt-3 text-sm">
            {current.p_yes == null && methodError(current) && <p>{methodError(current)}</p>}
            {current.provenance && <p>{current.provenance}</p>}
            {reference && current.p_yes != null && <p>Absolute difference: {Math.abs((current.p_yes - reference.probability) * 100).toFixed(2)} percentage points</p>}
            {diagnostics(current)}
            {!!current.members?.length && <details className="text-sm">
              <summary className="cursor-pointer font-medium text-brand-800">Member Forecasts</summary>
              {current.members.map((member) => <p key={member.index}>#{member.index} · {percent(member.p_yes)} · {member.status} · {member.actual_http_requests} requests · {member.prompt_tokens + member.completion_tokens} tokens</p>)}
            </details>}
            {!!current.tree?.length && <details className="text-sm">
              <summary className="cursor-pointer font-medium text-brand-800">MCTS Search Tree</summary>
              <p>{current.search_iterations} expansions · {percent(current.unexplored_mass)} residual mass</p>
              <div className="max-w-full overflow-x-auto">
                <table className="min-w-max text-left"><thead><tr>
                  {["Path", "Depth", "Visits", "Search Value", "Path Mass", "Conditional Yes", "Contribution"].map((title) => <th key={title} scope="col" className="pr-3">{title}</th>)}
                </tr></thead><tbody>{current.tree.map((node) => <tr key={node.path.join(".")} className="border-t border-paper-200">
                  <th scope="row" className="pr-3">{node.path.join(".") || "Root"}</th>
                  <td>{node.depth}</td><td>{node.visits}</td><td>{node.search_value?.toFixed(3) ?? "—"}</td>
                  <td>{percent(node.mass)}</td><td>{percent(node.conditional_p_yes)}</td><td>{percent(node.contribution)}</td>
                </tr>)}</tbody></table>
              </div>
            </details>}
          </div>}
        </section>
      )}
      {tab === "Trajectory Explorer" && (
        <section className={panel} aria-label="Trajectory explorer">
          {!simulation || !trajectory ? <p>No simulation trajectories are available for this run.</p> : <>
            <div className="flex flex-wrap gap-3 text-sm">
              <label>Method <select value={simulation.method} onChange={(e) => { setSelection(e.target.value); setSample(undefined); }} className="ml-1 rounded border border-paper-300 px-2 py-1">
                {simulations.map((row) => <option key={row.method} value={row.method}>{label(row.method)}</option>)}
              </select></label>
              <label>Trajectory <select value={trajectory.sample_id} onChange={(e) => setSample(Number(e.target.value))} className="ml-1 rounded border border-paper-300 px-2 py-1">
                {simulation.trajectories?.map((row) => <option key={row.sample_id} value={row.sample_id}>#{row.sample_id} · {percent(row.p_yes)}</option>)}
              </select></label>
            </div>
            <div className="mt-3">
              {simulation.trajectories?.map((row) => row.p_yes != null && (
                <button key={row.sample_id} type="button" aria-pressed={row.sample_id === trajectory.sample_id}
                  onClick={() => setSample(row.sample_id)}
                  className="grid min-h-11 w-full grid-cols-[5rem_1fr_4rem] items-center gap-2 rounded px-2 text-left text-sm hover:bg-brand-50 focus-visible:outline focus-visible:outline-brand-700">
                  <span>#{row.sample_id}</span><ProbabilityTrack value={row.p_yes} /><strong className="text-right">{percent(row.p_yes)}</strong>
                </button>
              ))}
            </div>
            <h4 className="mt-4 font-semibold">Round Timeline</h4>
            {trajectory.status && trajectory.status !== "completed" && <p className="mt-1 text-sm text-ink-600">{runStates[trajectory.status] || trajectory.status}{trajectory.error ? ` · ${diagnosticError(trajectory.error)}` : ""}</p>}
            {trajectory.action_path_probability != null && <p className="mt-1 text-sm text-ink-600">
              Product of sampled action probabilities: {percent(trajectory.action_path_probability)} ·
              Final conditional Yes: {percent(trajectory.p_yes)}
            </p>}
            <ol className="mt-2 space-y-2 border-l border-paper-300 pl-4 text-sm">
              {trajectory.rounds?.map((round) => (
                <li key={round.step}>
                  <strong>Round {round.step}</strong> {round.interval?.interval_end && <span className="text-ink-600"> · through {round.interval.interval_end_utc ?? round.interval.interval_end}</span>}
                  <ul className="mt-1 space-y-1">{round.events.slice(0, 8).map((event) => <li key={event.id}>{displayText(event.text, "Original event text is available in the run export.")}</li>)}</ul>
                  {!round.events.length && <p>No public event was recorded.</p>}
                  {round.deliberation_count === 0 && <p className="text-ink-700">No actor entered deliberation.</p>}
                  {round.sampled_choices?.map((choice) => <div key={choice.actor_id}>
                    <p>
                    {choice.actor_name} · {percent(choice.p_selected)} selection probability
                    {choice.waited && <> · Chose to wait or monitor</>}
                    {choice.action && <> · {displayText(choice.action, "See the original action in the run export.")}</>}
                    </p>
                    {!!choice.public_options?.length && <details className="mt-1">
                      <summary className="min-h-11 cursor-pointer py-2 text-brand-800">Candidate probabilities</summary>
                      <p className="mb-1 text-xs text-ink-600">Public candidates · probabilities after audit</p>
                      <table className="w-full table-fixed text-left">
                        <caption className="sr-only">Candidate selection probabilities for {choice.actor_name}</caption>
                        <thead><tr><th scope="col" className="py-1">Action</th><th scope="col" className="w-28 py-1 text-right">Probability</th></tr></thead>
                        <tbody>{choice.public_options.map((option) => <tr key={option.candidate_id} className="border-t border-paper-200 align-top">
                          <td className="break-words py-2 pr-3">{displayText(option.action, "Original action in run export")}{option.selected && <strong className="ml-2 text-brand-800">Selected</strong>}</td>
                          <td className="py-2 text-right tabular-nums">{percent(option.probability)}</td>
                        </tr>)}</tbody>
                      </table>
                      {!!choice.public_rejections?.length && <ul className="mt-2 space-y-1 text-ink-600">{choice.public_rejections.map((option) => <li key={option.candidate_id}>
                        Excluded · {displayText(option.action, "Original action in run export")} · {option.code.replace(/_/g, " ")}
                      </li>)}</ul>}
                    </details>}
                  </div>)}
                  {!!round.action_results?.length && <p className="mt-1 text-ink-600">
                    {round.action_results.map((result) => `${result.actor_name} · ${result.status}`).join(" / ")}
                  </p>}
                  {(!!round.interactions?.length || !!round.question_updates?.length || round.pending_trigger_rejected) && <details className="mt-1">
                    <summary className="min-h-11 cursor-pointer py-2 text-brand-800">Settlement and question progress</summary>
                    {round.pending_trigger_rejected && <p className="mb-1 text-ink-700">A proposed trigger for the next round could not be verified.</p>}
                    <ul className="space-y-2">
                      {round.interactions?.map((row, index) => <li key={`interaction-${index}`}>
                        <strong>{row.kind.replace(/_/g, " ")} · {row.outcome}</strong>
                        <p className="text-ink-600">{displayText(row.effect, "See the public event in the export.")}</p>
                      </li>)}
                      {round.question_updates?.map((row) => <li key={row.condition_id}>
                        <strong>{row.label} · {row.before === row.after ? row.after : `${row.before} → ${row.after}`}</strong>
                        <p className="text-ink-600">{displayText(row.summary, "No new public evidence.")}</p>
                        {!!row.pending_actor_ids?.length && <p className="text-ink-700">
                          Follow-up recorded for {row.pending_actor_ids.length} {row.pending_actor_ids.length === 1 ? "actor" : "actors"}.
                        </p>}
                      </li>)}
                    </ul>
                  </details>}
                  {round.state_changes?.map((change, index) => <p key={`${change.path}-${index}`} className="text-ink-600">
                    State update · {displayText(change.reason, "See the state update in the run export.")}
                  </p>)}
                  {round.contract_outlook && <p className="mt-1 text-ink-700">
                    {round.contract_outlook.estimate_kind === "terminal_estimate" ? "Final estimated Yes" :
                      round.contract_outlook.estimate_kind === "terminal_carried_forward" ? "Final Yes estimate carried forward" :
                      round.contract_outlook.estimate_kind === "verified_simulation" ? "Verified in simulation" : "Conditional Yes"}
                    {" "}{percent(round.contract_outlook.p_yes)} · {displayText(round.contract_outlook.basis, "See the original explanation in the run export.")}
                  </p>}
                  {round.interval?.sequential_harness && round.contract_outlook?.evidence_quote && <details className="mt-1 text-ink-600">
                    <summary className="cursor-pointer">Public evidence</summary>
                    <blockquote className="ml-2 border-l border-paper-300 pl-2">{round.contract_outlook.evidence_quote}</blockquote>
                  </details>}
                  {round.outlook_issue && <p className="mt-1 text-ink-600">{outlookIssues[round.outlook_issue] || "Conditional estimate unavailable."}</p>}
                </li>
              ))}
            </ol>
            {!trajectory.rounds?.length && <p className="mt-1 text-sm text-ink-600">No timeline was saved for this trajectory.</p>}
            {!!trajectory.branches.length && <h4 className="mt-4 font-semibold">Final-round Branches</h4>}
            {!!trajectory.branches.length && <p className="mt-1 text-sm text-ink-600">Directly evaluated action weight: {
              percent(trajectory.branches.filter((row) => row.conditional_p_yes != null)
                .reduce((sum, row) => sum + row.joint_action_probability, 0))
            }. Remaining-weight samples represent combinations beyond the one evaluated.</p>}
            <div className="mt-2 divide-y divide-paper-200">
              {trajectory.branches.map((branch) => (
                <details key={branch.branch_id} className="py-2 text-sm">
                  <summary className="cursor-pointer font-medium text-brand-800">
                    {branch.kind === "tail_draw" ? "Remaining-weight sample" : `Branch ${branch.branch_id}`}
                    {" · "}{percent(branch.mass)} weight
                    {simulation.probability_method !== "conditional_terminal_v1" && <> · {percent(branch.conditional_p_yes)} conditional Yes</>}
                  </summary>
                  <div className="mt-2 grid gap-2 sm:grid-cols-3">
                    {[
                      { label: "Branch weight", value: branch.mass },
                      { label: "Conditional Yes", value: branch.conditional_p_yes },
                      { label: "Contribution to forecast", value: branch.conditional_p_yes == null ? null : branch.mass * branch.conditional_p_yes },
                    ].map(({ label, value }) => <div key={label}>
                      <p>{label}: <strong>{percent(value)}</strong></p>
                      {value != null && <div className="mt-1 h-1.5 rounded bg-paper-200" aria-hidden="true">
                        <div className="h-full rounded bg-brand-600" style={{ width: percent(value) }} />
                      </div>}
                    </div>)}
                  </div>
                  <p className="mt-2">Replay verdict: {branch.answer === "unknown" ? "Undetermined" : branch.answer === "yes" ? "Yes" : "No"}</p>
                  {branch.kind === "tail_draw" && <p>Sampled action combination: {percent(branch.joint_action_probability)} original weight</p>}
                  {branch.reason && <p>{displayText(branch.reason, "See the exported replay evidence.")}</p>}
                  {branch.yes_path && <p>Yes path: {displayText(branch.yes_path, "See the exported forecast explanation.")}</p>}
                  {branch.no_path && <p>No path: {displayText(branch.no_path, "See the exported forecast explanation.")}</p>}
                </details>
              ))}
            </div>
          </>}
        </section>
      )}
      {tab === "Diagnostics" && <section className={panel} aria-label="Run diagnostics">
        {showForecastCoverage && <p className="text-sm text-ink-600">Coverage measures returned forecasts or forecast-weight coverage. It does not measure forecast accuracy.</p>}
        {run.settings && <details className="mt-3 text-sm">
          <summary className="cursor-pointer font-medium text-brand-800">Saved Run Configuration</summary>
          <dl className="mt-2 grid gap-2 sm:grid-cols-2">
            {Object.entries(run.settings).map(([key, value]) => <div key={key} className="flex justify-between gap-2 border-b border-paper-200 pb-1">
              <dt className="capitalize">{key.replace(/_/g, " ")}</dt><dd>{key === "steps" && value == null ? "Adaptive" : String(value).replace(/_/g, " ")}</dd>
            </div>)}
          </dl>
        </details>}
        {!!roundPlans.length && <details className="mt-3 text-sm">
          <summary className="cursor-pointer font-medium text-brand-800">Planned Round Boundaries</summary>
          <dl className="mt-2 space-y-2">
            {roundPlans.map(([method, plan]) => <div key={method}>
              <dt className="font-medium">{label(method)} · {plan.steps} {plan.steps === 1 ? "round" : "rounds"}</dt>
              <dd className="break-words text-ink-600">{plan.boundaries.join(" → ")}</dd>
              {plan.milestones.length > 0 && <dd className="text-ink-600">Scheduled milestones · {plan.milestones.join(", ")}</dd>}
            </div>)}
          </dl>
        </details>}
        <div className="mt-3 divide-y divide-paper-200">
          {results.map((result) => <div key={result.method} className="py-3">
            <h4 className="mb-1 font-semibold">{label(result.method)}</h4>
            {result.simulation_route && <p className="mb-2 text-sm text-ink-700">{routeLabels[result.simulation_route]}</p>}
            {diagnostics(result)}
            {result.readiness_counts && <p className="mt-2 text-sm text-ink-600">
              Actor readiness · {result.readiness_counts.ready || 0} ready · {result.readiness_counts.restricted || 0} restricted · {result.readiness_counts.needs_evidence || 0} need evidence · {result.readiness_counts.retryable_error || 0} profile errors
            </p>}
            {!!result.decision_coverage?.length && <ul className="mt-2 text-sm text-ink-600">
              {result.decision_coverage.map((item) => <li key={`${item.kind}-${item.label}`}>
                {item.label} · {item.ready ? "Available" : "Missing"}
              </li>)}
            </ul>}
            {result.assumption && <p className="mt-2 text-sm text-ink-600">{result.assumption}</p>}
            {methodError(result) && <p className="mt-2 text-sm text-ink-700">{methodState(result)} · {methodError(result)}</p>}
            {result.probability_method === "sequential_sampled_outlook_v2" && <p className="mt-2 text-sm text-ink-600">
              {result.unknown_samples || 0} unestimated · {result.invalid_trace_samples || 0} invalid action traces · {result.failed_samples || 0} failed · {result.interrupted_samples || 0} interrupted · {result.unfinished_samples || 0} unfinished
              {result.p_yes == null && result.covered_p_yes != null && <> · Partial mean {percent(result.covered_p_yes)}
                {result.unknown_bounds && <> · Range with missing trajectories {percent(result.unknown_bounds[0])}–{percent(result.unknown_bounds[1])}</>}</>}
            </p>}
          </div>)}
        </div>
      </section>}
    </div>
  );
}
