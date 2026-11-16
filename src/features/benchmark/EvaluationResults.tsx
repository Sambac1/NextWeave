import { useState } from "react";
import type { BenchmarkRoundPlan } from "@/services/benchmarkService";
import { comparisonMethods, getReference, isProbability, scoreEvaluation } from "./evaluationComparison";
import { displayText, percent } from "./RunReport";

export const evaluationMethods = {
  direct_qa: "Direct QA",
  qa_ensemble: "QA Ensemble",
  single_step_simulation: "Single-step Simulation",
  single_trajectory_simulation: "Single-trajectory Simulation",
  single_step_ensemble: "Single-step Ensemble",
  multi_trajectory_simulation: "NextWeave",
  mcts_simulation: "MCTS Simulation",
} as const;

export type EvaluationMethod = keyof typeof evaluationMethods;
type Bin = { lower: number; upper: number; count: number; mean_p_yes: number; observed_yes_rate: number };
type Pair = { delta_brier: number; confidence_interval_95: [number, number] | null; case_count: number };
type Score = { brier: number; ece: number; log_loss: number; coverage: number; fallback_count: number;
  mean_calls: number | null; mean_tokens: number | null; mean_elapsed_seconds: number | null; calibration_bins: Bin[] };
type PartialForecast = { valid_samples: number; planned_samples: number; covered_p_yes: number | null; unknown_bounds: [number, number] };

export type EvaluationMethodUsage = Record<"calls" | "tokens" | "elapsed_seconds" | "unknown_usage_requests", {
  total: number; count: number;
}>;

export interface EvaluationRun {
  run_id: string;
  model: string;
  mode: "historical" | "live";
  status: string;
  methods: EvaluationMethod[];
  quick_test?: boolean;
  market_url?: string;
  market_choices?: { slug: string; question: string; url: string }[];
  error?: string;
  current?: number;
  total?: number;
  active_stage?: string;
  selection?: { policy: string; target?: number; available?: number; collected: number; accepted: number; not_screened: number;
    excluded: { question: string; reason: string }[] };
  knowledge_cutoff?: { status: string; at?: string };
  method_usage?: Partial<Record<EvaluationMethod, EvaluationMethodUsage>>;
  questions?: { case_id: string; question: string; as_of: string; run_id?: string; status?: string;
    resolution?: string; observed_yes?: boolean; quote?: number; quote_at?: string; reason?: string;
    forecasts: Partial<Record<EvaluationMethod, number | null>>;
    partial_forecasts?: Partial<Record<EvaluationMethod, PartialForecast>>;
    round_plans?: Partial<Record<EvaluationMethod, BenchmarkRoundPlan>> }[];
  scores?: { case_count: number; event_count: number; reference_method: EvaluationMethod;
    metrics: Partial<Record<EvaluationMethod, Score>>; paired: Partial<Record<EvaluationMethod, Pair>>;
    common_completed?: { case_count: number; metrics: Partial<Record<EvaluationMethod, Pick<Score, "brier" | "log_loss">>> };
    market_comparison?: { case_count: number; brier: number; paired: Partial<Record<EvaluationMethod, Pair>> } | null } | null;
}

const fixed = (value?: number | null) => value == null ? "—" : value.toFixed(3);

export default function EvaluationResults({ run, openQuestion }: {
  run: EvaluationRun; openQuestion: (id: string) => void;
}) {
  const methods: EvaluationMethod[] = comparisonMethods.map(([id]) => id).filter((id) => run.methods.includes(id));
  const [selected, setSelected] = useState<EvaluationMethod | undefined>(methods[0]);
  const rows = run.questions || [];
  const scores = run.scores;
  const bins = selected ? scores?.metrics[selected]?.calibration_bins || [] : [];
  const ranked = methods.flatMap((id) => {
    const score = scores?.metrics[id];
    return score ? [{ id, score }] : [];
  }).sort((a, b) => a.score.brier - b.score.brier);
  const snapshots = run.mode === "live" ? rows.flatMap((question) => {
    const reference = getReference(question, "live");
    return reference === null ? [] : [{ question, reference }];
  }) : [];
  const showSnapshots = !["Collecting", "Running", "Pausing", "Paused"].includes(run.status)
    && snapshots.some(({ question }) => methods.some((method) => isProbability(question.forecasts[method])));
  const pending = rows.filter((row) => row.resolution === "Awaiting Resolution").length;
  const resolved = rows.filter((row) => row.resolution === "Resolved").length;
  const name = (id: string) => evaluationMethods[id as EvaluationMethod] || id;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-3 text-sm text-ink-700">
        <span><strong>{rows.length}</strong> questions</span>
        <span><strong>{resolved}</strong> resolved</span>
        {run.mode === "live" && <span><strong>{pending}</strong> awaiting resolution</span>}
        {scores && <span><strong>{scores.event_count}</strong> event groups</span>}
      </div>
      {run.knowledge_cutoff?.status === "Knowledge Cutoff Unverified" && (
        <p className="rounded bg-paper-100 p-3 text-sm">Knowledge Cutoff Unverified. This cohort cannot support a leakage-controlled accuracy claim.</p>
      )}
      {showSnapshots && (
        <section aria-label="Market snapshot comparison" className="space-y-3 rounded-md border border-paper-300 bg-white p-4">
          <h4 className="font-semibold text-ink-900">Market Snapshot Comparison</h4>
          <p className="text-sm text-ink-600">{snapshots.length} questions with a saved market quote. Missing forecasts use a 50% fallback. Lower error is better.</p>
          <table className="w-full text-left text-sm">
            <thead className="text-ink-600">
              <tr><th scope="col">Method</th><th scope="col" className="text-right">MAE (pp)</th><th scope="col" className="text-right">RMSE (pp)</th></tr>
            </thead>
            <tbody>
              {methods.map((id) => {
                const { primary, secondary } = scoreEvaluation(snapshots, id, "live");
                return <tr key={id} className="border-t border-paper-200">
                  <th scope="row" className="py-2 font-medium">{name(id)}</th>
                  <td className="px-2 text-right">{primary?.toFixed(2) ?? "—"}</td>
                  <td className="text-right">{secondary?.toFixed(2) ?? "—"}</td>
                </tr>;
              })}
            </tbody>
          </table>
        </section>
      )}
      {scores && ranked.length > 0 && (
        <section aria-label="Benchmark scores" className="space-y-3 rounded-md border border-paper-300 bg-white p-4">
          <h4 className="font-semibold text-ink-900">Mean Brier Score <span className="text-sm font-normal text-ink-600">· Lower is better</span></h4>
          <p className="text-sm text-ink-600">{scores.case_count} resolved contract–cutoff pairs. Missing or invalid forecasts use a 50% fallback.</p>
          {ranked.map(({ id, score: row }) => (
            <div key={id} className="text-sm">
              <div className="flex justify-between gap-2"><strong>{name(id)}</strong><span>{fixed(row.brier)}</span></div>
              <div aria-hidden="true" className="my-1 h-1.5 rounded bg-paper-200"><div className="h-full rounded bg-brand-600" style={{width: percent(row.brier)}} /></div>
              <p className="text-ink-600">Log Loss {fixed(row.log_loss)} · ECE {fixed(row.ece)} · {row.fallback_count} fallbacks</p>
              <p className="text-ink-600">Mean cost: {fixed(row.mean_calls)} requests · {fixed(row.mean_tokens)} tokens · {fixed(row.mean_elapsed_seconds)} s</p>
            </div>
          ))}
          {scores.case_count < 50 && <p className="text-sm text-ink-600">Calibration is provisional for this cohort.</p>}
          <details className="border-t border-paper-200 pt-2 text-sm">
            <summary className="cursor-pointer font-medium text-brand-800">Calibration and Forecast Distribution</summary>
            <label className="mt-2 block">Method <select value={selected} onChange={(event) => setSelected(event.target.value as EvaluationMethod)}
              className="rounded border border-paper-300 px-2 py-1">
              {methods.map((id) => <option value={id} key={id}>{name(id)}</option>)}
            </select></label>
            <svg viewBox="0 0 260 160" role="img" aria-label="Calibration chart: predicted probability versus observed frequency" className="w-full max-w-xs">
              <path d="M24 130H244 M24 130V10 M24 130L244 10" fill="none" stroke="currentColor" opacity=".5" />
              {bins.map((bin) => <circle key={bin.lower} cx={24 + 220 * bin.mean_p_yes} cy={130 - 120 * bin.observed_yes_rate}
                r="4" fill="currentColor" className="text-brand-700"><title>{bin.count} forecasts</title></circle>)}
              <text x="22" y="146" fontSize="10">0%</text><text x="219" y="146" fontSize="10">100%</text><text x="0" y="12" fontSize="9">100%</text>
            </svg>
            {bins.map((bin) => <p key={bin.lower}>{percent(bin.lower)}–{percent(bin.upper)} · {bin.count} forecasts · {percent(bin.mean_p_yes)} predicted · {percent(bin.observed_yes_rate)} observed</p>)}
          </details>
          <details className="border-t border-paper-200 pt-2 text-sm">
            <summary className="cursor-pointer font-medium text-brand-800">Paired Brier Difference</summary>
            {[
              ...Object.entries(scores.paired).map(([id, pair]) => ({ id, pair, baseline: name(scores.reference_method) })),
              ...Object.entries(scores.market_comparison?.paired || {}).map(([id, pair]) => ({ id, pair, baseline: "Market at Cutoff" })),
            ].filter(({ id }) => methods.includes(id as EvaluationMethod)).map(({ id, pair, baseline }) => <p className="mt-1" key={`${id}:${baseline}`}>
              {name(id)} vs. {baseline}: {fixed(pair.delta_brier)}
              {pair.confidence_interval_95 ? ` · 95% CI [${pair.confidence_interval_95.map(fixed).join(", ")}]` : " · interval unavailable"}
              {" · "}{pair.case_count} matched cases
            </p>)}
            {!scores.market_comparison && <p className="mt-1">Market Baseline Unavailable.</p>}
          </details>
        </section>
      )}
      {!!rows.length && <section aria-label="Questions">
        <h4 className="mb-2 font-semibold text-ink-900">Questions</h4>
        <ul className="divide-y divide-paper-200 border-y border-paper-300">
          {rows.map((row) => <li key={row.case_id} className="py-3 text-sm">
            <p className="font-medium text-ink-900">
              {row.run_id ? <button type="button" onClick={() => openQuestion(row.run_id!)}
                className="text-left text-brand-800 underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-700">{displayText(row.question, "Contract question")}</button> : displayText(row.question, "Contract question")}
            </p>
            <p className="text-ink-600">{row.resolution || "Pending"}
              {row.status && <> · Forecast run: {row.status}</>}
              {row.observed_yes != null && <> · Official outcome: {row.observed_yes ? "Yes" : "No"}</>}
              {row.quote != null && <> · Market at cutoff: {percent(row.quote)} · {row.quote_at}</>}
              {row.reason && <> · {row.reason}</>}
            </p>
            {methods.some((id) => id in row.forecasts) && <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-ink-700">
              {methods.filter((id) => id in row.forecasts).map((id) => {
                const value = row.forecasts[id];
                const plan = row.round_plans?.[id];
                const partial = row.partial_forecasts?.[id];
                return <li key={id}>{name(id)} · {value == null ? "Forecast unavailable" : percent(value)}
                  {plan && plan.mode !== "single_step" && <span className="text-ink-600"> · {plan.steps} {plan.mode} {plan.steps === 1 ? "round" : "rounds"}</span>}
                  {partial && <span className="block text-xs text-ink-600">
                    {partial.valid_samples}/{partial.planned_samples} trajectories with estimates
                    {partial.covered_p_yes != null && <> · Partial mean {percent(partial.covered_p_yes)} · Range with missing trajectories {percent(partial.unknown_bounds[0])}–{percent(partial.unknown_bounds[1])}</>}
                  </span>}
                </li>;
              })}
            </ul>}
          </li>)}
        </ul>
      </section>}
      {run.selection && <details className="border-t border-paper-300 pt-2 text-sm">
        <summary className="cursor-pointer font-medium text-brand-800">
          Cohort selection · {run.selection.accepted}{run.mode === "historical" && run.selection.target ? ` / ${run.selection.target}` : ""} selected
          {run.selection.available != null && <> · {run.selection.available} verified in reserve</>}
          {" · "}{run.selection.excluded.length} excluded
        </summary>
        <p className="mt-2">{run.selection.policy} · {run.selection.collected} collected · {run.selection.not_screened} not screened</p>
        <ul className="mt-2 max-h-64 space-y-1 overflow-y-auto">
          {run.selection.excluded.map((row, index) => <li key={index}>{displayText(row.question, "Contract question")} · {row.reason}</li>)}
        </ul>
      </details>}
    </div>
  );
}
