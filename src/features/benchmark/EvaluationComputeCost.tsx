import type { EvaluationMethodUsage, EvaluationRun } from "./EvaluationResults";

const methods = [
  ["mcts_simulation", "MCTS Simulation"],
  ["multi_trajectory_simulation", "NextWeave"],
] as const;

const metrics = [
  ["calls", "Model requests", "fewer requests"],
  ["tokens", "Tokens", "fewer tokens"],
  ["elapsed_seconds", "Runtime (s)", "shorter runtime"],
] as const;

type Method = (typeof methods)[number][0];
type Group = { title: string; run?: EvaluationRun; loading: boolean };

const decimal = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const percentage = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });

function usageTotals(groups: Group[], method: Method, metric: keyof EvaluationMethodUsage) {
  let total = 0;
  let count = 0;
  for (const { run } of groups) {
    const value = run?.method_usage?.[method]?.[metric];
    if (!value) return null;
    total += value.total;
    count += value.count;
  }
  return { total, count };
}

function average(groups: Group[], method: Method, metric: keyof EvaluationMethodUsage) {
  const value = usageTotals(groups, method, metric);
  return value?.count ? value.total / value.count : null;
}

export default function EvaluationComputeCost({ groups }: { groups: Group[] }) {
  const rows = [
    ...groups.map((group) => ({ title: group.title, groups: [group] })),
    { title: "Combined", groups },
  ];
  const changes = metrics.flatMap(([metric, , label]) => {
    const mcts = average(groups, "mcts_simulation", metric);
    const nextWeave = average(groups, "multi_trajectory_simulation", metric);
    return mcts && nextWeave != null && nextWeave < mcts ? [`${percentage.format(1 - nextWeave / mcts)} ${label}`] : [];
  });
  const missingUsage = methods.map(([method, name]) => {
    const missing = usageTotals(groups, method, "unknown_usage_requests");
    const requests = usageTotals(groups, method, "calls");
    return missing && requests && missing.count === requests.count && requests.count
      ? { name, count: missing.total }
      : null;
  });
  const hasUsage = groups.some(({ run }) => run?.method_usage);
  const loading = groups.some((group) => group.loading);

  return (
    <section aria-labelledby="compute-cost-title" className="mt-6 overflow-hidden rounded-xl border border-paper-300 bg-white shadow-panel">
      <header className="border-b border-paper-300 px-4 py-5 sm:px-6">
        <h2 id="compute-cost-title" className="text-2xl font-semibold tracking-tight text-ink-900">Average compute cost per question</h2>
        <p className="mt-1 text-sm text-ink-500">Model requests, reported input and output tokens, and saved runtime.</p>
      </header>
      {hasUsage ? (
        <>
          <div tabIndex={0} role="region" aria-label="Compute cost comparison" className="overflow-x-auto focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-brand-700">
            <table className="w-full min-w-[30rem] border-collapse text-left text-xs sm:text-sm">
              <caption className="sr-only">Mean compute cost per question for MCTS Simulation and NextWeave</caption>
              <thead className="bg-paper-50 text-ink-500">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium sm:pl-6">Evaluation</th>
                  <th scope="col" className="px-3 py-3 font-medium">Method</th>
                  {metrics.map(([metric, label]) => <th key={metric} scope="col" className="px-3 py-3 text-right font-medium last:pr-4 sm:last:pr-6">{label}</th>)}
                </tr>
              </thead>
              {rows.map(({ title, groups: rowGroups }) => {
                const questionCount = rowGroups.every(({ run }) => run?.questions)
                  ? rowGroups.reduce((sum, { run }) => sum + (run?.questions?.length ?? 0), 0)
                  : null;
                return (
                  <tbody key={title} className="border-t border-paper-300">
                    {methods.map(([method, name], index) => (
                      <tr key={method} className={index === 1 ? "border-t border-paper-200 bg-brand-50 text-brand-800" : "text-ink-800"}>
                        {index === 0 && (
                          <th scope="rowgroup" rowSpan={methods.length} className="bg-white px-4 py-3 align-top font-semibold sm:pl-6">
                            {title}
                            {questionCount !== null && <span className="block text-xs font-normal text-ink-500">{questionCount} questions</span>}
                          </th>
                        )}
                        <th scope="row" className="px-3 py-3 font-semibold">{name}</th>
                        {metrics.map(([metric]) => {
                          const value = average(rowGroups, method, metric);
                          return (
                            <td key={metric} className="px-3 py-3 text-right tabular-nums last:pr-4 sm:last:pr-6">
                              {value === null ? "—" : (metric === "tokens" ? integer : decimal).format(value)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                );
              })}
            </table>
          </div>
          <div className="space-y-2 border-t border-paper-300 px-4 py-4 text-xs text-ink-500 sm:px-6">
            {changes.length === metrics.length && (
              <p className="text-sm font-semibold text-brand-800">Across both evaluations, NextWeave had {changes[0]}, {changes[1]}, and {changes[2]} on average than MCTS Simulation.</p>
            )}
            <p>Averages include saved runs without a final forecast. Each average uses runs with a recorded value for that metric. Shared contract parsing and actor preparation are accounted for separately from method requests and tokens.</p>
            {missingUsage.every((entry) => entry !== null) && missingUsage.some((entry) => entry && entry.count > 0) && (
              <p>The provider did not report token usage for {missingUsage.map((entry) => `${entry?.count} ${entry?.name} requests`).join(" and ")}.</p>
            )}
          </div>
        </>
      ) : (
        <p role="status" className="px-4 py-5 text-sm text-ink-500 sm:px-6">
          {loading ? "Loading saved compute usage…" : "Compute usage is unavailable in the saved evaluation reports."}
        </p>
      )}
    </section>
  );
}
