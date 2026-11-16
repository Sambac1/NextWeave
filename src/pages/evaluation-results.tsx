import { useEffect, useState } from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { ArrowLeft, ChevronDown } from "lucide-react";
import { readBackendUrl } from "@/config/backend";
import { marketComparisonPath as resultsPath, marketEvaluationPath } from "@/config/evaluationRoutes";
import EvaluationComputeCost from "@/features/benchmark/EvaluationComputeCost";
import type { EvaluationRun } from "@/features/benchmark/EvaluationResults";
import { comparisonMethods as methods, comparisonRunLabels as runMethodLabels, getReference, isProbability, scoreEvaluation } from "@/features/benchmark/evaluationComparison";
import RunReport, { displayText } from "@/features/benchmark/RunReport";
import type { BenchmarkRun } from "@/services/benchmarkService";
import { apiRequest } from "@/services/experimentService";


const cohorts = [
  {
    mode: "historical",
    id: "5d150920-e548-4769-87ab-9905d3f86071",
    title: "Historical Evaluation",
    reference: "Official settlement",
    metrics: ["Brier score", "Log loss"],
  },
  {
    mode: "live",
    id: "9f5b61d8-5f32-4ed5-8fb6-fb800a64eebe",
    title: "Live Market Evaluation",
    reference: "Market snapshot",
    metrics: ["MAE (pp)", "RMSE (pp)"],
  },
] as const;

type Cohort = (typeof cohorts)[number];
type Question = NonNullable<EvaluationRun["questions"]>[number];
type Report = { run: EvaluationRun } | { error: string };

const snapshotFormat = new Intl.DateTimeFormat("en-GB", {
  dateStyle: "medium", timeStyle: "medium", timeZone: "UTC",
});
const formatProbability = (value: unknown) => isProbability(value) ? `${(value * 100).toFixed(1)}%` : "—";
const nextWeave = "multi_trajectory_simulation";
const focus = "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-700";

function CohortPanel({ cohort, report, retry }: {
  cohort: Cohort; report?: Report; retry: () => void;
}) {
  const [search, setSearch] = useState("");
  const run = report && "run" in report ? report.run : undefined;
  const questions = run?.questions ?? [];
  const eligible = questions.flatMap((question) => {
    const reference = getReference(question, cohort.mode);
    return reference === null ? [] : [{ question, reference }];
  });
  const query = search.trim().toLowerCase();
  const visibleQuestions = questions.filter((question) => question.question.toLowerCase().includes(query));
  const rows = methods.map(([method, name]) => ({ method, name, ...scoreEvaluation(eligible, method, cohort.mode) }));
  const digits = cohort.mode === "historical" ? 4 : 2;
  const bestPrimary = Math.min(...rows.flatMap(({ primary }) => primary === null ? [] : [primary]));
  const bestSecondary = Math.min(...rows.flatMap(({ secondary }) => secondary === null ? [] : [secondary]));
  const snapshotTimes = new Set(questions.map((question) => question.quote_at));
  const snapshotTime = snapshotTimes.size === 1 ? questions[0]?.quote_at : undefined;
  const snapshot = snapshotTime ? new Date(snapshotTime) : null;
  const scoredCount = eligible.length;

  return (
    <section className="min-w-0 self-start overflow-hidden rounded-xl border border-paper-300 bg-white shadow-panel" aria-labelledby={`${cohort.mode}-title`}>
      <header className="border-b border-paper-300 px-4 py-5 sm:px-6">
        <p className="text-xs font-semibold uppercase tracking-wider text-brand-800">{cohort.reference}</p>
        <h2 id={`${cohort.mode}-title`} className="mt-1 text-2xl font-semibold tracking-tight text-ink-900">{cohort.title}</h2>
        {run && (
          <p className="mt-1 text-xs text-ink-400">
            <span>{scoredCount} of {questions.length} questions scored</span>
            {snapshot && Number.isFinite(snapshot.getTime()) && (
              <> · <time dateTime={snapshot.toISOString()}>{snapshotFormat.format(snapshot)} UTC</time></>
            )}
          </p>
        )}
      </header>
      {!report && <p className="px-4 py-6 text-sm text-ink-400 sm:px-6" role="status">Loading saved results…</p>}
      {report && "error" in report && (
        <div className="px-4 py-6 sm:px-6">
          <p className="text-sm text-ink-600" role="alert">{report.error}</p>
          <button className={`mt-2 min-h-11 text-sm font-semibold text-brand-800 underline underline-offset-4 ${focus}`} type="button" onClick={retry}>
            Try again
          </button>
        </div>
      )}
      {run && (
        <>
          <table className="w-full border-collapse text-left text-xs sm:text-sm">
            <caption className="sr-only">{cohort.title} scores. Lower errors are better.</caption>
            <thead className="bg-paper-50 text-ink-400">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium sm:pl-6">Method</th>
                {cohort.metrics.map((metric) => <th key={metric} scope="col" className="px-2 py-3 text-right text-xs font-medium last:pr-4 sm:last:pr-6">{metric}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map(({ method, name, primary, secondary }) => (
                <tr key={method} className={`border-t border-paper-200 ${method === nextWeave ? "bg-brand-50" : ""}`}>
                  <th scope="row" className={`px-4 py-3.5 leading-snug sm:pl-6 ${method === nextWeave ? "font-semibold text-brand-800" : "font-medium text-ink-800"}`}>{name}</th>
                  <td className={`px-1.5 py-3.5 text-right tabular-nums sm:px-2 ${primary === bestPrimary ? "font-semibold text-brand-800" : "text-ink-800"}`}>{primary?.toFixed(digits) ?? "—"}</td>
                  <td className={`py-3.5 pl-2 pr-4 text-right tabular-nums sm:pr-6 ${secondary === bestSecondary ? "font-semibold text-brand-800" : "text-ink-800"}`}>{secondary?.toFixed(digits) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <details className="group border-t border-paper-300">
            <summary className={`flex min-h-12 list-none items-center justify-between gap-3 px-4 py-3 text-sm font-semibold text-brand-800 hover:bg-paper-50 focus-visible:-outline-offset-2 sm:px-6 ${focus}`}>
              Browse question reports
              <ChevronDown aria-hidden="true" className="h-4 w-4 shrink-0 group-open:rotate-180" />
            </summary>
            <label className="block border-t border-paper-200 px-4 py-3 text-xs font-medium text-ink-500 sm:px-6">
              Search questions
              <input type="search" value={search} onChange={(event) => setSearch(event.target.value)}
                className={`mt-1 block min-h-11 w-full rounded-md border border-paper-300 px-3 text-sm text-ink-800 ${focus}`}
                placeholder="Filter by question text" />
            </label>
            {visibleQuestions.length ? (
              <div tabIndex={0} role="region" aria-label={`${cohort.title} question forecasts`} className={`max-h-[28rem] overflow-auto border-t border-paper-200 focus-visible:-outline-offset-2 ${focus}`}>
                <table className="w-full min-w-[44rem] text-left text-xs">
                  <caption className="sr-only">Forecast probability of Yes by question</caption>
                  <thead className="sticky top-0 bg-paper-50 text-ink-500">
                    <tr>
                      <th scope="col" className="w-5/12 px-4 py-2 font-medium sm:pl-6">Question</th>
                      <th scope="col" className="px-2 py-2 text-right font-medium">{cohort.mode === "historical" ? "Outcome" : "Snapshot"}</th>
                      {methods.map(([id, name]) => <th key={id} scope="col" className="px-2 py-2 text-right font-medium last:pr-4">{name}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {visibleQuestions.map((question) => {
                      const reference = getReference(question, cohort.mode);
                      return (
                        <tr key={question.case_id} className="border-t border-paper-200">
                          <th scope="row" className="px-4 py-2 font-medium text-ink-800 sm:pl-6">
                            {question.run_id ? (
                              <Link href={{ pathname: resultsPath, query: { questionRun: question.run_id } }} shallow scroll prefetch={false}
                                className={`inline-block rounded-sm py-1 font-semibold text-brand-800 underline decoration-brand-300 underline-offset-2 hover:decoration-brand-800 ${focus}`}>
                                {question.question}
                              </Link>
                            ) : question.question}
                          </th>
                          <td className="px-2 py-2 text-right">{reference === null ? "—" : cohort.mode === "historical"
                            ? (reference === 1 ? "Yes" : "No")
                            : formatProbability(reference)}</td>
                          {methods.map(([id]) => <td key={id} className={`px-2 py-2 text-right last:pr-4 ${id === nextWeave ? "bg-brand-50 font-semibold text-brand-800" : ""}`}>{formatProbability(question.forecasts[id])}</td>)}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : <p className="px-4 py-3 text-xs text-ink-400 sm:px-6">No questions match your search.</p>}
            <p className="px-4 py-3 text-xs text-ink-400 sm:px-6">Select a question to open its saved run. Forecasts show the probability of Yes.</p>
          </details>
        </>
      )}
    </section>
  );
}

function QuestionRunPanel({ cohort, question }: { cohort: Cohort; question: Question }) {
  const [result, setResult] = useState<BenchmarkRun | "error">();
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    const { run_id, case_id } = question;
    const api = readBackendUrl();
    if (!api || !run_id) {
      setResult("error");
      return;
    }
    apiRequest(api, `/benchmark/comparison/runs/${run_id}`).then((run: BenchmarkRun) => {
      if (active) setResult(run.run_id === run_id && run.case_id === case_id ? run : "error");
    }).catch(() => {
      if (active) setResult("error");
    });
    return () => { active = false; };
  }, [question, attempt]);

  const reference = getReference(question, cohort.mode);
  const run = result && result !== "error" ? result : undefined;
  const detail = run && cohort.mode === "live" && reference !== null && question.quote_at ? {
    ...run, market_reference: { probability: reference, label: "Market snapshot", observed_on: question.quote_at },
  } : run;

  return (
    <section aria-labelledby="question-title" className="py-8">
      <header className="rounded-xl border border-paper-300 bg-white p-5 sm:p-6">
        <p className="text-xs font-semibold uppercase tracking-wider text-brand-800">{cohort.title}</p>
        <h1 id="question-title" className="mt-2 text-2xl font-semibold leading-tight tracking-tight text-ink-900 sm:text-3xl">{question.question}</h1>
        <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-ink-600">
          <div><dt className="inline font-semibold">Forecast cutoff</dt>{" · "}<dd className="inline"><time dateTime={question.as_of}>{question.as_of}</time></dd></div>
          {reference !== null && (
            <div><dt className="inline font-semibold">{cohort.reference}</dt>{" · "}<dd className="inline">{cohort.mode === "live"
              ? formatProbability(reference) : (reference === 1 ? "Yes" : "No")}</dd></div>
          )}
          {run && <div><dt className="inline font-semibold">Run status</dt>{" · "}<dd className="inline capitalize">{run.status}</dd></div>}
        </dl>
        <p className="mt-2 break-all font-mono text-xs text-ink-500">Run ID · {question.run_id}</p>
      </header>
      {!result && <p className="py-4 text-sm text-ink-500" role="status">Loading the saved question run…</p>}
      {result === "error" && (
        <div className="py-4">
          <p className="text-sm text-ink-600" role="alert">This saved question run could not be loaded.</p>
          <button type="button" className={`mt-2 min-h-11 text-sm font-semibold text-brand-800 underline underline-offset-4 ${focus}`}
            onClick={() => { setResult(undefined); setAttempt((value) => value + 1); }}>Try again</button>
        </div>
      )}
      {detail && (
        <>
          {detail.error && <p role="alert" className="mt-4 text-sm text-signal-red">{displayText(detail.error, "This run stopped. See the saved diagnostics.")}</p>}
          {detail.results?.some(({ method }) => Object.prototype.hasOwnProperty.call(runMethodLabels, method))
            ? <RunReport run={detail} methodLabels={runMethodLabels} showForecastCoverage={false} />
            : <p className="py-4 text-sm text-ink-500">No method results were saved for this run.</p>}
        </>
      )}
    </section>
  );
}

export default function EvaluationResultsPage() {
  const router = useRouter();
  const questionRun = typeof router.query.questionRun === "string" ? router.query.questionRun : undefined;
  const [attempt, setAttempt] = useState(0);
  const [reports, setReports] = useState<Partial<Record<Cohort["mode"], Report>>>({});
  const questionMatch = questionRun ? cohorts.flatMap((cohort) => {
    const report = reports[cohort.mode];
    const question = report && "run" in report ? report.run.questions?.find((row) => row.run_id === questionRun) : undefined;
    return question ? [{ cohort, question }] : [];
  })[0] : undefined;
  const loading = cohorts.some((cohort) => !reports[cohort.mode]);
  const reportError = cohorts.some((cohort) => {
    const report = reports[cohort.mode];
    return report && "error" in report;
  });

  useEffect(() => {
    let active = true;
    const api = readBackendUrl();
    for (const cohort of cohorts) {
      const load = async () => {
        let report: Report;
        try {
          if (!api) {
            report = { error: "Connect a backend in NextWeave settings to load this saved evaluation." };
          } else {
            const run: EvaluationRun = await apiRequest(api, `/benchmark/comparison/evaluations/${cohort.id}`);
            if (run.run_id !== cohort.id || run.mode !== cohort.mode || !run.questions?.length) {
              throw new Error("Invalid saved evaluation");
            }
            report = { run };
          }
        } catch {
          report = { error: "This saved evaluation could not be loaded. Check the backend connection and the comparison archive." };
        }
        if (active) setReports((current) => ({ ...current, [cohort.mode]: report }));
      };
      void load();
    }
    return () => { active = false; };
  }, [attempt]);

  function retry() {
    setReports({});
    setAttempt((value) => value + 1);
  }

  return (
    <>
      <Head>
        <title>Evaluation Results · NextWeave</title>
        <meta name="description" content="Compare five forecasting methods on two saved sets of 30 questions." />
      </Head>
      <main lang="en" className="min-h-screen bg-paper-100 px-4 pb-12 text-ink-800 sm:px-8">
        <div className="mx-auto max-w-6xl">
          <nav className="border-b border-paper-300 py-3" aria-label="NextWeave">
            <Link href={questionRun ? resultsPath : marketEvaluationPath} shallow={!!questionRun} scroll className={`inline-flex min-h-11 items-center gap-2 rounded text-sm font-semibold text-brand-800 hover:underline ${focus}`}>
              <ArrowLeft className="h-4 w-4" aria-hidden="true" />
              {questionRun ? "Back to evaluation results" : "Back to Market Evaluation"}
            </Link>
          </nav>
          <div hidden={!!questionRun}>
            <header className="pb-6 pt-8 sm:pt-10">
              <h1 className="text-4xl font-semibold leading-tight tracking-tight text-ink-900 sm:text-5xl">Evaluation Results</h1>
              <p className="mt-2 text-sm text-ink-500 sm:text-base">Five forecasting methods across 30 historical questions and 30 live market questions.</p>
            </header>
            <p className="mb-4 text-xs text-ink-500">Lower scores are better. pp = percentage points.</p>
            <div className="grid gap-5 lg:grid-cols-2">
              {cohorts.map((cohort) => <CohortPanel key={cohort.mode} cohort={cohort} report={reports[cohort.mode]} retry={retry} />)}
            </div>
            <p className="mt-4 text-xs text-ink-500">Missing forecasts are scored at 50% Yes. Live scores measure distance to the saved market snapshot.</p>
            <EvaluationComputeCost groups={cohorts.map((cohort) => {
              const report = reports[cohort.mode];
              return { title: cohort.title, run: report && "run" in report ? report.run : undefined, loading: !report };
            })} />
          </div>
          {questionRun && (questionMatch
            ? <QuestionRunPanel key={questionRun} {...questionMatch} />
            : loading ? <p className="py-6 text-sm text-ink-500" role="status">Finding the saved question…</p>
            : <div className="py-6 text-sm text-ink-600">
              <p role="alert">{reportError ? "Could not check every saved evaluation for this question." : "This question run was not found in either evaluation."}</p>
              {reportError && <button type="button" className={`mt-2 min-h-11 font-semibold text-brand-800 underline ${focus}`} onClick={retry}>Try again</button>}
            </div>)}
        </div>
      </main>
    </>
  );
}
