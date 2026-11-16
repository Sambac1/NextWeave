import type { ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import type { ForecastReportData } from "@/services/forecastTaskService";
import TrajectoryReplay from "./TrajectoryReplay";
import { formatReportDate, normalizeReportText, similarReportText } from "./forecastReportPresentation";

const percent = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });
const format = (value: number) => percent.format(value);
const section = "space-y-3 border-t border-paper-300 px-5 py-6 sm:px-8 sm:py-7";
const heading = "text-xl font-semibold leading-tight tracking-tight text-ink-900";
const link = "font-medium text-brand-800 underline decoration-brand-300 underline-offset-4 hover:decoration-brand-800";

function summarizeTrajectories(report: ForecastReportData) {
  const deadline = Date.parse(report.outcome_deadline);
  const recorded = report.trajectories.filter(row => row.rounds.length > 0);
  const finished = recorded.filter(row => {
    const end = row.rounds.at(-1)?.interval?.interval_end_utc;
    return end ? Date.parse(end) === deadline : row.status === "completed";
  }).length;
  const awaiting = recorded.filter(row =>
    row.rounds.some(round => round.sampled_choices?.some(choice => choice.execution_status === "awaiting_prerequisite")),
  ).length;
  const developments = new Map<string, { text: string; trajectories: Set<number>; decisive: boolean }>();
  const driverRefs = new Set(report.drivers.map(driver => driver.reference));
  for (const row of recorded) {
    for (const round of row.rounds) {
      for (const event of round.events) {
        const key = normalizeReportText(event.text);
        if (!key) continue;
        const found = developments.get(key);
        const decisive = driverRefs.has(event.id);
        if (found) {
          found.trajectories.add(row.sample_id);
          found.decisive ||= decisive;
        } else {
          developments.set(key, { text: event.text, trajectories: new Set([row.sample_id]), decisive });
        }
      }
    }
  }
  const ranked = Array.from(developments.values()).sort((a, b) => Number(b.decisive) - Number(a.decisive) || b.trajectories.size - a.trajectories.size);
  const highlighted = ranked.slice(0, 1);
  const second = ranked.find(item => !similarReportText(item.text, highlighted[0]?.text ?? ""));
  if (second) highlighted.push(second);
  return { finished, awaiting, recorded, developments: highlighted };
}

function Disclosure({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="group/disclosure border-t border-paper-200">
      <summary className="flex min-h-11 list-none items-center justify-between gap-3 rounded-sm py-2.5 text-sm font-semibold text-ink-800 hover:text-brand-800 [&::-webkit-details-marker]:hidden">
        {label}
        <ChevronDown size={16} className="shrink-0 text-brand-700 motion-safe:transition-transform group-open/disclosure:rotate-180" aria-hidden="true" />
      </summary>
      <div className="pb-3 text-sm text-ink-600">{children}</div>
    </details>
  );
}

export default function ForecastReport({ report, standalone = false }: { report: ForecastReportData; standalone?: boolean }) {
  const Question = standalone ? "h1" : "h3";
  const SectionTitle = standalone ? "h2" : "h4";
  const { finished, awaiting, recorded, developments } = summarizeTrajectories(report);
  const { bounds, p_yes: estimate, probability_mass: mass } = report;
  const provisional = estimate == null ? report.conditional_estimate : null;
  const hasRange = bounds != null && bounds[0] < bounds[1] && (bounds[0] > 0 || bounds[1] < 1);
  const headline = estimate != null ? format(estimate) : provisional != null ? format(provisional)
    : hasRange ? `${format(bounds[0])}–${format(bounds[1])}` : "Forecast incomplete";
  const unresolved = mass?.unresolved ?? 0;
  const detailsAvailable = Boolean(report.warnings?.length || report.scheduled_milestones?.length);
  return (
    <article aria-label="Forecast report" className="overflow-hidden rounded-2xl border border-paper-300 bg-white shadow-panel">
      <header className="border-t-4 border-brand-600 bg-brand-50 px-5 py-6 sm:px-8 sm:py-8">
        <p className="text-sm font-semibold text-brand-800">{provisional != null ? "Provisional Yes estimate · completed trajectories"
          : estimate == null && hasRange ? "Estimated range for Yes" : "Estimated probability of Yes"}</p>
        <p className="mt-1 break-words font-semibold leading-none tracking-[-0.055em] text-ink-900 [font-size:clamp(2.35rem,8vw,4.65rem)]">{headline}</p>
        <Question className="mt-4 max-w-2xl break-words text-lg font-medium leading-snug text-ink-700 sm:text-xl">{report.question}</Question>
      </header>

      <section className={section}>
        <SectionTitle className={heading}>Simulation summary</SectionTitle>
        <p className="text-sm text-ink-700">{recorded.length
          ? `${finished} of ${report.planned_trajectories} trajectories reached the forecast deadline.`
          : "No trajectory rounds were saved."}
          {awaiting > 0 && ` ${awaiting === 1 ? "One trajectory" : `${awaiting} trajectories`} recorded an action waiting on a prerequisite.`}
          {recorded.length > 0 && developments.length === 0 && " No public developments were recorded."}
        </p>
        {report.aggregation_method === "weighted_conditional_probabilities" && <p className="text-sm text-ink-700">
          The estimate combines each completed trajectory's final conditional probability using its sampling weight.
          A conditional estimate can be low even when the recorded outcome remains unverified.
        </p>}
        {developments.length > 0 && <ul className="space-y-2 text-sm text-ink-700">
          {developments.map(({ text, trajectories }) => <li key={text} className="break-words border-l-2 border-brand-300 pl-3">
            {recorded.length > 1 && <p className="text-xs font-medium text-ink-500">In {trajectories.size} of {report.planned_trajectories} trajectories</p>}
            {text}
          </li>)}
        </ul>}
      </section>

      <section className={section}>
        <SectionTitle className={heading}>Evidence and uncertainty</SectionTitle>
        <p className="text-sm text-ink-700">{unresolved > 0
          ? <>{format(unresolved)} of the sampled weight remains unresolved. {unresolved >= 1
            ? "The runs leave the full 0%–100% range possible."
            : "The range assigns that unresolved weight to No at the lower end and Yes at the upper end."}</>
          : "The report summarizes simulations using the evidence and rules fixed at the cutoff."}
        </p>
        {!mass && report.status !== "complete" && <p className="text-sm text-ink-700">
          {report.estimated_trajectories} of {report.planned_trajectories} trajectories produced an estimate.
          {report.conditional_estimate != null && <> The estimate among those trajectories was {format(report.conditional_estimate)}.</>}
        </p>}
        {!mass && report.status === "complete" && report.verified_outcomes < report.estimated_trajectories && <p className="text-sm text-ink-700">
          {report.verified_outcomes} of {report.estimated_trajectories} trajectories had a verified simulated outcome. The remaining trajectories used model estimates.
        </p>}
        {report.error && <p role="alert" className="border-l-2 border-signal-red/50 pl-3 text-sm text-ink-700">{report.error}</p>}
        <dl className="grid gap-3 border-t border-paper-200 pt-3 text-sm sm:grid-cols-2">
          <div><dt className="text-xs text-ink-500">Evidence cutoff</dt><dd className="font-medium text-ink-800"><time dateTime={report.as_of}>{formatReportDate(report.as_of)}</time></dd></div>
          <div><dt className="text-xs text-ink-500">Forecast deadline</dt><dd className="font-medium text-ink-800"><time dateTime={report.outcome_deadline}>{formatReportDate(report.outcome_deadline)}</time></dd></div>
        </dl>
        <div>
          {report.sources.length > 0 && <Disclosure label={`Wikipedia sources (${report.sources.length})`}>
            <ul className="space-y-2">
              {report.sources.map(source => <li key={source.url} className="flex flex-wrap gap-x-3">
                <a className={link} target="_blank" rel="noreferrer"
                  href={source.revision_id ? `${new URL(source.url).origin}/w/index.php?oldid=${source.revision_id}` : source.url}>{source.title}</a>
                {source.revision_at && <time className="text-xs" dateTime={source.revision_at}>{formatReportDate(source.revision_at)}</time>}
              </li>)}
            </ul>
          </Disclosure>}
          <Disclosure label="Outcome criteria">
            <dl className="space-y-2">
              <div><dt className="font-semibold text-ink-800">Yes</dt><dd>{report.criteria.yes}</dd></div>
              <div><dt className="font-semibold text-ink-800">No</dt><dd>{report.criteria.no}</dd></div>
            </dl>
            {report.contract_rules && <p className="mt-3 whitespace-pre-wrap break-words">{report.contract_rules}</p>}
            {report.market_url && <a href={report.market_url} target="_blank" rel="noreferrer" className={`mt-2 inline-block ${link}`}>Original contract</a>}
          </Disclosure>
          {detailsAvailable && <Disclosure label="Run details">
            {!!report.scheduled_milestones?.length && <ul className="mb-2 space-y-1">
              {report.scheduled_milestones.map(milestone => <li key={milestone.date}>
                <time dateTime={milestone.date}>{milestone.date}</time> · <a href={milestone.url} target="_blank" rel="noreferrer" className={link}>{milestone.quote}</a>
              </li>)}
            </ul>}
            {!!report.warnings?.length && <ul className="space-y-1">{report.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul>}
          </Disclosure>}
        </div>
      </section>

      {report.trajectories.length > 0 && <section aria-label="Trajectory replay" className={section}>
        <SectionTitle className={heading}>Trajectory replay</SectionTitle>
        <TrajectoryReplay rows={report.trajectories} plannedRounds={report.planned_rounds} />
      </section>}
    </article>
  );
}
