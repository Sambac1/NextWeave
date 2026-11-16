import { useEffect, useRef, useState } from "react";
import { ArrowDown, ChevronLeft, ChevronRight, Check, GitBranch } from "lucide-react";
import type { ForecastReportData } from "@/services/forecastTaskService";
import { formatReportDate, normalizeReportText, similarReportText, verdictLabels } from "./forecastReportPresentation";

type Trajectory = ForecastReportData["trajectories"][number];
type Round = Trajectory["rounds"][number];
type Choice = NonNullable<Round["sampled_choices"]>[number];
type ReplayOption = { id: string; action: string; probability?: number | null; selected: boolean };

const percent = new Intl.NumberFormat("en-US", { style: "percent", maximumSignificantDigits: 3 });
const isProbability = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;

function probability(value: number | null | undefined) {
  if (!isProbability(value)) return "Unavailable";
  return value > 0 && value < 0.000001 ? "<0.0001%" : percent.format(value);
}

function resultLabel(row: Trajectory) {
  if (row.status === "failed" || row.status === "interrupted") return row.status;
  if (row.terminal_status === "estimated" && isProbability(row.p_yes)) return `${probability(row.p_yes)} estimated Yes`;
  if (row.terminal_status) return verdictLabels[row.terminal_status];
  if (isProbability(row.p_yes)) return `${probability(row.p_yes)} estimated Yes`;
  return row.status.replaceAll("_", " ");
}

function ActorChoices({ choice }: { choice: Choice }) {
  const publicOptions = choice.public_options ?? [];
  const sampled = publicOptions.find(option => option.selected || option.candidate_id === choice.candidate_id);
  const options: ReplayOption[] = publicOptions.map(option => ({
    id: option.candidate_id,
    action: option.action,
    probability: option === sampled && isProbability(choice.p_selected) ? choice.p_selected : option.probability,
    selected: option === sampled,
  }));
  const sampledAction = choice.action?.trim() || (choice.waited ? "Wait or monitor" : "Action details unavailable");
  if (!sampled) {
    options.unshift({ id: "sampled", action: sampledAction, probability: choice.p_selected, selected: true });
  }
  const visibleMass = options.reduce((total, option) => total + (isProbability(option.probability) ? option.probability : 0), 0);
  const execution = choice.execution_status === "awaiting_prerequisite" ? "Awaiting a prerequisite"
    : choice.execution_status === "attempt" ? "Attempt recorded"
    : choice.waited ? "Wait or monitor" : null;

  return (
    <div className="min-w-0 space-y-2.5 border-t border-paper-200 pt-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <p className="min-w-0 break-words font-semibold text-ink-900">{choice.actor_name || choice.actor_id}</p>
        {execution && <span className="rounded-full bg-paper-100 px-2.5 py-0.5 text-xs font-medium text-ink-700">{execution}</span>}
      </div>
      <ul className="space-y-2">
        {options.map(option => (
          <li key={option.id} className={`space-y-2 rounded-lg border px-3 py-2.5 ${option.selected
            ? "border-brand-200 bg-brand-50" : "border-paper-200 bg-white"}`}>
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 space-y-0.5">
                {option.selected && <p className="flex items-center gap-1.5 text-xs font-semibold text-brand-800">
                  <Check size={14} className="shrink-0" aria-hidden="true" /> Sampled
                </p>}
                <p className="whitespace-pre-line break-words text-sm leading-snug text-ink-800">{option.action}</p>
              </div>
              <span className={`shrink-0 font-semibold tabular-nums ${option.selected ? "text-brand-800" : "text-ink-700"}`}>
                {probability(option.probability)}
              </span>
            </div>
            {isProbability(option.probability) && <div className="h-1.5 overflow-hidden rounded-full bg-paper-200" aria-hidden="true">
              <div className={`h-full rounded-full ${option.selected ? "bg-brand-600" : "bg-brand-300"}`}
                style={{ width: `${option.probability * 100}%` }} />
            </div>}
          </li>
        ))}
      </ul>
      {!publicOptions.length ? <p className="text-xs text-ink-500">Only the sampled choice is available in this recording.</p>
        : (visibleMass < 1 - 1e-6 || (!sampled && !choice.action && !choice.waited)) &&
          <p className="text-xs text-ink-500">Some candidate details are unavailable in the public replay.</p>}
    </div>
  );
}

const conditionLabels: Record<string, string> = { target_event: "Yes criterion", negative_outcome: "No criterion" };

function RoundSettlement({ round }: { round: Round }) {
  const choiceTexts = new Set((round.sampled_choices ?? []).flatMap(choice => [
    choice.action ?? "", ...(choice.public_options ?? []).filter(option => option.selected).map(option => option.action),
  ]).map(normalizeReportText).filter(Boolean));
  const seen = new Set<string>();
  const notes: { text: string; label?: string }[] = [];
  let repeatedAction = false;
  for (const item of round.events) {
    const key = normalizeReportText(item.text);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    if (choiceTexts.has(key)) repeatedAction = true;
    else notes.push({ text: item.text });
  }
  for (const result of round.action_results ?? []) {
    const key = normalizeReportText(result.summary);
    if (!key || key === "no public consequence was recorded" || seen.has(key)) continue;
    if (choiceTexts.has(key)) { repeatedAction = true; continue; }
    seen.add(key);
    notes.push({ text: result.summary, label: [result.actor_name, result.status].filter(Boolean).join(" · ") });
  }
  const changes = (round.question_updates ?? []).filter(change => change.before !== change.after);
  const assessment = round.contract_outlook;
  const assessmentText = assessment?.estimate_kind === "verified_simulation"
    ? verdictLabels[assessment.p_yes === 1 ? "yes" : "no"]
    : `${probability(assessment?.p_yes)} estimated Yes`;
  const showBasis = assessment?.basis && !notes.some(note => similarReportText(note.text, assessment.basis));

  return (
    <div className="space-y-3 rounded-xl border border-paper-300 bg-paper-50 p-3 sm:p-4">
      <p className="font-semibold text-ink-900">Public settlement</p>
      {notes.length > 0 && <ul className="space-y-2.5 text-sm leading-relaxed text-ink-700">
        {notes.map(note => <li key={note.text} className="break-words border-l-2 border-brand-300 pl-3">
          {note.label && <p className="mb-0.5 text-xs font-medium text-ink-500">{note.label}</p>}
          {note.text}
        </li>)}
      </ul>}
      {repeatedAction && !notes.length && <p className="text-sm text-ink-600">A sampled action also appears in the public event record.</p>}
      {!notes.length && !repeatedAction && !changes.length &&
        <p className="text-sm text-ink-600">No new public development was recorded.</p>}
      {changes.length > 0 && <ul aria-label="Outcome criteria updates" className="space-y-1.5 text-xs text-ink-700">
        {changes.map((change, index) => <li key={`${change.condition_id}-${index}`} className="flex flex-wrap items-baseline gap-x-2 break-words">
          <span className="min-w-0 font-semibold text-ink-800">{conditionLabels[change.condition_id] ?? change.label}</span>
          <span>{change.before} → {change.after}</span>
        </li>)}
      </ul>}
      {assessment && <div className="space-y-1 border-t border-paper-200 pt-3">
        <p className="flex flex-wrap items-baseline gap-x-2 text-xs text-ink-600">
          <span>Conditional round assessment</span>
          <strong className="font-semibold text-brand-800">{assessmentText}</strong>
        </p>
        <p className="text-xs text-ink-500">Estimated Yes given the recorded actions and settlement through this round.</p>
        {!assessment.readout_policy && assessment.estimate_kind !== "verified_simulation" &&
          <p className="text-xs text-amber-800">Legacy estimate · not checked by the current contract-qualifier review.</p>}
        {showBasis && <details className="text-xs text-ink-600">
          <summary className="min-h-9 py-2 font-medium text-brand-800">Assessment basis</summary>
          <p className="break-words pb-1">{assessment.basis}</p>
        </details>}
      </div>}
    </div>
  );
}

const roundControl = "inline-flex min-h-11 items-center gap-1.5 rounded-lg border px-2.5 py-2 text-xs font-semibold sm:px-3";

export default function TrajectoryReplay({ rows, plannedRounds }: { rows: Trajectory[]; plannedRounds?: number }) {
  const [sampleId, setSampleId] = useState(rows[0].sample_id);
  const [roundIndex, setRoundIndex] = useState(0);
  const roundHeading = useRef<HTMLDivElement>(null);
  const focusRound = useRef(false);
  const row = rows.find(trajectory => trajectory.sample_id === sampleId) ?? rows[0];
  const current = Math.max(0, Math.min(roundIndex, row.rounds.length - 1));
  const round = row.rounds[current];
  const totalRounds = row.planned_rounds ?? plannedRounds ?? row.rounds[0]?.interval?.horizon_steps ?? row.rounds.length;
  const closingTime = round?.interval?.interval_end_utc ?? round?.interval?.interval_end;
  const closingLabel = round?.interval?.interval_end_utc ? formatReportDate(round.interval.interval_end_utc) : closingTime;
  const lastRound = current === row.rounds.length - 1;
  const choices = round?.sampled_choices ?? [];

  useEffect(() => {
    if (!focusRound.current) return;
    focusRound.current = false;
    roundHeading.current?.focus();
  }, [current]);

  function goToRound(index: number) {
    focusRound.current = true;
    setRoundIndex(index);
  }

  return (
    <div className="min-w-0 space-y-4">
      <p className="max-w-prose text-sm text-ink-600">Follow one trajectory from sampled actions to public settlement. Each bar shows the recorded chance of selecting that action for its actor in that round.</p>
      <div className="min-w-0 space-y-3 rounded-xl border border-paper-300 bg-paper-50 p-3 sm:p-4">
        <label className="grid min-w-0 gap-1.5 text-xs font-semibold text-ink-700">
          <span className="flex items-center gap-1.5"><GitBranch size={14} aria-hidden="true" /> Trajectory</span>
          <select value={row.sample_id} onChange={event => { setSampleId(Number(event.target.value)); setRoundIndex(0); }}
            className="min-h-11 w-full min-w-0 rounded-lg border border-paper-300 bg-white px-3 py-2 text-sm font-medium text-ink-900">
            {rows.map(trajectory => <option key={trajectory.sample_id} value={trajectory.sample_id}>
              Trajectory {trajectory.sample_id} · {resultLabel(trajectory)}
            </option>)}
          </select>
        </label>
        <p className="text-xs text-ink-600">{row.rounds.length} of {totalRounds} rounds recorded · {row.status.replaceAll("_", " ")}</p>
        {row.error && <p role="alert" className="break-words text-xs text-amber-800">{row.error}</p>}
        {(isProbability(row.action_path_probability) || isProbability(row.represented_mass)) &&
          <dl className="flex flex-wrap gap-x-6 gap-y-1.5 text-xs text-ink-600">
            {isProbability(row.action_path_probability) && <div className="flex flex-wrap gap-x-2">
              <dt>Action path probability</dt><dd className="font-semibold text-ink-900">{probability(row.action_path_probability)}</dd>
            </div>}
            {isProbability(row.represented_mass) && <div className="flex flex-wrap gap-x-2">
              <dt>Weight in forecast</dt><dd className="font-semibold text-ink-900">{probability(row.represented_mass)}</dd>
            </div>}
          </dl>}
        <nav aria-label="Replay rounds" className="space-y-1.5 border-t border-paper-200 pt-2.5">
          <p className="text-xs font-medium text-ink-600">Rounds</p>
          <ol className="flex flex-wrap gap-1.5">
            {Array.from({ length: totalRounds }, (_, index) => (
              <li key={index}>
                <button type="button" disabled={!row.rounds[index]} aria-label={`Round ${index + 1}${!row.rounds[index] ? " · not recorded" : ""}`} aria-current={row.rounds[index] && index === current ? "step" : undefined}
                  onClick={() => setRoundIndex(index)}
                  className={`min-h-11 min-w-11 rounded-lg border px-2.5 text-xs font-semibold tabular-nums disabled:cursor-not-allowed disabled:opacity-40 ${row.rounds[index] && index === current
                    ? "border-brand-700 bg-brand-700 text-white" : "border-paper-300 bg-white text-ink-700 hover:border-brand-400 hover:text-brand-800"}`}>
                  {index + 1}
                </button>
              </li>
            ))}
          </ol>
        </nav>
      </div>
      {round ? <><div ref={roundHeading} tabIndex={-1} role="status"
        className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 rounded-sm">
        <p className="font-semibold text-ink-900">Round {round.step}</p>
        {closingTime && <p className="text-xs text-ink-600">Ends <time dateTime={closingTime}>{closingLabel}</time></p>}
      </div>
      <div key={`${row.sample_id}-${round.step}`} className="space-y-2.5">
        <div className="space-y-3 rounded-xl border border-paper-300 p-3 sm:p-4">
          <p className="font-semibold text-ink-900">Actor decisions</p>
          {choices.length ? choices.map(choice => <ActorChoices key={choice.actor_id} choice={choice} />)
            : <p className="text-sm text-ink-600">No sampled actor choices were recorded for this round.</p>}
        </div>
        <div aria-hidden="true" className="flex justify-center text-brand-600"><ArrowDown size={17} /></div>
        <RoundSettlement round={round} />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-paper-200 pt-3">
        <button type="button" disabled={current === 0} onClick={() => goToRound(current - 1)}
          className={`${roundControl} border-paper-300 text-ink-700 hover:bg-paper-100 disabled:opacity-40`}>
          <ChevronLeft size={15} aria-hidden="true" /> Previous round
        </button>
        {lastRound ? <p className="text-xs text-ink-500">{row.rounds.length < totalRounds
          ? "Recording stopped before the deadline. Resume the forecast to finish the remaining rounds."
          : "Forecast deadline reached"}</p> :
          <button type="button" onClick={() => goToRound(current + 1)}
            className={`${roundControl} border-brand-700 bg-brand-700 text-white hover:bg-brand-800`}>
            Next round <ChevronRight size={15} aria-hidden="true" />
          </button>}
      </div></> : <p role="status" className="rounded-xl border border-paper-200 p-4 text-sm text-ink-600">
        This trajectory stopped before its first round was saved. Resume the forecast to retry it.
      </p>}
    </div>
  );
}
