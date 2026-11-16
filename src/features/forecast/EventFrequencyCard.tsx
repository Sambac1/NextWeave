"use client";
import type { EventFrequency } from "@/services/experimentService";

const pct = (value: number) => `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
const strictLabels = { yes: "Confirmed occurrence", no: "Confirmed non-occurrence", unclear: "Undetermined", unfinished: "Unfinished" };

export default function EventFrequencyCard({ value, onInspect }: {
  value: EventFrequency;
  onInspect: (sample: number) => void;
}) {
  const planned = value.planned_samples;
  const trace = value.trace_metrics;
  const labels = trace ? {yes:"Matched",no:"No match",unclear:"Needs review",unfinished:"Unfinished"} : strictLabels;
  const displayedAnswers = value.answers_by_sample.map(a=>trace && a.answer!=="unfinished" ? {...a,
    answer:(a.trace_answer==="matched" ? "yes" : a.trace_answer==="not_matched" ? "no" : "unclear") as "yes"|"no"|"unclear",
    reason:a.trace_reason || a.reason,
  } : a);
  const groups = [
    { label: labels.yes, count: trace?.matched_count ?? value.count, color: "bg-signal-blue" },
    { label: labels.no, count: trace?.not_matched_count ?? value.no_count, color: "bg-emerald-600" },
    { label: labels.unclear, count: trace?.unclear_count ?? (value.unclear_count || 0), color: "bg-amber-400" },
    { label: "Unfinished", count: value.unfinished_count, color: "bg-paper-400" },
  ];
  const grouped = new Map<string, typeof value.answers_by_sample>();
  for (const answer of displayedAnswers) {
    const reason = (answer.reason || "No explanation was recorded").split(/。原判断：| Original assessment · /, 1)[0].trim();
    const key = `${answer.answer}\0${reason}`;
    grouped.set(key, [...(grouped.get(key) || []), answer]);
  }
  return <article className="border border-paper-300 rounded-lg p-4 space-y-3 bg-white">
    <div className="flex flex-wrap items-start justify-between gap-2">
      <h6 className="text-sm font-medium">{value.label}</h6>
      <p className="text-sm font-semibold tabular-nums text-brand-800">{trace
        ? trace.match_frequency == null ? "No assessment available" : `Matched ${trace.matched_count}/${trace.reviewed_count} · ${pct(trace.match_frequency)}`
        : value.frequency == null ? "No assessment available" : `Confirmed occurrence ${value.count}/${value.assessable_samples} · ${pct(value.frequency)}`}</p>
    </div>
    {(value.unfinished_count > 0 || (trace?.unclear_count ?? value.unclear_count ?? 0) > 0) && <p className="text-xs text-ink-500">
      Needs review {trace?.unclear_count ?? value.unclear_count ?? 0} · Unfinished {value.unfinished_count}
    </p>}
    <details className="text-xs">
      <summary className="py-2 text-brand-800">Assessment criteria and evidence</summary>
      <div className="space-y-4 pt-3">
    <p className="text-xs text-ink-500 leading-relaxed">{value.criterion}</p>
    <div className="flex h-2.5 rounded-full overflow-hidden bg-paper-200" role="img"
      aria-label={groups.map(g => `${g.label} ${g.count}`).join(", ")}>
      {groups.map(g => <div key={g.label} className={g.color} style={{ width: pct(planned ? g.count / planned : 0) }} />)}
    </div>
    <div className="grid grid-cols-2 gap-2 text-xs">
      {groups.map(g => <div key={g.label} className="flex items-center gap-1.5">
        <span className={`h-2 w-2 rounded-full ${g.color}`} />{g.label}<strong className="ml-auto font-mono">{g.count}</strong>
      </div>)}
    </div>
    {trace ? <>
      <p className="text-xs text-ink-400">The model checks each trajectory against the full event criteria.</p>
      <details className="text-xs text-ink-500"><summary className="cursor-pointer">Evidence check · {value.assessable_samples}/{planned} assessable</summary>
        <p className="mt-2">Confirmed occurrence {value.count} · Confirmed non-occurrence {value.no_count} · Insufficient evidence {value.unclear_count || 0} · Unfinished {value.unfinished_count}</p>
        <p className="mt-1 text-ink-400">This check reviews dates, durations, recorded states, and evidence of non-occurrence.</p>
      </details>
    </> : <p className="text-xs text-ink-500">{value.frequency == null ? "There is insufficient evidence to estimate the frequency." : `Occurrence frequency ${pct(value.frequency)}.`}</p>}
    <details className="text-xs">
      <summary className="cursor-pointer text-signal-blue">View trajectory assessments and records</summary>
      <div className="space-y-3 pt-3">
        {Array.from(grouped.entries()).map(([key, answers]) => {
          const first = answers[0];
          const reason = (first.reason || "No explanation was recorded").split(/。原判断：| Original assessment · /, 1)[0];
          const evidenceByText = new Map<string, {description:string;samples:number[]}>();
          for (const answer of answers) for (const evidence of answer.evidence) {
            const text=evidence.description.trim().replace(/\s+/g," ");
            if (!text) continue;
            const item=evidenceByText.get(text) || {description:evidence.description,samples:[]};
            if (!item.samples.includes(answer.sample_id)) item.samples.push(answer.sample_id);
            evidenceByText.set(text,item);
          }
          return <div key={key} className="rounded-lg border border-paper-200 p-3 space-y-2">
            <p className="font-medium text-ink-700">{labels[first.answer]} · {answers.length} trajectories</p>
            <p className="text-ink-500">{reason}</p>
            <div className="flex flex-wrap gap-2">{answers.map(a=><button key={a.sample_id} type="button"
              className="rounded-md bg-paper-100 px-2 py-1 text-signal-blue hover:bg-paper-200"
              onClick={()=>onInspect(a.sample_id)}>View trajectory #{a.sample_id}</button>)}</div>
            {!!evidenceByText.size && <details>
              <summary className="cursor-pointer text-ink-500">Show linked evidence · {evidenceByText.size} unique records</summary>
              <div className="space-y-3 pt-2">{Array.from(evidenceByText.entries()).map(([text,item])=><div key={text} className="border-l-2 border-paper-300 pl-2">
                <p className="text-ink-500">{item.description}</p>
                <div className="mt-1 flex flex-wrap gap-2">{item.samples.map(id=><button key={id} type="button" onClick={()=>onInspect(id)} className="text-signal-blue underline">Trajectory #{id}</button>)}</div>
              </div>)}</div>
              <p className="mt-2 text-ink-400">Repeated text is grouped for display. Each trajectory is assessed separately.</p>
            </details>}
          </div>;
        })}
      </div>
    </details>
      </div>
    </details>
  </article>;
}
