"use client";
import type { IssueChange, ReplayRound, TerminalDimension } from "@/services/experimentService";

const normal = (text: string) => text.trim().replace(/\s+/g, " ").replace(/[。.!?！？]+$/, "");

export default function ReplayStepCard({ round, previousRound, actorNames = {}, pendingIssues, previousPending = [], dimensions = [] }: {
  round: ReplayRound; previousRound?: ReplayRound; actorNames?: Record<string, string>;
  pendingIssues?: IssueChange[]; previousPending?: IssueChange[]; dimensions?: TerminalDimension[];
}) {
  const seen = new Set<string>();
  const events = (round.timeline || []).filter(event => {
    const key = normal(event.description || "");
    if (!key || seen.has(key)) return false;
    seen.add(key); return true;
  });
  const updates = (round.issue_updates || []).filter(c => c.kind !== "unchanged");
  const changes = updates.filter(c => normal(c.before || "") !== normal(c.after || ""));
  const actions = round.agent_actions || [];
  const actingNames = Array.from(new Set(actions.filter(a => a.decision === "act")
    .map(a => a.agent_name || actorNames[a.agent_id] || a.agent_id)));
  const pending = pendingIssues ?? (round.issue_updates || []).filter(c => c.pending_actor_ids?.length);
  const newlyPending = pending.filter(item => !previousPending.some(previous =>
    previous.issue_id === item.issue_id && previous.pending_actor_ids.length === item.pending_actor_ids.length
    && item.pending_actor_ids.every(id => previous.pending_actor_ids.includes(id))));
  const actionCount = round.public_action_count ?? actions.filter(a => a.decision === "act").length;
  const waitCount = round.public_wait_count ?? actions.filter(a => ["wait", "monitor", "no_action"].includes(a.decision || "")).length;
  // External events may occur even when no actor takes a public action.
  const hasChange = changes.length > 0;
  const recordedTexts = new Set([...changes.map(c => c.after), ...events.map(e => e.description)].map(normal));
  const notes = [round.state_summary, ...(round.continuing_state_notes || [])].filter((note): note is string => {
    const key = normal(note || "");
    if (!key || recordedTexts.has(key)) return false;
    recordedTexts.add(key);
    return true;
  });
  const decisionLabels: Record<string, string> = { act: "Act", wait: "Wait", monitor: "Monitor", no_action: "Take no action" };
  const previousTexts = new Set((previousRound?.timeline || []).map(event => normal(event.description || "")));
  const previewEvents = hasChange ? [] : events.filter(event => !previousTexts.has(normal(event.description))).slice(0, 2);
  const remainingEvents = events.filter(event => !previewEvents.includes(event));
  const measurements = Object.entries(round.measurement_states || {});
  const hasDetails = measurements.length > 0 || actions.length > 0 || updates.length > 0 || notes.length > 0
    || remainingEvents.length > 0 || pending.length > newlyPending.length;

  return <section aria-label={`Step ${round.step}`} className="replay-step space-y-4 rounded-xl border border-paper-300 bg-white">
    <header className="flex flex-wrap items-center justify-between gap-2">
      <h5 className="font-semibold text-brand-800">Step {round.step}</h5>
      {round.simulation_interval?.interval_start && <p className="text-xs text-ink-500">
        {round.simulation_interval.interval_start}{round.simulation_interval.interval_end ? ` to ${round.simulation_interval.interval_end}` : ""}
      </p>}
    </header>
    <p className="text-xs leading-relaxed text-ink-500">
      {actingNames.length ? <><span className="font-medium text-ink-700">{actingNames.join(", ")}</span> · {actionCount} public actions</>
        : actionCount ? `${actionCount} public actions` : "No public actor action was recorded."}
      {waitCount > 0 ? ` · ${waitCount} explicit waits` : ""}
    </p>
    {hasChange ? <div className="space-y-4">{changes.map((c, index) => <article key={`${c.issue_id}-${index}`} className="border-l-2 border-brand-200 pl-4">
      <h6 className="text-xs font-medium text-brand-800">{c.title}</h6>
      <p className="mt-1 text-sm leading-relaxed">{c.after}</p>
    </article>)}</div> : previewEvents.length > 0 ? <ol className="space-y-3">{previewEvents.map(event => <li key={event.timeline_id} className="border-l-2 border-brand-200 pl-4">
      <p className="text-sm leading-relaxed">{event.description}</p>
    </li>)}</ol> : <p className="text-sm text-ink-500">{events.length
      ? "Public events repeat the previous step. Expand the record to review them." : "This step recorded no public events or issue changes."}</p>}
    {newlyPending.length > 0 && <div className="rounded-lg bg-brand-50 px-3 py-2 text-xs leading-relaxed text-brand-800">
      {newlyPending.map((c, index) => <p key={`${c.issue_id}-${index}`}>
        Awaiting response · {c.title} · {c.pending_actor_ids.map(id => actorNames[id] || id).join(", ")}
      </p>)}
    </div>}
    {hasDetails && <details className="border-t border-paper-200 pt-2 text-xs text-ink-500">
      <summary className="py-2 font-medium text-brand-800">View step record{remainingEvents.length > 0 ? ` · ${remainingEvents.length} more events` : ""}</summary>
      <div className="space-y-2 pt-2">
        {remainingEvents.length > 0 && <details>
          <summary className="py-2">Public event text · {remainingEvents.length} events</summary>
          <div className="space-y-3 pb-2 pl-3">
            {remainingEvents.map(e => <p key={e.timeline_id} className="leading-relaxed">{e.description}</p>)}
          </div>
        </details>}
        {actions.length > 0 && <details>
          <summary className="py-2">Public decisions · {actions.length}</summary>
          <div className="space-y-3 pb-2 pl-3">
            {actions.map((a, i) => <p key={`${a.agent_id}-${i}`} className="leading-relaxed">
              <span className="font-medium text-ink-700">{a.agent_name || actorNames[a.agent_id] || a.agent_id}</span>：
              {a.action || decisionLabels[a.decision || ""] || "No decision explanation was recorded"}
            </p>)}
          </div>
        </details>}
        {updates.length > 0 && <details>
          <summary className="py-2">Issue evidence · {updates.length} records</summary>
          <div className="space-y-3 pb-2 pl-3">
            {updates.map((c, index) => <div key={`${c.issue_id}-${index}`} className="space-y-1 leading-relaxed">
              <p className="font-medium">{c.title}</p>
              {c.before && <p>Previously · {c.before}</p>}
              {c.evidence_quote && <blockquote className="border-l-2 border-paper-300 pl-3">{c.evidence_quote}</blockquote>}
          </div>)}
          </div>
        </details>}
        {pending.length > newlyPending.length && <details>
          <summary className="py-2">Awaiting response · {pending.length} issues</summary>
          <ul className="space-y-2 pb-2 pl-3">{pending.map(item => <li key={item.issue_id}>
            {item.title} · {item.pending_actor_ids.map(id => actorNames[id] || id).join(", ")}
          </li>)}</ul>
        </details>}
        {measurements.length > 0 && <details>
          <summary className="py-2">Terminal observations · {measurements.length}</summary>
          <ul className="space-y-2 pb-2 pl-3">{measurements.map(([id, observation]) => {
            const dimension = dimensions.find(item => item.id === id);
            const category = dimension?.categories.find(item => item.id === observation.category_id);
            return <li key={id} className="space-y-1">
              <p className="font-medium">{dimension?.label || id} · {category?.label || "Conflicting evidence or unknown state"}</p>
              <p>{observation.observed_on}{observation.expires_on ? ` · Expires on ${observation.expires_on}` : ""}</p>
              {observation.quote && <blockquote className="border-l-2 border-paper-300 pl-3">{observation.quote}</blockquote>}
            </li>;
          })}</ul>
        </details>}
        {notes.length > 0 && <details>
          <summary className="py-2">State summary and continuing issues</summary>
          <div className="space-y-2 pb-2 pl-3">{notes.map((note, i) => <p key={i} className="whitespace-pre-wrap leading-relaxed">{note}</p>)}</div>
        </details>}
      </div>
    </details>}
  </section>;
}
