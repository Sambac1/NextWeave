"use client";
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Building2,
  CalendarDays,
  ExternalLink,
  FileCheck2,
  FileText,
  Fingerprint,
  GitBranch,
  Search,
  Users,
} from "lucide-react";
import LinkedKnowledgeGraph from "@/components/LinkedKnowledgeGraph";
import { apiRequest } from "@/services/experimentService";

const entityName = (kind: string) =>
  (
    ({
      person: "Person",
      individual: "Individual",
      country: "Country",
      government: "Public institution",
      organization: "Organization",
      institution: "Institution",
      company: "Company",
      group: "Group",
      collective: "Collective",
      asset: "Asset or object",
      location: "Location",
      event: "Event object",
    }) as Record<string, string>
  )[kind] || "Participants";
const roleName = (role: string) =>
  (
    ({
      principal: "Decision actor",
      member: "Member",
      mentioned: "Related participants",
    }) as Record<string, string>
  )[role] || "Participants";
export const dateReason = (reason: string) =>
  (
    ({
      evidence_not_located: "The quoted passage has not been located in the source",
      coverage_period_is_not_occurrence_month:
        "The source reports a quarter or season without a specific month",
      multiple_event_months_require_narrower_evidence:
        "The quote contains several dates and needs an event-specific match",
      no_unambiguous_source_month: "The source does not establish a unique year and month",
      source_section_missing: "The source section is unavailable",
      temporal_bound_is_not_occurrence_month: "The time bounds do not establish an occurrence month",
      planned_event_is_not_observed_occurrence:
        "The source describes a future plan",
      announcement_act_not_supported: "The source does not establish that an announcement was made",
    }) as Record<string, string>
  )[reason] || "The date or quotation needs review";
function urlSafe(url: string) {
  try {
    const u = new URL(url);
    return ["http:", "https:"].includes(u.protocol) ? u.href : undefined;
  } catch {
    return undefined;
  }
}

function ActorDossier({
  node,
  onEvent,
  onOpenProfile,
}: {
  node: any;
  onEvent: (event: any) => void;
  onOpenProfile?: (id: string) => void;
}) {
  if (!node)
    return (
      <div className="p-7 text-sm text-ink-400">
        Select an entity to see its role, activities, and evidence.
      </div>
    );
  const m = node.detail || {}, p = m.profile || {};
  return (
    <aside
      className="rounded-2xl border border-paper-200 bg-white overflow-hidden"
      aria-label="Entity profile"
    >
      <div className="bg-gradient-to-br from-[#f2f6fa] to-[#faf8f3] p-5 border-b border-paper-200/70">
        <div className="flex items-center gap-2 text-xs tracking-normal text-ink-500 ">
          <Fingerprint size={14} />
          ENTITY DOSSIER
        </div>
        <h3 className="mt-3 text-xl font-semibold text-ink-800 leading-tight">
          {m.title || node.label}
        </h3>
        <div className="flex flex-wrap gap-1.5 mt-3 text-xs">
          <span className="rounded-md bg-white border border-paper-200 px-2 py-1">
            {entityName(p.agent_type)}
          </span>
          <span className="rounded-md bg-brand-50 text-brand-800 px-2 py-1">
            {roleName(p.decision_role)}
          </span>
          {p.membership_status === "model_inferred" && (
            <span className="rounded-md bg-amber-50 text-amber-800 px-2 py-1">
              Affiliation unverified
            </span>
          )}
        </div>
      </div>
      <div className="p-5 space-y-5 text-xs text-ink-600 leading-relaxed">
        {p.entity_type_status === "conflict_requires_review" && (
          <p className="text-xs text-amber-800 rounded-lg bg-amber-50 p-2">
            The saved entity type conflicts with its description. Reanalysis can verify the type.
          </p>
        )}
        {["asset", "location", "event"].includes(p.agent_type) && (
          <p className="rounded-lg bg-paper-50 p-3">
            This object links events to sources. Only independent decision actors receive a
            BDI profile. Ownership, operation, and responsibility follow the source evidence.
          </p>
        )}
        {(p.title || p.affiliation) && (
          <dl className="space-y-2">
            {p.title && (
              <div>
                <dt className="text-xs text-ink-400">Position or role</dt>
                <dd>{p.title}</dd>
              </div>
            )}
            {p.affiliation && (
              <div>
                <dt className="text-xs text-ink-400">Affiliation or appointment</dt>
                <dd>{p.affiliation}</dd>
              </div>
            )}
          </dl>
        )}
        {p.profile_scope !== "evidence_summary" &&
          p.description &&
          ![p.title, p.affiliation].includes(p.description) && (
            <p>{p.description}</p>
          )}
        <div className="grid grid-cols-2 gap-2">
          <div className="p-3 rounded-xl bg-paper-50">
            <b className="block text-lg font-semibold text-ink-700">
              {p.event_count || 0}
            </b>
            <span className="text-xs text-ink-400">Linked historical events</span>
          </div>
          <div className="p-3 rounded-xl bg-paper-50">
            <b className="block text-lg font-semibold text-ink-700">
              {p.source_count || 0}
            </b>
            <span className="text-xs text-ink-400">Source snapshots</span>
          </div>
        </div>
        {p.date_range?.[0] && (
          <p className="flex gap-2 text-xs text-ink-400">
            <CalendarDays size={13} />
            {p.date_range[0]} — {p.date_range[1]}
          </p>
        )}
        {!!p.observed_activities?.length && (
          <details>
            <summary className="cursor-pointer font-semibold text-ink-700 mb-2">Recent historical events · {p.observed_activities.length} of {p.event_count || p.observed_activities.length}</summary>
            <div className="space-y-2">
              {p.observed_activities.map((e: any) => (
                <button
                  key={e.timeline_id}
                  onClick={() => onEvent(e)}
                  className="block w-full text-left border-l-2 border-brand-200 pl-3 py-1.5 hover:bg-paper-50 rounded-r-lg"
                >
                  <span className="text-xs text-brand-700 tabular-nums">
                    {e.timestamp}
                  </span>
                  <p className="text-xs mt-1 line-clamp-3">{e.description}</p>
                </button>
              ))}
            </div>
          </details>
        )}
        {!!p.background_records?.length && (
          <details className="rounded-xl bg-amber-50/70 p-3">
            <summary className="cursor-pointer text-xs text-amber-800">
              Background and records to review ({p.background_records.length})
            </summary>
            <p className="text-xs text-ink-400 mt-2">
              These records are excluded from the dated event count and date range.
            </p>
            {p.background_records.map((item: any) => (
              <button
                key={item.timeline_id}
                onClick={() => onEvent(item)}
                className="text-left block py-2 text-xs border-b last:border-0"
              >
                <p>{item.description}</p>
                <p className="text-xs text-amber-700 mt-1">
                  {dateReason(item.reason)}
                </p>
              </button>
            ))}
          </details>
        )}
        {p.profile_scope === "evidence_summary" && (
          <p className="text-xs rounded-lg bg-brand-50 px-3 py-2 text-brand-800">
            This profile summarizes the entity's identity and linked events from the available sources.
          </p>
        )}
        {m.simulation_agent && onOpenProfile && (
          <button
            onClick={() => onOpenProfile(node.id)}
            className="w-full flex items-center justify-between border border-paper-300 rounded-xl px-3 py-2.5 text-ink-700 hover:bg-paper-50"
          >
            View decision profile and initial state
            <ArrowRight size={14} />
          </button>
        )}
      </div>
    </aside>
  );
}

function EventDetail({
  event,
  actors,
  onActor,
  graph,
  showGraph,
  onGraph,
}: {
  event: any;
  actors: any[];
  onActor: (id: string) => void;
  graph: any;
  showGraph: boolean;
  onGraph: () => void;
}) {
  if (!event)
    return (
      <div className="rounded-2xl border p-8 text-sm text-ink-400">
        Select an event to see participants and supporting sources.
      </div>
    );
  const m = event.metadata || {},
    supported = m.date_validation?.status === "supported";
  return (
    <article
      className="rounded-2xl border border-paper-200 bg-white overflow-hidden"
      aria-label="Event and source details"
    >
      <div className="p-5 border-b bg-[#fafbfd]">
        <div className="flex items-center gap-2 text-xs">
          <span className="font-mono text-brand-800">
            {supported ? m.timestamp : "Event month unknown"}
          </span>
          <span
            className={`ml-auto rounded-md px-2 py-1 text-xs ${supported ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800"}`}
          >
            {supported ? "Date and source verified" : "Date needs review"}
          </span>
        </div>
        <h3 className="mt-3 text-base font-semibold text-ink-800 leading-relaxed">
          {m.description || event.title}
        </h3>
        {!supported && (
          <p className="text-xs text-amber-700 mt-2">
            {dateReason(m.date_validation?.reason)}
            {m.proposed_timestamp
              ? `. Earlier proposed date ${m.proposed_timestamp} was excluded.`
              : ""}
          </p>
        )}
        <div className="flex flex-wrap gap-1.5 mt-4">
          {m.actor_ids?.map((id: string) => {
            const a = actors.find((n) => n.id === id);
            return (
              <button
                key={id}
                onClick={() => onActor(id)}
                className="rounded-full bg-white border border-paper-200 px-2.5 py-1.5 text-xs text-ink-600 hover:border-brand-400"
              >
                {a?.label || m.actor_names?.[id] || "Participants"}
              </button>
            );
          })}
        </div>
      </div>
      <div className="p-5 space-y-4">
        <h4 className="text-xs font-semibold flex gap-2 items-center text-ink-700">
          <FileText size={15} />
          Sources and original text
        </h4>
        {m.evidence_refs?.length ? (
          m.evidence_refs.map((r: any, i: number) => (
            <section
              key={`${r.source_id}:${i}`}
              className="border border-paper-200 rounded-xl overflow-hidden"
            >
              <div className="px-4 py-2.5 bg-paper-50 flex items-center gap-2 text-xs text-ink-500">
                <FileCheck2 size={14} />
                {(() => {
                  try {
                    return new URL(r.url).hostname;
                  } catch {
                    return "Source snapshots";
                  }
                })()}
                <span className="ml-auto text-xs">
                  {r.attribution === "exact_span" ? "Exact excerpt" : "Page-level citation"}
                </span>
              </div>
              {r.quote ? (
                <blockquote className="m-4 pl-3 border-l-2 border-emerald-200 text-xs leading-7 text-ink-600 whitespace-pre-line">
                  {r.quote}
                </blockquote>
              ) : (
                <p className="p-4 text-xs text-ink-400">
                  The saved citation could not be located as a continuous passage.
                </p>
              )}
              <div className="px-4 pb-3 flex gap-2 items-center text-xs text-ink-400">
                <span className="truncate">
                  {r.section_paths?.join(" / ") || "Page snapshot"}
                </span>
                {urlSafe(r.url) && (
                  <a
                    href={urlSafe(r.url)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-auto shrink-0 text-brand-700 flex gap-1 items-center"
                  >
                    Open source
                    <ExternalLink size={11} />
                  </a>
                )}
              </div>
            </section>
          ))
        ) : (
          <p className="text-xs text-ink-400">No traceable sources are saved.</p>
        )}
        {m.date_evidence?.quote_resolution === "ordered_ellipsis_restored" && (
          <p className="text-xs text-emerald-700">
            Omitted text was restored from a unique match in the same source paragraph.
          </p>
        )}
        {m.date_evidence?.date_resolution === "explicit_start_end_clause" && (
          <p className="text-xs text-emerald-700">
            Explicit start and end clauses establish the event dates.
          </p>
        )}
        <button
          onClick={onGraph}
          aria-expanded={showGraph}
          className="flex items-center gap-2 text-xs text-brand-800"
        >
          <GitBranch size={14} />
          {showGraph ? "Hide linked graph" : "Show participants and sources for this event"}
        </button>
        {showGraph && (
          <LinkedKnowledgeGraph
            graph={graph}
            onSelect={(n) => {
              if (n.type === "actor") onActor(n.id);
            }}
          />
        )}
      </div>
    </article>
  );
}

export default function CaseGraphExplorer({
  apiBase,
  caseId,
  filters = {},
  onSelect,
  onOpenProfile,
  revision = 0,
  historyFocus,
}: {
  apiBase: string;
  caseId: string | null;
  filters?: Record<string, string>;
  onSelect?: (node: any) => void;
  onOpenProfile?: (id: string) => void;
  revision?: number;
  historyFocus?: {actorId?:string;eventId?:string};
}) {
  const [view, setView] = useState("actors"),
    [query, setQuery] = useState(""),
    [from, setFrom] = useState(""),
    [to, setTo] = useState("");
  const [applied, setApplied] = useState<Record<string, string>>({}),
    [actorFilter, setActorFilter] = useState("");
  const [graph, setGraph] = useState<any>({ nodes: [], edges: [] }),
    [events, setEvents] = useState<any[]>([]),
    [selected, setSelected] = useState<any>(null),
    [event, setEvent] = useState<any>(null);
  const [eventGraph, setEventGraph] = useState<any>({ nodes: [], edges: [] }),
    [showGraph, setShowGraph] = useState(false),
    [directory, setDirectory] = useState(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const dossierRef = useRef<HTMLDivElement>(null),
    eventRef = useRef<HTMLDivElement>(null),
    listRef = useRef<HTMLDivElement>(null),
    actorsRef = useRef<HTMLDivElement>(null);
  const scrollDossier = useRef(false),
    scrollEvent = useRef(false);
  useEffect(() => {
    if (scrollDossier.current && view === "actors" && selected) {
      scrollDossier.current = false;
      if (window.innerWidth < 1280)
        dossierRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
    }
  }, [selected, view]);
  useEffect(() => {
    if (scrollEvent.current && view === "events" && event) {
      scrollEvent.current = false;
      if (window.innerWidth < 768)
        eventRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
    }
  }, [event, view]);
  const generation = useRef(0),
    recordGeneration = useRef(0),
    filterKey = JSON.stringify({
      ...filters,
      ...applied,
      actor_id: actorFilter || filters.actor_id || "",
    });
  useEffect(() => {
    setSelected(null);
    setEvent(null);
    setQuery("");
    setApplied({});
    setFrom("");
    setTo("");
    setActorFilter("");
  }, [caseId]);
  useEffect(() => {
    if (!apiBase || !caseId) return;
    const current = ++generation.current;
    setBusy(true);
    setError("");
    const params = new URLSearchParams(JSON.parse(filterKey));
    Promise.all([
      apiRequest(
        apiBase,
        `/knowledge/${caseId}/graph?${params}&view=actors&limit=100`,
      ),
      apiRequest(
        apiBase,
        `/knowledge/${caseId}/search?${params}&kind=timeline&limit=100`,
      ),
    ])
      .then(([g, data]) => {
        if (generation.current !== current) return;
        setGraph({ ...g, scope_key: `${caseId}:${g.version}:${filterKey}` });
        const ordered = [...data.hits].sort(
          (a: any, b: any) =>
            Number(a.metadata.date_validation?.status !== "supported") -
              Number(b.metadata.date_validation?.status !== "supported") ||
            String(a.metadata.timestamp).localeCompare(
              String(b.metadata.timestamp),
            ) ||
            a.id.localeCompare(b.id),
        );
        setEvents(ordered);
        setSelected(
          (old: any) =>
            g.nodes.find((n: any) => n.id === old?.id) ||
            g.nodes.find((n: any) => n.decision_role === "principal") ||
            g.nodes.find((n: any) => n.type === "actor") ||
            null,
        );
        setEvent(
          (old: any) =>
            ordered.find((e: any) => e.id === old?.id) || ordered[0] || null,
        );
      })
      .catch((e) => {
        if (generation.current === current) setError(e.message);
      })
      .finally(() => {
        if (generation.current === current) setBusy(false);
      });
    return () => {
      generation.current++;
      recordGeneration.current++;
    };
  }, [apiBase, caseId, filterKey, revision]);
  useEffect(() => {
    if (!showGraph || !event || !caseId) return;
    let cancelled = false;
    setEventGraph({ nodes: [], edges: [] });
    apiRequest(
      apiBase,
      `/knowledge/${caseId}/graph?view=evidence&focus_id=${encodeURIComponent(event.id)}&limit=50`,
    )
      .then((g) => {
        if (!cancelled)
          setEventGraph({ ...g, scope_key: `evidence:${caseId}:${event.id}` });
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [event, showGraph, apiBase, caseId]);
  const actors = graph.nodes.filter((n: any) => n.type === "actor");
  async function openEvent(item: any) {
    const token = ++recordGeneration.current;
    setView("events");
    scrollEvent.current = true;
    setShowGraph(false);
    const local = events.find((e) => e.id === item.timeline_id);
    if (local) {
      setEvent({ ...local });
      return;
    }
    try {
      const value = await apiRequest(
        apiBase,
        `/knowledge/${caseId}/records/${encodeURIComponent(item.timeline_id)}`,
      );
      if (token === recordGeneration.current) setEvent(value);
    } catch (e: any) {
      if (token === recordGeneration.current) setError(e.message);
    }
  }
  async function selectActor(id: string) {
    const token = ++recordGeneration.current;
    const node = actors.find((a: any) => a.id === id);
    if (node) {
      scrollDossier.current = true;
      setSelected({ ...node });
      setView("actors");
      onSelect?.(node);
      return;
    }
    try {
      const record = await apiRequest(
        apiBase,
        `/knowledge/${caseId}/records/${encodeURIComponent(id)}`,
      );
      if (token !== recordGeneration.current) return;
      const loaded = {
        id: record.id,
        label: record.title,
        type: "actor",
        decision_role: record.metadata.profile?.decision_role,
        detail: record.metadata,
      };
      scrollDossier.current = true;
      setSelected(loaded);
      setView("actors");
      onSelect?.(loaded);
      setQuery("");
      setApplied((old) => ({ ...old, q: "" }));
      setActorFilter(id);
    } catch (e: any) {
      if (token === recordGeneration.current) setError(e.message);
    }
  }

  const appliedFocus = useRef("");
  useEffect(() => {
    const key = `${caseId || ""}:${historyFocus?.actorId || ""}:${historyFocus?.eventId || ""}`;
    if (busy || !caseId || !graph.scope_key || appliedFocus.current === key || (!historyFocus?.actorId && !historyFocus?.eventId)) return;
    appliedFocus.current = key;
    if (historyFocus.eventId) void openEvent({timeline_id:historyFocus.eventId});
    else if (historyFocus.actorId) void selectActor(historyFocus.actorId);
  // Focus is applied once after the requested knowledge graph finishes loading.
  }, [caseId, busy, graph.scope_key, historyFocus?.actorId, historyFocus?.eventId]);

  return (
    <section aria-label="Historical Knowledge Browser" className="knowledge-browser space-y-5">
      <div className="flex flex-wrap gap-2 border-b border-paper-200 pb-3">
        {[
          ["actors", "Actors and Members", Users],
          ["events", "Events and Sources", CalendarDays],
        ].map(([id, label, Icon]: any) => (
          <button
            key={id}
            aria-pressed={view === id}
            onClick={() => {
              recordGeneration.current++;
              setView(id);
            }}
            className={`flex gap-2 items-center rounded-xl px-4 py-2.5 text-xs transition-colors ${view === id ? "bg-signal-blue text-white" : "text-ink-500 hover:bg-paper-100"}`}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
        <span className="ml-auto self-center text-xs text-ink-400">
          Historical evidence
        </span>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          setApplied({ q: query, from_month: from, to_month: to });
          setShowGraph(false);
        }}
        className="flex flex-wrap gap-2"
      >
        <div className="flex-1 min-w-[150px] relative">
          <Search size={14} className="absolute left-3 top-3 text-ink-400" />
          <input
            aria-label="Search knowledge"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search names, events, or keywords"
            className="w-full rounded-xl border border-paper-200 pl-9 pr-3 py-2.5 text-xs bg-white"
          />
        </div>
        <input
          type="month"
          aria-label="Start month"
          value={from}
          onChange={(e) => setFrom(e.target.value)}
          className="rounded-xl border border-paper-200 px-2 py-2 text-xs max-w-[140px]"
        />
        <input
          type="month"
          aria-label="End month"
          value={to}
          onChange={(e) => setTo(e.target.value)}
          className="rounded-xl border border-paper-200 px-2 py-2 text-xs max-w-[140px]"
        />
        <button className="px-4 rounded-xl bg-white border border-paper-300 text-xs text-ink-700">
          Search
        </button>
      </form>
      {actorFilter && (
        <button
          className="text-xs text-brand-700"
          onClick={() => setActorFilter("")}
        >
          Clear entity filter ×
        </button>
      )}
      {busy && (
        <p role="status" className="text-xs text-ink-400">
          Loading saved knowledge…
        </p>
      )}
      {error && (
        <p
          role="alert"
          className="text-xs text-red-700 bg-red-50 rounded-xl p-3"
        >
          {error}
        </p>
      )}
      {view === "actors" && (
        <div className="knowledge-actor-layout grid gap-6 items-start">
          <div ref={actorsRef} className="space-y-3 min-w-0">
            <div className="flex items-center justify-between">
              <p className="text-xs text-ink-500">
                Start with decision actors, then expand their members or related entities.
              </p>
              <button
                className="text-xs text-brand-800 border rounded-lg px-2.5 py-1.5"
                onClick={() => setDirectory(!directory)}
              >
                {directory ? "Back to graph" : "Entity directory"}
              </button>
            </div>
            {directory ? (
              <div className="grid sm:grid-cols-2 gap-2">
                {actors.map((a: any) => (
                  <button
                    key={a.id}
                    onClick={() => selectActor(a.id)}
                    className={`text-left p-4 rounded-xl border bg-white ${selected?.id === a.id ? "border-brand-400 ring-1 ring-brand-100" : "border-paper-200"}`}
                  >
                    <Building2 size={18} className="text-brand-600 mb-2" />
                    <p className="text-sm font-semibold text-ink-700">
                      {a.label}
                    </p>
                    <p className="text-xs text-ink-400 mt-1">
                      {roleName(a.decision_role)} ·{" "}
                      {a.detail?.profile?.event_count || 0} events
                    </p>
                  </button>
                ))}
              </div>
            ) : (
              <LinkedKnowledgeGraph
                graph={graph}
                onSelect={(n) => {
                  if (n.type === "actor") selectActor(n.id);
                }}
              />
            )}
            <p className="text-xs text-ink-400">
              Affiliations are inferred from available profiles. Unconfirmed participants have their own evidence profiles.
            </p>
            {selected && (
              <button
                onClick={() => {
                  setActorFilter(selected.id);
                  setView("events");
                }}
                className="text-xs text-brand-800 inline-flex gap-2 items-center"
              >
                View all events linked to {selected.label}
                <ArrowRight size={13} />
              </button>
            )}
          </div>
          <div ref={dossierRef} className="scroll-mt-3 min-w-0">
            <button
              className="knowledge-back mb-2 text-xs text-brand-800"
              onClick={() =>
                actorsRef.current?.scrollIntoView({
                  behavior: "smooth",
                  block: "start",
                })
              }
            >
              ↑ Back to actors and members
            </button>
            <ActorDossier
              node={selected}
              onEvent={openEvent}
              onOpenProfile={onOpenProfile}
            />
          </div>
        </div>
      )}
      {view === "events" && (
        <div className="knowledge-events-layout grid gap-6 items-start">
          <div
            ref={listRef}
            className="rounded-2xl border border-paper-200 bg-white overflow-hidden"
          >
            <div className="px-4 py-3 text-xs text-ink-500 bg-paper-50 border-b">
              {events.length} matching records
              {events.length === 100 ? " · Narrow the date range to see more" : ""}
            </div>
            <div className="max-h-[560px] overflow-y-auto">
              {events.map((e: any, i: number) => {
                const m = e.metadata,
                  month = m.timestamp?.slice(0, 7) || "Background or under review",
                  newMonth =
                    i === 0 ||
                    month !==
                      (events[i - 1].metadata.timestamp?.slice(0, 7) ||
                        "Background or under review");
                return (
                  <div key={e.id}>
                    {newMonth && (
                      <div className="sticky top-0 bg-[#f7f9fc] border-b px-4 py-2 text-xs font-mono font-semibold text-ink-500">
                        {month}
                      </div>
                    )}
                    <button
                      onClick={() => {
                        recordGeneration.current++;
                        scrollEvent.current = true;
                        setEvent({ ...e });
                        setShowGraph(false);
                      }}
                      className={`w-full text-left border-b last:border-0 p-4 flex gap-3 ${event?.id === e.id ? "bg-brand-50/70 border-l-2 border-l-blue-500" : "hover:bg-paper-50 border-l-2 border-l-transparent"}`}
                    >
                      <span
                        className={`w-2 h-2 shrink-0 rounded-full mt-1.5 ${m.date_validation?.status === "supported" ? "bg-emerald-500" : "bg-amber-400"}`}
                      />
                      <div className="min-w-0">
                        <p className="text-xs tabular-nums text-ink-400">
                          {m.timestamp}
                        </p>
                        <p className="text-xs text-ink-700 leading-relaxed mt-1 line-clamp-3">
                          {m.description || e.title}
                        </p>
                        <span className="text-xs text-ink-400 mt-2 block">
                          {m.actor_ids?.length || 0} participants ·{" "}
                          {m.evidence_refs?.length || 0} sources
                        </span>
                      </div>
                    </button>
                  </div>
                );
              })}
              {!events.length && !busy && (
                <p className="p-8 text-xs text-ink-400">
                  No events match the current filters.
                </p>
              )}
            </div>
          </div>
          <div ref={eventRef} className="min-w-0 scroll-mt-3">
            <button
              className="md:hidden mb-2 text-xs text-brand-800"
              onClick={() =>
                listRef.current?.scrollIntoView({
                  behavior: "smooth",
                  block: "start",
                })
              }
            >
              ↑ Back to events
            </button>
            <EventDetail
              event={event}
              actors={actors}
              onActor={selectActor}
              graph={eventGraph}
              showGraph={showGraph}
              onGraph={() => setShowGraph(!showGraph)}
            />
          </div>
        </div>
      )}
    </section>
  );
}
