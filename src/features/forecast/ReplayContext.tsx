"use client";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import DecisionDossier from "@/components/DecisionDossier";
import { apiRequest, type SampleReplay } from "@/services/experimentService";

const LinkedKnowledgeGraph = dynamic(() => import("@/components/LinkedKnowledgeGraph"), {
  ssr: false, loading: () => <p role="status" className="py-4 text-xs text-ink-500">Loading the event graph…</p>,
});

// Mounted only when requested. Changing samples remounts it to isolate snapshots.
export default function ReplayContext({ replay, agents, apiBase, runId }: {
  replay: SampleReplay; agents: any[]; apiBase: string; runId: string;
}) {
  const [graphOpen, setGraphOpen] = useState(false);
  const [step, setStep] = useState(0);
  const [bdiOpen, setBdiOpen] = useState(false);
  const [actorStates, setActorStates] = useState<any>(null);
  const [actorId, setActorId] = useState("");
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!bdiOpen) return;
    let cancelled = false;
    setError("");
    apiRequest(apiBase, `/experiments/${runId}/samples/${replay.sample_id}/actor-states`)
      .then(data => { if (!cancelled) { setActorStates(data); setActorId(data.actors?.[0]?.agent_id || ""); } })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [bdiOpen, apiBase, runId, replay.sample_id, retry]);

  const graph = useMemo(() => {
    const events = replay.rounds[step]?.timeline || [];
    const involved = new Set(events.flatMap(e => e.actors || []));
    return {
      nodes: [
        ...agents.filter(a => involved.has(a.agent_id)).map(a => ({ id: a.agent_id, label: a.name, type: "actor" })),
        ...events.map(e => ({ id: e.timeline_id, label: e.description, type: "simulated" })),
      ],
      edges: events.flatMap(e => (e.actors || []).filter(id => agents.some(a => a.agent_id === id))
        .map(id => ({ source: e.timeline_id, target: id, label: "Participating actors" }))),
    };
  }, [replay, step, agents]);
  const actor = actorStates?.actors?.find((a: any) => a.agent_id === actorId);
  const core = agents.find(a => a.agent_id === actorId)?.decision_core || {
    version: "decision-core-1", origin: "legacy_projection", mandate: [], goals: [], means: [], resources: [], limits: [], decision_rules: [],
  };
  return <div className="space-y-2 border-t border-paper-200 pt-3 text-xs">
    {replay.rounds.length > 0 && <details onToggle={e => setGraphOpen(e.currentTarget.open)}>
      <summary className="py-2 text-brand-800">Event Graph</summary>
      {graphOpen && <div className="space-y-3 pt-3">
        <label className="flex flex-wrap items-center gap-3">Graph step
          <select value={step} onChange={e => setStep(Number(e.target.value))} className="rounded-lg border border-paper-300 bg-white p-2">
            {replay.rounds.map((r, i) => <option key={r.step} value={i}>Step {r.step}</option>)}
          </select>
        </label>
        {graph.nodes.length ? <LinkedKnowledgeGraph graph={graph} onSelect={() => {}} /> : <p className="text-ink-500">This step has no public events to display.</p>}
        <p className="text-ink-500">The graph shows simulated events for the selected step.</p>
      </div>}
    </details>}
    <details onToggle={e => setBdiOpen(e.currentTarget.open)}>
      <summary className="py-2 text-brand-800">Actor BDI · latest saved state</summary>
      {bdiOpen && <div className="space-y-4 pt-3">
        {error ? <div role="alert" className="text-red-700"><p>Could not load actor states. {error}</p>
          <button type="button" onClick={() => setRetry(v => v + 1)} className="mt-2 min-h-11 underline">Retry</button></div>
          : !actorStates ? <p role="status" className="text-ink-500">Loading actor states…</p>
          : actorStates.actors?.length ? <>
            <p className="text-ink-500">Actor states saved at round {actorStates.round_id ?? "unknown"}.</p>
            <label className="flex flex-wrap items-center gap-3">Select actor
              <select value={actorId} onChange={e => setActorId(e.target.value)} className="max-w-full rounded-lg border border-paper-300 bg-white p-2">
                {actorStates.actors.map((a: any) => <option key={a.agent_id} value={a.agent_id}>{a.name}</option>)}
              </select>
            </label>
            {actor && <DecisionDossier key={actorId} compact core={core} beliefs={actor.beliefs}
              goals={actor.desires || []} plan={actor.current_plan} />}
          </> : <p className="text-ink-500">This trajectory has no saved actor snapshot.</p>}
      </div>}
    </details>
  </div>;
}
