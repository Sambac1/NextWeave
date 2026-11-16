"use client";
import { useEffect, useRef, useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  FileText,
  RefreshCw,
  ShieldCheck,
  Users,
} from "lucide-react";
import CaseGraphExplorer, { dateReason } from "@/features/CaseGraphExplorer";
import { apiRequest } from "@/services/experimentService";

export default function KnowledgePanel({
  apiBase,
  caseId,
  disabled,
  onReanalyze,
  onOpenProfile,
  historyFocus,
}: {
  apiBase: string;
  caseId: string | null;
  disabled?: boolean;
  onReanalyze?: () => void;
  onOpenProfile?: (id: string) => void;
  historyFocus?: {actorId?:string;eventId?:string};
}) {
  const [manifest, setManifest] = useState<any>(null),
    [quality, setQuality] = useState<any>(null),
    [error, setError] = useState("");
  const [busy, setBusy] = useState(false),
    [revision, setRevision] = useState(0);
  const generation = useRef(0);
  useEffect(() => {
    if (!apiBase || !caseId) return;
    const current = ++generation.current;
    setBusy(true);
    setError("");
    Promise.all([
      apiRequest(apiBase, `/knowledge/${caseId}/search?limit=1`),
      apiRequest(apiBase, `/knowledge/${caseId}/quality`),
    ])
      .then(([data, audit]) => {
        if (generation.current === current) {
          setManifest(data.manifest);
          setQuality(audit);
        }
      })
      .catch((e) => {
        if (generation.current === current) setError(e.message);
      })
      .finally(() => {
        if (generation.current === current) setBusy(false);
      });
    return () => {
      generation.current++;
    };
  }, [apiBase, caseId, revision]);
  return (
    <div className="bg-white text-ink-700 min-h-full">
      <header className="p-4 border-b border-paper-200 bg-white">
        <div className="flex gap-3 items-start">
          <div className="p-2.5 rounded-xl bg-brand-50 text-signal-blue">
            <BookOpen size={20} />
          </div>
          <div className="min-w-0">
            <p className="text-xs font-semibold tracking-normal text-ink-400">
              KNOWLEDGE WORKSPACE
            </p>
            <h2 className="text-base font-semibold text-ink-800 mt-1">
              Knowledge Workspace
            </h2>
            <p className="text-xs text-ink-400 mt-1">
              Trace participants and events back to their sources.
            </p>
          </div>
          <button
            aria-label="Refresh historical knowledge"
            disabled={busy || disabled}
            onClick={() => setRevision((v) => v + 1)}
            className="ml-auto p-2 border rounded-lg bg-white disabled:opacity-40"
          >
            <RefreshCw size={15} className={busy ? "animate-spin" : ""} />
          </button>
        </div>
        <div className="grid grid-cols-3 gap-2 mt-3">
          {[
            [
              BookOpen,
              quality?.source_supported_events ?? manifest?.timeline_count,
              "Dated events",
            ],
            [Users, manifest?.actor_count, "Participants"],
            [FileText, manifest?.source_count, "Source snapshots"],
          ].map(([Icon, count, label]: any) => (
            <div
              key={label}
              className="flex items-center gap-2.5 rounded-xl bg-paper-50 px-3 py-2 border border-paper-100"
            >
              <Icon size={16} className="text-ink-400 shrink-0" />
              <div>
                <span className="text-base font-semibold tabular-nums text-ink-700">
                  {count ?? "—"}
                </span>
                <span className="block text-xs text-ink-400">
                  {label}
                </span>
              </div>
            </div>
          ))}
        </div>
      </header>
      <div className="p-4 sm:p-5 space-y-4">
        {error && (
          <p
            role="alert"
            className="p-3 bg-red-50 text-red-700 text-xs rounded-xl"
          >
            {error}
          </p>
        )}
        {quality && (
          <details className="rounded-xl bg-white border border-paper-200">
            <summary className="cursor-pointer p-3.5 flex gap-2 items-center text-xs">
              <ShieldCheck
                size={16}
                className={
                  quality.needs_review?.length
                    ? "text-amber-600"
                    : "text-emerald-600"
                }
              />
              <span className="font-medium">Date and Evidence Check</span>
              <span className="ml-auto text-ink-500 tabular-nums">
                {quality.source_supported_events} / {quality.historical_events}{" "}
                events supported
              </span>
            </summary>
            <div className="px-4 pb-4 space-y-3 text-xs">
              <p className="text-ink-500 leading-relaxed">
                This check reads saved sources. It can restore omitted text and recover explicit dates. Quarterly statistics remain background material.
              </p>
              {quality.repaired_events > 0 && (
                <p className="text-emerald-700 flex gap-2 items-center">
                  <CheckCircle2 size={14} />
                  {quality.repaired_events} older records now have dates supported by their sources.
                </p>
              )}
              {quality.needs_review?.map((e: any) => (
                <div
                  key={e.timeline_id}
                  className="rounded-lg bg-amber-50/70 p-3"
                >
                  <p className="text-xs text-amber-800">
                    {e.timestamp} · {dateReason(e.reason)}
                  </p>
                  <p className="text-ink-600 mt-1 leading-relaxed">
                    {e.description}
                  </p>
                </div>
              ))}
              {quality.extraction_plan?.plan && (
                <p className="text-xs text-ink-400">
                  Last extraction used{quality.extraction_plan.plan.chunk_chars} characters per chunk
                  · concurrency {quality.extraction_plan.plan.concurrency}
                </p>
              )}
            </div>
          </details>
        )}
        {manifest?.requires_reanalysis && (
          <div className="rounded-xl border border-amber-200/70 bg-[#fffcf3] p-3.5 text-xs text-amber-800 leading-relaxed">
            <p>You can browse the saved analysis. Synchronize the complete profiles before starting a new experiment.</p>
            {onReanalyze && (
              <button
                disabled={disabled || busy}
                onClick={onReanalyze}
                className="mt-2 text-xs font-medium underline underline-offset-4 disabled:opacity-40"
              >
                Reanalyze using compatible cached data
              </button>
            )}
          </div>
        )}
        <CaseGraphExplorer
          key={caseId}
          apiBase={apiBase}
          caseId={caseId}
          onOpenProfile={onOpenProfile}
          historyFocus={historyFocus}
          revision={revision}
        />
      </div>
    </div>
  );
}
