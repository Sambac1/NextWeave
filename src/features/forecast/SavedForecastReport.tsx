import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import { readBackendUrl } from "@/config/backend";
import { apiRequest, type ExperimentStatus } from "@/services/experimentService";
import ForecastReport from "./ForecastReport";

export default function SavedForecastReport({ runId }: { runId: string }) {
  const [status, setStatus] = useState<ExperimentStatus>();
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const value: ExperimentStatus = await apiRequest(readBackendUrl(), `/experiments/${encodeURIComponent(runId)}`);
        if (!active) return;
        if (!value.question_forecast) throw new Error("This run does not have a question forecast report.");
        setStatus(value);
        if (value.status === "queued" || value.status === "running") timer = setTimeout(load, 1500);
      } catch (failure) {
        if (active) setError(failure instanceof Error ? failure.message : "The saved report could not be loaded.");
      }
    }
    void load();
    return () => { active = false; clearTimeout(timer); };
  }, [runId, retry]);

  return (
    <main className="min-h-screen bg-paper-100 px-4 py-8 text-ink-800 sm:px-8 sm:py-12">
      <div className="mx-auto max-w-3xl space-y-5">
        <nav aria-label="Report navigation" className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-paper-300 pb-3 text-sm">
          <Link href="/" className="inline-flex min-h-11 items-center gap-2 font-semibold text-ink-900 hover:text-brand-800">
            <Image src="/brand/nextweave-icon.png" width={30} height={30} alt="" /> NextWeave
          </Link>
          {status?.event_id && <Link href={`/?analysis=${encodeURIComponent(status.event_id)}`} className="inline-flex min-h-11 items-center font-medium text-brand-800 underline underline-offset-4">Open analysis</Link>}
        </nav>
        {!status?.report && <h1 className="text-2xl font-semibold leading-tight text-ink-900">Forecast report</h1>}
        {error ? <div role="alert" className="space-y-3 rounded-lg border border-signal-red/30 bg-white p-4 text-sm">
          <p className="text-signal-red">{error}</p>
          <button type="button" className="font-semibold text-brand-800 underline" onClick={() => { setError(""); setRetry(value => value + 1); }}>Try again</button>
        </div> : status?.report ? <ForecastReport report={status.report} standalone /> :
          <p role="status" className="text-sm text-ink-600">
            {!status ? "Loading report…" : status.status === "queued" || status.status === "running"
              ? "Forecast in progress. The report will appear here."
              : status.error || "No report was saved for this forecast."}
          </p>}
        <p className="break-words text-xs text-ink-500">Run ID {runId}</p>
      </div>
    </main>
  );
}
