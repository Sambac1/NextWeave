import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Loader2, Server } from "lucide-react";
import { normalizeBackendUrl } from "@/config/backend";

export default function BackendConnection({ value, onSave, disabled }: {
  value: string;
  onSave: (value: string) => void;
  disabled: boolean;
}) {
  const [draft, setDraft] = useState(value);
  const [status, setStatus] = useState("");
  const [checking, setChecking] = useState(false);
  const [connected, setConnected] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => { setDraft(value); }, [value]);
  useEffect(() => () => controller.current?.abort(), []);

  async function saveAndTest(event: React.FormEvent) {
    event.preventDefault();
    setConnected(false);
    let url: string;
    try {
      url = normalizeBackendUrl(draft);
      if (!url) throw new Error("Enter the backend URL.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "The backend URL is invalid.");
      return;
    }
    setChecking(true);
    setStatus("Testing the connection…");
    controller.current = new AbortController();
    const timeout = window.setTimeout(() => controller.current?.abort(), 10000);
    try {
      const response = await fetch(`${url.replace(/\/api$/, "")}/health`, {
        signal: controller.current.signal,
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);
      const health = await response.json();
      if (health.status !== "ok") throw new Error("The service did not return the expected health status.");
      onSave(url);
      setDraft(url);
      setConnected(true);
      setStatus("Connected and saved.");
    } catch (error) {
      const detail = error instanceof Error && error.name !== "AbortError" ? error.message : "Connection timed out.";
      setStatus(`Could not connect. ${detail}`);
    } finally {
      clearTimeout(timeout);
      setChecking(false);
    }
  }

  return (
    <section className="px-4 sm:px-6 py-5 border-b border-paper-300 bg-white shrink-0">
      <form className="max-w-4xl mx-auto space-y-3" onSubmit={saveAndTest}>
        <h2 className="flex items-center gap-2 text-sm font-semibold text-ink-800"><Server className="w-4 h-4 text-signal-blue" />Backend Connection</h2>
        <p className="text-xs text-ink-500 leading-relaxed">Use an HTTPS backend URL for hosted sites. For local development, use http://localhost:8000.</p>
        <label htmlFor="backend-url" className="block text-xs text-ink-600">Backend URL (/api is added automatically)</label>
        <div className="flex flex-col sm:flex-row gap-2">
          <input id="backend-url" type="url" value={draft} onChange={(event) => { setDraft(event.target.value); setStatus(""); setConnected(false); }} placeholder="https://your-backend.example.com" disabled={checking || disabled} className="min-w-0 flex-1 rounded-lg px-3 py-2 text-sm font-mono bg-paper-50 border border-paper-300 disabled:opacity-50" />
          <button type="submit" disabled={checking || disabled} className="inline-flex justify-center items-center gap-2 rounded-lg px-4 py-2 bg-signal-blue text-white text-xs disabled:opacity-50">
            {checking ? <Loader2 className="w-4 h-4 animate-spin" /> : connected ? <CheckCircle2 className="w-4 h-4" /> : null}Save and Test
          </button>
        </div>
        {disabled && <p className="text-xs text-ink-500">Reload before switching the backend during an analysis.</p>}
        <p role="status" aria-live="polite" className={`text-xs leading-relaxed ${connected ? "text-green-700" : "text-ink-500"}`}>{status}</p>
        <details className="space-y-2 text-xs text-ink-600">
          <summary className="cursor-pointer font-medium text-signal-blue">Local Setup Instructions</summary>
          <p>From the project folder, install the backend dependencies in your Python environment and start the server.</p>
          <pre className="overflow-x-auto rounded-lg bg-paper-50 p-3">{`cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`}</pre>
          <p>Then enter http://localhost:8000 above and select Save and Test.</p>
        </details>
      </form>
    </section>
  );
}
