import { useEffect, useState } from "react";
import { Trash2, RotateCcw } from "lucide-react";
import { apiRequest } from "@/services/experimentService";

type SavedCase = { id: string; title: string; has_actors?: boolean; resume_available?: boolean };
type TrashEntry = { trash_id: string; case_id: string; title: string; deleted_at: string; status: "deleted" | "purging" };
type PendingAction = { type: "recycle"; item: SavedCase } | { type: "purge"; item: TrashEntry };

export default function SavedCases({ apiBase, onOpen }: { apiBase: string; onOpen: (item: any) => void }) {
  const [cases, setCases] = useState<SavedCase[]>([]), [trash, setTrash] = useState<TrashEntry[]>([]);
  const [pending, setPending] = useState<PendingAction | null>(null), [busy, setBusy] = useState(false);
  const [error, setError] = useState(""), [message, setMessage] = useState("");
  async function refresh() {
    const [saved, recycled] = await Promise.all([
      apiRequest(apiBase, "/knowledge/cases"), apiRequest(apiBase, "/knowledge/trash"),
    ]);
    setCases(saved.cases); setTrash(recycled.entries);
  }
  useEffect(() => { if (apiBase) refresh().catch(() => {}); }, [apiBase]);
  async function remove() {
    if (!pending || pending.type !== "recycle") return;
    const item = pending.item;
    setBusy(true); setError(""); setMessage("");
    try {
      await apiRequest(apiBase, `/knowledge/cases/${item.id}`, undefined, "DELETE");
      clearSession(item.id);
      setMessage(`Moved “${item.title}” to Trash.`);
      setPending(null); await refresh();
    } catch (e: any) { setError(e.message); }
    finally { setBusy(false); }
  }
  function clearSession(caseId: string) {
    try {
      const session = JSON.parse(localStorage.getItem("analysis_session") || "null");
      if (session?.id === caseId) localStorage.removeItem("analysis_session");
    } catch { /* A stale browser session must not turn a successful deletion into an error. */ }
  }
  async function purge() {
    if (!pending || pending.type !== "purge") return;
    const item = pending.item;
    setBusy(true); setError(""); setMessage("");
    try {
      await apiRequest(apiBase, `/knowledge/trash/${item.trash_id}`, undefined, "DELETE");
      clearSession(item.case_id);
      setTrash((items) => items.filter((entry) => entry.trash_id !== item.trash_id));
      setPending(null);
      setMessage(`Permanently deleted “${item.title}”.`);
      await refresh();
    } catch (e: any) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function restore(item: TrashEntry) {
    setBusy(true); setError(""); setMessage("");
    try {
      await apiRequest(apiBase, `/knowledge/trash/${item.trash_id}/restore`, {});
      setMessage(`Restored “${item.title}”.`); await refresh();
    } catch (e: any) { setError(e.message); }
    finally { setBusy(false); }
  }
  if (!cases.length && !trash.length && !message && !error) return null;
  return <div className="saved-analysis-list mt-6 space-y-4">
    <details open className="bg-white border border-paper-200 rounded-xl p-4 text-sm">
      <summary className="cursor-pointer text-ink-600">Saved Analyses ({cases.length})</summary>
      <div className="mt-3 space-y-2">{cases.map((item) => <div key={item.id} className="saved-analysis-row flex items-center gap-2 px-1">
        <button disabled={busy} onClick={() => onOpen(item)} className="flex-1 min-w-0 text-left px-1 py-2 text-xs hover:text-signal-blue"><span className="block font-medium">{item.title}</span><span className="text-ink-400">{item.resume_available ? "Profile generation can resume" : item.has_actors ? "Historical knowledge and simulations available" : "Sources saved and ready for analysis"}</span></button>
        <button disabled={busy} aria-label={`Move ${item.title} to Trash`} title="Move to Trash" onClick={() => { setPending({ type: "recycle", item }); setError(""); setMessage(""); }} className="shrink-0 rounded-lg p-2 text-ink-400 hover:bg-red-50 hover:text-red-700 disabled:opacity-40"><Trash2 className="w-4 h-4" /></button>
      </div>)}</div>
      {!cases.length && <p className="text-xs text-ink-400 pt-3">No saved analyses.</p>}
    </details>
    {pending?.type === "recycle" && <div role="alertdialog" aria-modal="false" aria-label="Confirm move to Trash" className="border border-amber-200 bg-amber-50 rounded-xl p-4 text-xs space-y-3">
      <p className="font-semibold">Move “{pending.item.title}” to Trash?</p>
      <p className="leading-relaxed text-ink-500">Move the analysis, its knowledge index, and related experiments out of the workspace. You can restore them from Trash. Stop active tasks before continuing.</p>
      <div className="flex gap-2"><button disabled={busy} onClick={remove} className="bg-red-700 text-white rounded-lg px-3 py-2 disabled:opacity-40">{busy ? "Moving…" : "Move to Trash"}</button><button disabled={busy} onClick={() => setPending(null)} className="border rounded-lg px-3 py-2">Cancel</button></div>
    </div>}
    {!!trash.length && <details className="bg-white border border-paper-200 rounded-xl p-4 text-xs"><summary className="cursor-pointer text-ink-500">Trash ({trash.length})</summary><div className="mt-3 space-y-2">{trash.map((item) => <div key={item.trash_id} className="recycle-bin-row flex flex-wrap items-center gap-3 p-3">
      <div className="min-w-0 flex-1 basis-40"><p className="break-words font-medium">{item.title}</p><p className="mt-1 text-ink-400">{item.status === "purging" ? "Deletion is incomplete. Try permanently deleting the analysis again." : `Moved on ${new Date(item.deleted_at).toLocaleString("en-US")}`}</p></div>
      <button disabled={busy || item.status === "purging"} aria-label={`Restore ${item.title}`} onClick={() => { setPending(null); restore(item); }} className="flex items-center gap-1 rounded-lg px-2 py-2 text-signal-blue hover:bg-brand-50 disabled:opacity-40"><RotateCcw className="w-3 h-3" />Restore</button>
      <button disabled={busy} aria-label={`Permanently delete ${item.title}`} onClick={() => { setPending({ type: "purge", item }); setError(""); setMessage(""); }} className="flex items-center gap-1 rounded-lg px-2 py-2 text-red-700 hover:bg-red-50 disabled:opacity-40"><Trash2 className="w-3 h-3" />Delete Permanently</button>
    </div>)}</div></details>}
    {pending?.type === "purge" && <div role="alertdialog" aria-modal="false" aria-label="Confirm permanent deletion" className="border border-red-200 bg-red-50 rounded-xl p-4 text-xs space-y-3">
      <p className="font-semibold text-red-800">Permanently delete “{pending.item.title}”?</p>
      <p className="leading-relaxed text-ink-600">This permanently deletes the analysis, its source files, extraction cache, actor profiles, knowledge versions, and simulations. Sources referenced by other analyses remain available.</p>
      <div className="flex gap-2"><button disabled={busy} onClick={purge} className="bg-red-700 text-white rounded-lg px-3 py-2 disabled:opacity-40">{busy ? "Deleting…" : "Confirm permanent deletion"}</button><button disabled={busy} onClick={() => setPending(null)} className="border rounded-lg px-3 py-2">Cancel</button></div>
    </div>}
    {error && <p role="alert" className="text-xs text-red-700">{error}</p>}
    {message && <p role="status" className="text-xs text-green-700">{message}</p>}
  </div>;
}
