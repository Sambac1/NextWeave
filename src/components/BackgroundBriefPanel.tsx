"use client";
import {useEffect,useState} from "react";
import {Loader2,RefreshCw} from "lucide-react";
import {apiRequest} from "@/services/experimentService";

type Reference={source_index:number;sentence_id:string;title:string;url:string;quote:string;retrieved_at?:string};
type BriefDocument={version?:string;as_of?:string;evidence_gaps?:string[];text:string;
  paragraphs:{heading:string;text:string;references:Reference[]}[];additional_context:string};
function safeUrl(url:string) {try{const parsed=new URL(url);return ["http:","https:"].includes(parsed.protocol)?parsed.href:undefined;}catch{return undefined;}}

export default function BackgroundBriefPanel({text,apiBase,caseId,disabled,forecast,onRegenerate}:{
  text:string;apiBase:string;caseId:string|null;disabled?:boolean;forecast?:boolean;onRegenerate:()=>Promise<void>;
}) {
  const [document,setDocument]=useState<BriefDocument|null>(null);
  const [busy,setBusy]=useState(false),[error,setError]=useState("");
  useEffect(()=>{
    let cancelled=false;
    setDocument(null);setError("");
    if(apiBase && caseId) apiRequest(apiBase,`/knowledge/${caseId}/background`).then(result=>{
      if(!cancelled && result.document) setDocument(result.document);
    }).catch(()=>{if (!cancelled && forecast) setError("The saved briefing could not be loaded.");});
    return ()=>{cancelled=true;};
  },[apiBase,caseId,disabled,forecast,text]);
  async function regenerate(){setBusy(true);setError("");try{await onRegenerate();}catch(e){setError(e instanceof Error?e.message:"Could not generate the background brief. Try again.");}finally{setBusy(false);}}
  const references = Array.from(new Map((document?.paragraphs ?? []).flatMap(part => part.references)
    .map(ref => [`${ref.source_index}:${ref.sentence_id}`, ref] as const)).values());
  return <section aria-label="Background Brief" className="background-brief space-y-6">
    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
      <span className="text-ink-600">{document?.as_of
        ? `Evidence through ${new Date(document.as_of).toLocaleDateString("en-US", {timeZone:"UTC", dateStyle:"medium"})} UTC`
        : "Origins, decision-makers, and relevant developments"}</span>
      <button disabled={disabled || busy || !caseId || !apiBase} onClick={regenerate} className="flex items-center gap-1.5 text-signal-blue disabled:opacity-40">
        {busy?<Loader2 size={13} className="animate-spin"/>:<RefreshCw size={13}/>} {busy?"Updating…":"Regenerate Background"}
      </button>
    </div>
    {error && <p role="alert" className="text-xs text-red-700">{error}</p>}
    {document?.version === "forecast-background-preview-1" && <p className="text-sm text-ink-600">
      Source preview. The background paragraph is generated during actor preparation.
    </p>}
    {document ? <div className="max-w-prose space-y-3">
      <p className="text-sm leading-7 text-ink-700">
        {document.paragraphs.map(paragraph => paragraph.text.trim()).filter(Boolean).join(" ")}
      </p>
      {(references.length > 0 || !!document.evidence_gaps?.length || !!document.additional_context) &&
        <details className="text-xs text-ink-600">
          <summary className="cursor-pointer font-medium text-signal-blue">Sources and context</summary>
          <div className="brief-evidence mt-3 space-y-4 border-l-2">
            {references.map(ref => <div key={`${ref.source_index}:${ref.sentence_id}`}>
              <p className="font-medium text-ink-600">{ref.title}</p>
              <blockquote className="mt-1 leading-relaxed">{ref.quote}</blockquote>
              {safeUrl(ref.url) && <a className="mt-1 inline-block text-signal-blue underline" href={safeUrl(ref.url)} target="_blank" rel="noopener noreferrer">Open source</a>}
            </div>)}
            {!!document.evidence_gaps?.length && <div>
              <p className="font-semibold text-ink-700">Evidence gaps</p>
              <ul className="mt-1 list-disc space-y-1 pl-4">{document.evidence_gaps.map(gap => <li key={gap}>{gap}</li>)}</ul>
            </div>}
            {document.additional_context && <div>
              <p className="font-semibold text-ink-700">User-provided context</p>
              <p className="mt-1 whitespace-pre-wrap leading-relaxed">{document.additional_context}</p>
            </div>}
          </div>
        </details>}
    </div>
      : forecast && (!text || text.startsWith("Background Brief · Traceable Source Excerpts"))
        ? (!error && <p role="status" className="text-sm text-ink-600">Loading the briefing…</p>)
        : <p className="brief-plain-text text-ink-700 whitespace-pre-wrap">{text}</p>}
  </section>;
}
