"use client";

export interface DossierItem {
  id?: string; description: string; conditions?: string; timeline_refs?: string[];
  provenance?: string; priority?: string; tradeoff?: string; success_condition?: string;
  value?: number | null; unit?: string; kind?: string;
  holder?:string; valid_from?:string; valid_until?:string; review_condition?:string;
  source_refs?:{evidence_id:string;quote:string;url:string;title:string;section?:string}[];
}
export interface DecisionCore {
  version: string; origin: string; as_of?: string; mandate: DossierItem[]; goals: DossierItem[];
  means: DossierItem[]; resources: DossierItem[]; limits: DossierItem[]; decision_rules: DossierItem[];
  initial_intentions?: DossierItem[];
  decision_scope?: string; evidence_gaps?:string[]; review_notes?:string[]; simulation_eligible?: boolean | null;
  seat?: { owner_name: string; represents?: string; entity_type?: string | null } | null;
}
export interface PlanState {
  description: string; goal_ids?: string[]; status: string; review_when?: string;
  change_reason?: string; updated_round?: number;
}
type Judgment = { assessment?: string; timeline_refs?: string[] };
type BeliefState = {
  situation_views?: (Judgment & { topic?: string })[];
  actor_models?: (Judgment & { agent_name?: string })[];
};
const provenance: Record<string,string> = {source_statement:"Source statement",inferred:"Inferred",assumption:"Scenario assumption",legacy_unreviewed:"Legacy record · unverified"};
const priority: Record<string,string> = {high:"High",medium:"Medium",low:"Low",unknown:"Priority unknown"};
const status: Record<string,string> = {active:"Active",satisfied:"Satisfied",completed:"Completed",blocked:"Blocked",abandoned:"Abandoned"};

export default function DecisionDossier({core,beliefs,goals,plan,onEvidence,legacyBeliefs,stateLabel,compact,hideDecisionState}: {
  core: DecisionCore; beliefs?: BeliefState;
  goals?: Array<{desire_id:string;description:string;status:string}>;
  plan?: PlanState | null; onEvidence?: (id:string)=>void; legacyBeliefs?: boolean; stateLabel?: string; compact?: boolean; hideDecisionState?: boolean;
}) {
  const seen = new Set<string>();
  const items = (rows?:DossierItem[]) => (rows || []).filter(item => {
    const key=item.description.replace(/\s+/g,"").toLowerCase();
    if(seen.has(key)) return false; seen.add(key); return true;
  });
  const definitions = new Map((core.goals||[]).map(g=>[g.id,g]));
  const desires = items(goals ? goals.map(g=>({...definitions.get(g.desire_id),id:g.desire_id,description:g.description})) : core.goals);
  const groups = [
    {label:"Decision authority", rows:items(core.mandate)},
    {label:"Available means", rows:items(core.means)},
    {label:"Resources and capacity", rows:items(core.resources)},
    {label:"Constraints and prerequisites", rows:items((core.limits||[]).filter(v=>v.provenance!=="legacy_unreviewed"))},
    {label:"Conditional decision rules", rows:items(core.decision_rules)},
  ];
  const state = new Map((goals||[]).map(g=>[g.desire_id,g.status]));
  const judgmentKeys = new Set<string>();
  const judgments=[...(beliefs?.situation_views||[]),...(beliefs?.actor_models||[])]
    .filter(v=>{
      const key=String(v.assessment||"").trim();
      if(!key || judgmentKeys.has(key)) return false;
      judgmentKeys.add(key); return true;
    });
  const evidence = (refs?:string[]) => !!refs?.length && <details className="mt-2 text-xs text-ink-400">
    <summary className="cursor-pointer py-1 text-brand-700">Evidence · {refs.length} linked records</summary>
    <div className="mt-1 flex flex-wrap gap-2">{refs.map((ref,i)=>onEvidence ? <button key={ref} onClick={()=>onEvidence(ref)} title={ref} className="rounded-md bg-brand-50 px-2 py-1 text-brand-800">View evidence {i+1} ↗</button> : <span key={ref} className="break-all">{ref}</span>)}</div>
  </details>;
  if (compact) {
    const unique = new Map<string, { assessment: string; labels: string[]; refs: string[] }>();
    const groupJudgments = (rows: (Judgment & { label?: string })[], fallback: string) => {
      const group: Array<{ assessment: string; labels: string[]; refs: string[] }> = [];
      for (const row of rows) {
        const assessment = row.assessment?.trim();
        if (!assessment) continue;
        const key = assessment.replace(/\s+/g, " ").replace(/[。.!?！？]+$/, "");
        const existing = unique.get(key);
        if (existing) {
          if (row.label && !existing.labels.includes(row.label)) existing.labels.push(row.label);
          existing.refs = Array.from(new Set([...existing.refs, ...(row.timeline_refs || [])]));
          continue;
        }
        const item = { assessment, labels: [row.label || fallback], refs: Array.from(new Set(row.timeline_refs || [])) };
        unique.set(key, item);
        group.push(item);
      }
      return group;
    };
    const situations = groupJudgments((beliefs?.situation_views || []).map(row => ({ ...row, label: row.topic })), "Situational assessments");
    const counterparts = groupJudgments((beliefs?.actor_models || []).map(row => ({ ...row, label: row.agent_name })), "Assessments of other actors");

    return <div aria-label="Actor BDI summary" className="divide-y divide-paper-100 rounded-xl border border-paper-300 bg-white text-sm">
      <section className="space-y-2 px-4 py-3">
        <h5 className="font-semibold text-brand-800">I · Current intentions</h5>
        {plan?.description ? <>
          {plan.description.length > 220 ? <details className="group">
            <summary className="py-2 text-xs text-brand-800">
              View full action plan
              <span className="mt-2 line-clamp-3 text-sm leading-relaxed text-ink-700 group-open:hidden">{plan.description}</span>
            </summary>
            <p className="whitespace-pre-wrap leading-relaxed">{plan.description}</p>
          </details> : <p className="leading-relaxed">{plan.description}</p>}
          {plan.status && <p className="text-xs text-ink-500">{status[plan.status] || plan.status}</p>}
          {(plan.review_when || plan.change_reason) && <details className="text-xs text-ink-500">
            <summary className="py-2">Review triggers and reasons for changes</summary>
            {plan.review_when && <p className="mt-2">Reassess when {plan.review_when}</p>}
            {plan.change_reason && <p className="mt-2">Reason for change · {plan.change_reason}</p>}
          </details>}
        </> : <p className="text-xs text-ink-500">No action plan is recorded.</p>}
      </section>
      <details className="px-4 py-2">
        <summary className="py-2 font-semibold text-brand-800">D · Desires · {desires.length} goals</summary>
        {desires.length ? <ul className="space-y-3 pb-2 pl-3 text-xs">{desires.map((goal, index) => {
          const goalStatus = goal.id ? state.get(goal.id) : undefined;
          return <li key={goal.id || index} className="leading-relaxed">
            {goalStatus && <span className="mr-2 font-medium text-brand-800">{status[goalStatus] || goalStatus}</span>}
            {goal.description}
            {goal.tradeoff && <p className="mt-1 text-ink-500">Trade-off · {goal.tradeoff}</p>}
            {goal.success_condition && <p className="mt-1 text-ink-500">Success condition · {goal.success_condition}</p>}
            {evidence(goal.timeline_refs)}
          </li>;
        })}</ul> : <p className="pb-2 text-xs text-ink-500">No goals are recorded.</p>}
      </details>
      <details className="px-4 py-2">
        <summary className="py-2 font-semibold text-brand-800">B · Beliefs · {unique.size} assessments</summary>
        {unique.size ? <div className="space-y-3 pb-2 pl-3 text-xs">
          {[{ label: "Situational assessments", rows: situations }, { label: "Assessments of other actors", rows: counterparts }]
            .filter(group => group.rows.length > 0).map(group => <section key={group.label}>
              <h5 className="font-medium text-ink-500">{group.label}</h5>
              {group.rows.map(item => <details key={item.assessment}>
                <summary className="py-2 text-brand-800">{item.labels.join(" / ")}</summary>
                <p className="pl-3 leading-relaxed">{item.assessment}</p>
                {item.refs.length > 0 && <div className="pl-3">{evidence(item.refs)}</div>}
              </details>)}
            </section>)}
        </div> : <p className="pb-2 text-xs text-ink-500">No belief assessments are recorded.</p>}
      </details>
    </div>;
  }
  return <div className="decision-dossier space-y-6 text-sm text-ink-700" aria-label="Decision profile">
    <div className="flex flex-wrap items-center gap-2 text-xs text-ink-500">
      <span className="rounded-full bg-paper-100 px-2.5 py-1">{stateLabel || (plan ? `Trajectory state · Round ${plan.updated_round||0}` : "Historical baseline")}</span>
      {core.as_of && <span>Evidence cutoff {core.as_of}</span>}
      {core.seat?.represents && <span>Represents {core.seat.represents}</span>}
    </div>
    {core.decision_scope && <p className="text-sm leading-relaxed">{core.decision_scope}</p>}
    {core.origin === "legacy_projection" && <p className="rounded-xl border border-amber-100 bg-amber-50/60 px-3 py-2 text-xs leading-relaxed text-amber-900">This legacy profile shows the available evidence. Historical activity alone provides limited evidence of lasting capabilities.</p>}
    {!hideDecisionState && <section aria-label="BDI decision state" className="dossier-state border bg-white overflow-hidden">
      <div className="border-b border-paper-100 px-4 py-3"><h3 className="text-sm font-semibold text-ink-800">BDI decision state</h3></div>
      <div className="divide-y divide-paper-100">
        <section className="px-4 py-3" aria-label="Beliefs"><h4 className="mb-2 text-xs font-semibold text-brand-800">B · Beliefs</h4>
          {judgments.length ? <div className="space-y-3">{judgments.map((b,i)=><article key={i} className="text-xs leading-relaxed"><p>{b.assessment}</p>{legacyBeliefs && <p className="mt-1 text-xs text-amber-700">Legacy model inference · unverified</p>}{evidence(b.timeline_refs)}</article>)}</div> : <p className="text-xs text-ink-400">No supported belief assessments are available.</p>}
        </section>
        <section className="px-4 py-3" aria-label="Desires"><h4 className="mb-2 text-xs font-semibold text-brand-800">D · Desires</h4>
          {desires.length ? <div className="space-y-3">{desires.map((g,i)=><article key={g.id||i} className="text-xs leading-relaxed">
            <p>{g.description}</p>
            {g.tradeoff && <p className="mt-1 text-ink-500">Trade-off · {g.tradeoff}</p>}
            {g.success_condition && <p className="mt-1 text-ink-500">Success condition · {g.success_condition}</p>}
            <div className="mt-1 flex flex-wrap gap-x-3 text-xs text-ink-400">
              {g.priority && g.priority!=="unknown" && <span>{priority[g.priority]||g.priority}</span>}
              {g.id && state.has(g.id) && <span>{status[state.get(g.id)!]||state.get(g.id)}</span>}
              {g.provenance && core.origin!=="legacy_projection" && <span>{provenance[g.provenance]||g.provenance}</span>}
            </div>{evidence(g.timeline_refs)}
          </article>)}</div> : <p className="text-xs text-ink-400">The actor's goals are unknown.</p>}
        </section>
        <section className="px-4 py-3" aria-label="Intentions"><h4 className="mb-2 text-xs font-semibold text-brand-800">I · Intentions</h4>
          {plan ? <article className="text-xs leading-relaxed"><p>{plan.description}</p><p className="mt-1 text-xs text-ink-400">{status[plan.status]||plan.status}</p>
            {plan.review_when && <p className="mt-2 text-ink-500">Reassess when {plan.review_when}</p>}
            {plan.change_reason && <p className="mt-1 text-ink-500">Reason for change · {plan.change_reason}</p>}
            {!plan.updated_round && core.initial_intentions?.[0]?.source_refs?.map(ref => (
              /^https?:\/\//i.test(ref.url) && <a key={ref.evidence_id} className="mt-2 block text-brand-800 underline" href={ref.url}
                target="_blank" rel="noreferrer">{ref.title || "Open source"}</a>
            ))}
          </article> : <p className="text-xs text-ink-400">No supported starting plan is documented.</p>}
        </section>
      </div>
    </section>}
    {groups.filter(g=>g.rows.length).map(group=><section key={group.label} className="dossier-facts">
      <h4 className="mb-2 text-xs font-semibold tracking-wide text-ink-500">{group.label}</h4>
      <div className="space-y-2">{group.rows.map((item,i)=><article key={item.id||i} className="dossier-fact bg-white min-w-0 break-words">
        <p className="text-xs leading-relaxed">{item.description}</p>
        {item.holder && <p className="mt-1.5 text-xs text-ink-500">Authority or resource holder · {item.holder}</p>}
        {(item.valid_from || item.valid_until) && <p className="mt-1 text-xs text-ink-500">Valid from {item.valid_from || "an unknown start"} to {item.valid_until || "an unknown end"}</p>}
        {item.review_condition && <p className="mt-1 text-xs text-ink-500">Review trigger · {item.review_condition}</p>}
        {item.conditions && <p className="mt-1.5 text-xs text-ink-500">Applies when {item.conditions}</p>}
        {item.tradeoff && <p className="mt-1.5 text-xs text-ink-500">Trade-off · {item.tradeoff}</p>}
        {item.success_condition && <p className="mt-1.5 text-xs text-ink-500">Success condition · {item.success_condition}</p>}
        {group.label === "Resources and capacity" && <p className="mt-1.5 text-xs font-medium">Baseline quantity · {item.value == null ? "Unknown" : `${item.value} ${item.unit||""}`}</p>}
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-400">
          {item.provenance && <span>{provenance[item.provenance]||item.provenance}</span>}
          {item.priority && <span>{priority[item.priority]||item.priority}</span>}
          {item.id && state.has(item.id) && <span>{status[state.get(item.id)!]||state.get(item.id)}</span>}
          {!!item.timeline_refs?.length && <details className="w-full"><summary className="cursor-pointer py-1 text-brand-700">Evidence · {item.timeline_refs.length} linked records</summary>
            <div className="mt-1 flex flex-wrap gap-2">{item.timeline_refs.map((ref,j)=>onEvidence ? <button key={ref} onClick={()=>onEvidence(ref)} title={ref} className="rounded-md bg-brand-50 px-2 py-1 text-brand-800">View evidence {j+1} ↗</button> : <span key={ref} className="break-all">{ref}</span>)}</div>
          </details>}
          {!!item.source_refs?.length && <details className="w-full"><summary className="cursor-pointer py-1 text-brand-700">Supporting sources · {item.source_refs.length} references</summary>
            {item.source_refs.map(ref=><div key={ref.evidence_id} className="mt-2 border-l-2 border-brand-200 pl-3"><p className="text-ink-600">{ref.quote}</p>
              {/^https?:\/\//i.test(ref.url) && <a className="mt-1 inline-block text-brand-700 underline" href={ref.url} target="_blank" rel="noopener noreferrer">{ref.title || "Open source"}</a>}</div>)}
          </details>}
        </div>
      </article>)}</div>
    </section>)}
    {core.simulation_eligible === false ? (
      <p className="text-xs text-ink-500">Evidence is insufficient for an independent decision role. The profile remains available for reference.</p>
    ) : !core.means?.length && !core.mandate?.length && (
      <p className="text-xs text-ink-400">Decision authority and available means remain unknown.</p>
    )}
    {(!!core.evidence_gaps?.length || !!core.review_notes?.length) && <details className="border-t border-paper-300 pt-3 text-xs text-ink-500"><summary>Evidence and profile review</summary>
      {Array.from(new Set([...(core.evidence_gaps||[]),...(core.review_notes||[])])).map(note=><p key={note} className="mt-2 leading-relaxed">{note}</p>)}
    </details>}
  </div>;
}
