"use client";
import {X,Users,History} from "lucide-react";
import type {Agent,AgentSubordinate} from "@/domain/types";
import DecisionDossier,{type DecisionCore} from "@/components/DecisionDossier";

export interface AgentProfile extends Omit<Partial<Agent>,"agent_id"|"name"|"description"|"role"|"historical_actions"> {
  agent_id:string;name:string;description:string;role:string;historical_actions:Array<Record<string,unknown>>;
}
export default function AgentProfilePanel({agent,onClose,onOpenHistory}: {
  agent:AgentProfile;allAgents:AgentProfile[];relationships?:any[];onClose:()=>void;onOpenHistory?:(eventId?:string)=>void;
}) {
  const referenceOnly = agent.forecast_role === "evidence_participant";
  const core:DecisionCore=agent.decision_core || {version:"decision-core-1",origin:"legacy_projection",mandate:[],
    goals:(agent.desires||[]).map(g=>({id:g.desire_id,description:g.description,priority:"unknown",provenance:"legacy_unreviewed"})),means:[],resources:[],
    limits:(agent.constraints||[]).map((d,i)=>({id:`l${i}`,description:d,provenance:"legacy_unreviewed"})),decision_rules:[]};
  const members:AgentSubordinate[]=agent.public_state?.subordinates||[];
  const legacy=core.origin === "legacy_projection";
  const runtime=!!agent.current_plan || Number(agent.activation?.last_information_update_round||0)>0;
  const beliefs=legacy && !runtime && agent.beliefs ? {
    situation_views:(agent.beliefs.situation_views||[]).filter(v=>v.assessment.trim()!==agent.description.trim()),
    actor_models:(agent.beliefs.actor_models||[]).filter(v=>v.agent_name!=="the other actors in this event"),
  } : agent.beliefs;
  const validRefs=new Set(agent.relevant_timeline_refs||[]);
  const supportedCapabilities=(agent.behavioral_model?.capabilities||[])
    .map(item=>({...item,timeline_refs:(item.timeline_refs||[]).filter(ref=>validRefs.has(ref))}))
    .filter(item=>item.description && item.timeline_refs.length);
  const identityDetails=[agent.description,agent.title,agent.affiliation]
    .filter((value,index,all)=>!!value && value!==agent.name && all.indexOf(value)===index);
  const displayedCore:DecisionCore={...core, means:[...(core.means||[]), ...supportedCapabilities.map((item,i)=>({
    id:`legacy-capability-${i}`, description:item.description, conditions:item.conditions,
    timeline_refs:item.timeline_refs, provenance:"legacy_unreviewed",
  }))]};
  return <div className="fixed inset-0 z-[70] flex items-center justify-center bg-paper-950/50 p-3 sm:p-6" onClick={onClose}>
    <section role="dialog" aria-modal="true" aria-label={`${agent.name} profile`} className="profile-dialog relative w-full max-h-[92dvh] overflow-y-auto shadow-elevated" onClick={e=>e.stopPropagation()}>
      <header className="sticky top-0 z-10 flex items-start justify-between gap-4 border-b border-paper-200 bg-white/95 p-5 backdrop-blur">
        <div className="min-w-0"><p className="text-xs font-medium text-ink-500">{referenceOnly ? "Evidence participant" : "Decision dossier"}</p><h2 className="mt-1 break-words text-xl font-semibold text-ink-800">{agent.name}</h2>
          </div>
        <button onClick={onClose} aria-label="Close actor profile" className="shrink-0 rounded-full p-2 text-ink-500 hover:bg-paper-100"><X size={18}/></button>
      </header>
      <div className="space-y-5 p-5">
        {!!identityDetails.length && <section className="profile-identity bg-white">
          <h3 className="text-xs font-semibold text-ink-700">Identity and responsibilities</h3>
          {identityDetails.map((detail,i)=><p key={i} className="mt-2 text-xs leading-relaxed text-ink-600">{detail}</p>)}
        </section>}
        {referenceOnly && <p className="text-sm text-ink-600">This participant provides historical context and does not make independent decisions in new forecasts.</p>}
        <DecisionDossier hideDecisionState={referenceOnly} core={displayedCore} beliefs={beliefs} goals={agent.desires} plan={agent.current_plan} legacyBeliefs={legacy && !runtime} onEvidence={onOpenHistory}/>
        {!!members.length && <details className="rounded-xl border border-paper-200 bg-white p-3"><summary className="cursor-pointer text-xs font-medium text-ink-600"><Users size={14} className="mr-2 inline"/>Members and delegates · {members.length}</summary>
          <div className="mt-3 space-y-2">{members.map(m=><p key={m.agent_id} className="text-xs text-ink-600"><b>{m.name}</b> · {m.role_in_entity||m.title||"Affiliation unverified"}</p>)}</div></details>}
        {onOpenHistory && <button onClick={()=>onOpenHistory()} className="flex w-full items-center gap-2 rounded-xl border border-paper-200 bg-white p-3 text-left text-xs text-brand-700 hover:bg-brand-50"><History size={14}/>View historical events and sources</button>}
      </div>
    </section>
  </div>;
}
