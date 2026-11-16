export interface ScrapedArticle {
  title: string;
  url: string;
  content?: string;
  summary?: string;
  domain: string;
  error?: string;
}

export interface KeyEntity {
  name: string;
  type: string;
  description: string;
}

export interface AgentSituationView {
  topic: string;
  assessment: string;
  confidence: number | null;
  timeline_refs: string[];
}

export interface AgentActorModel {
  agent_name: string;
  assessment: string;
  confidence: number | null;
  timeline_refs: string[];
}

export interface AgentBeliefs {
  situation_views: AgentSituationView[];
  actor_models: AgentActorModel[];
}

export interface AgentDesire {
  desire_id: string;
  description: string;
  priority: number | null;
  status: string;
}

export interface AgentCapability {
  name: string;
  level: number;
}

export interface AgentDecisionPolicy {
  risk_tolerance: number;
  compromise_tendency: number;
  response_speed: number;
}

export interface BehaviorStatement {
  description: string;
  dimension?: string;
  scope?: string;
  counterpart?: string;
  conditions?: string;
  timeline_refs?: string[];
  status?: string;
}
export interface BehavioralModel {
  contextual_roles: BehaviorStatement[];
  capabilities: BehaviorStatement[];
  decision_patterns: BehaviorStatement[];
  objectives: BehaviorStatement[];
  calibration: string;
}

export interface Agent {
  agent_id: string;
  name: string;
  aliases: string[];
  agent_type: string;
  decision_subject: string;
  role: string;
  forecast_role?: "decision_actor" | "evidence_participant";
  decision_core?: import("@/components/DecisionDossier").DecisionCore;
  current_plan?: import("@/components/DecisionDossier").PlanState | null;
  behavioral_model?: BehavioralModel;
  relevant_timeline_refs?: string[];
  title?: string;
  affiliation?: string;
  description: string;
  relationships: any[];
  historical_actions: Array<Record<string, unknown>>;
  beliefs: AgentBeliefs;
  desires: AgentDesire[];
  capabilities: AgentCapability[];
  constraints: string[];
  decision_policy: AgentDecisionPolicy;
  public_state?: AgentPublicState;
  activation?: Record<string, unknown>;
}


export interface AgentSubordinate {
  agent_id: string;
  name: string;
  entity_type?: string;
  role_in_entity?: string;
  title?: string;
  affiliation?: string;
  capability?: string;
  style?: string;
}

/**
 * Fields the layered-figure stage carries on a principal beyond the forecast's
 * own model. `profile_kind` picks which semantic shape the body uses, and
 * `semantic` holds the narrative profile the panel renders.
 */
export interface AgentPublicState {
  profile_kind?: string;
  knowledge_boundary?: string;
  stance?: string;
  summary?: string;
  semantic?: Record<string, any>;
  source_profile?: Record<string, any>;
  subordinates?: AgentSubordinate[];
  [key: string]: unknown;
}

export interface GraphData {
  nodes: any[];
  edges: any[];
}
