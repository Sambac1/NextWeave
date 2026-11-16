"""Scenario-neutral experiment contracts; categories are fixed before sampling."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal, Any, get_args

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from app.domain.forecast import SimulationRequest
from app.domain.common import LLMOptions
from app.domain.activities import ActivityStage
from app.domain.process import ProcessStateCode, ProcessTopic
from app.domain.context_limits import DEFAULT_CONTEXT_CHARS, MIN_CONTEXT_CHARS, MAX_CONTEXT_CHARS


class OutcomeCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(pattern=r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    value: str | float | bool


class StateInvariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: list[str] = Field(min_length=1, max_length=30)
    rule: Literal["range", "fixed", "nondecreasing", "conserved_sum"]
    minimum: float | None = None
    maximum: float | None = None


class OutcomeDefinition(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,50}$")
    label: str = Field(min_length=1, max_length=120)
    criterion: str = Field(min_length=1, max_length=1200)
    conditions: list[OutcomeCondition] = Field(default_factory=list, max_length=20)


class TerminalCategory(OutcomeDefinition):
    required_actor_ids: list[str] = Field(default_factory=list, max_length=8)
    evidence_stages: list[ActivityStage] = Field(
        default_factory=list,
        max_length=len(get_args(ActivityStage)),
        description="Optional filter. Select only the activity stages that support this terminal criterion.",
    )
    issue_state_codes: list[ProcessStateCode] = Field(default_factory=list, max_length=8)

    @field_validator("evidence_stages", mode="before")
    @classmethod
    def deduplicate_evidence_stages(cls, stages: Any) -> Any:
        if isinstance(stages, list) and all(isinstance(stage, str) for stage in stages):
            return list(dict.fromkeys(stages))
        return stages


class OutcomeDimension(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,50}$")
    label: str = Field(min_length=1, max_length=120)
    categories: list[TerminalCategory] = Field(min_length=2, max_length=8)
    observation_mode: Literal["legacy", "standing", "boundary", "mechanism", "unavailable"] = "legacy"
    observation_scope: str = Field(default="", max_length=420)
    required_for_sampling: bool = True
    issue_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,60}$")

    @model_validator(mode="after")
    def unique_categories(self):
        ids = [category.id for category in self.categories]
        if len(set(ids)) != len(ids) or "other" in ids:
            raise ValueError("Dimension category IDs must be unique; 'other' is reserved")
        if self.observation_mode != "legacy" and not self.observation_scope.strip():
            raise ValueError("A scoped observation is required for each measurement dimension")
        states = [state for category in self.categories for state in category.issue_state_codes]
        if states and (not self.issue_id or "unknown" in states or len(set(states)) != len(states)):
            raise ValueError("Issue evidence requires a linked issue and non-overlapping known state codes")
        if self.issue_id and self.observation_mode not in {"boundary", "legacy"}:
            raise ValueError("Issue evidence can only be used for boundary observations")
        if self.observation_mode == "mechanism":
            if not all(category.conditions for category in self.categories):
                raise ValueError("Mechanism outcomes require explicit conditions for each category")
        elif any(category.conditions for category in self.categories):
            raise ValueError("Outcome conditions require mechanism observation mode")
        return self


class EventQuestion(OutcomeDefinition):
    topic: str = Field(default="", max_length=120)
    milestone: ActivityStage | Literal[""] = ""
    coverage_status: Literal['modeled', 'limited'] = 'modeled'
    condition_coverage: Literal['boundary', 'cumulative_horizon'] = 'boundary'
    scope: Literal['during_horizon', 'terminal'] = 'during_horizon'
    required_actor_ids: list[str] = Field(default_factory=list, max_length=8)
    required_duration_days: float | None = Field(default=None, gt=0, le=3660)
    # Only an explicit caller may assume continuity between observed boundaries.
    assume_continuity: bool = False


class OutcomePlan(BaseModel):
    categories: list[OutcomeDefinition] = Field(default_factory=list, max_length=8)
    outcome_dimensions: list[OutcomeDimension] = Field(default_factory=list, max_length=3)
    event_questions: list[EventQuestion] = Field(default_factory=list, max_length=8)
    process_topics: list[ProcessTopic] = Field(default_factory=list, max_length=4)
    partition_basis: str = Field(default='', max_length=800)
    measurement_warnings: list[str] = Field(default_factory=list, max_length=30)
    partition_validation: Literal['program_rules', 'model_reviewed', 'unreviewed'] = 'unreviewed'


    @model_validator(mode="after")
    def validate_partition(self):
        ids = [dimension.id for dimension in self.outcome_dimensions]
        if len(ids) != len(set(ids)):
            raise ValueError("Outcome dimension IDs must be unique")
        if bool(self.categories) == bool(self.outcome_dimensions) or len(self.categories) == 1:
            raise ValueError("Provide dimensions or at least two legacy categories")
        return self


class OutcomePlanAudit(BaseModel):
    mutually_exclusive: StrictBool
    overlapping_category_ids: list[str] = Field(default_factory=list)
    reason: str = Field(max_length=1500, description="One to three short sentences; aim for at most 500 characters. Decisions and category IDs belong in their structured fields.")
    uncovered_question_ids: list[str] = Field(default_factory=list)

    @field_validator('reason', mode='before')
    @classmethod
    def bound_explanation(cls, value):
        # Explanation text is not the audit decision. Preserve the typed decision
        # and ID lists while avoiding a paid retry for verbose valid JSON.
        if isinstance(value, str) and len(value) > 1500:
            return value[:1499].rstrip() + '…'
        return value


class QuestionSimulation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_boundaries: list[str] = Field(min_length=2, max_length=13)
    resolution_rules: str = Field(min_length=1, max_length=20000)
    outcome_deadline: str = Field(min_length=1, max_length=80)
    as_of: str | None = Field(default=None, max_length=80)
    prompt_profile: Literal["standard", "compact"] = "standard"
    round_retrieval: bool = True
    bdi_updates: bool = True
    action_selection: Literal["weighted", "greedy"] = "weighted"
    terminal_branching: bool = False
    sequential_outcome: bool = False
    terminal_probability: Literal["replay", "forecast"] = "replay"
    terminal_head_branches: int = Field(default=2, ge=1, le=2)

    @model_validator(mode="after")
    def sequential_sampling_contract(self):
        if self.sequential_outcome and (self.terminal_branching or self.action_selection != "weighted"):
            raise ValueError("Sequential outlooks require weighted action sampling and one realized path per trajectory.")
        return self


BenchmarkSimulation = QuestionSimulation


class MainQuestionSimulation(QuestionSimulation):
    algorithm: Literal["evidence-gated-stratified-v1", "conditional-stratified-v2"]


class EnsembleRequest(SimulationRequest):
    sample_count: int = Field(default=8, ge=1, le=400)
    benchmark: BenchmarkSimulation | None = None
    question_simulation: MainQuestionSimulation | QuestionSimulation | None = None
    steps: int = Field(default=3, ge=1, le=12)
    sample_concurrency: int = Field(default=2, ge=1, le=4)
    call_concurrency: int = Field(default=4, ge=1, le=8)
    max_runtime_seconds: int | None = Field(default=900, ge=30, le=86400)
    max_model_calls: int | None = Field(default=1000, ge=1, le=50000)
    step_timeout_seconds: int = Field(default=180, ge=5, le=1800)
    step_unit: str = Field(default="month", pattern=r"^(auto|day|week|month|year)$")
    step_size: int = Field(default=1, ge=1, le=365, strict=True)
    planning_mode: Literal["auto", "rules"] = "auto"
    decision_policy: Literal["single", "weighted"] = "single"
    decision_seed: int | None = Field(default=None, ge=0, le=(1 << 53) - 1, strict=True)
    context_max_chars: int = Field(default=DEFAULT_CONTEXT_CHARS, ge=MIN_CONTEXT_CHARS, le=MAX_CONTEXT_CHARS)
    scenario_protocol: dict[str, Any] = Field(default_factory=dict)
    scenario_description: str = Field(default="", max_length=6000)
    initial_world_state: dict[str, Any] = Field(default_factory=dict)
    state_invariants: list[StateInvariant] = Field(default_factory=list, max_length=100)
    initial_private_information: dict[str, list[str]] = Field(default_factory=dict)
    as_of_month: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    start_date: str | None = Field(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    outcome_definitions: list[OutcomeDefinition] = Field(default_factory=list, max_length=8)
    outcome_dimensions: list[OutcomeDimension] = Field(default_factory=list, max_length=3)
    event_questions: list[EventQuestion] = Field(default_factory=list, max_length=8)
    update_relationships: bool = False
    resume_run_id: str | None = Field(default=None, pattern=r"^[0-9a-f-]{36}$")

    @property
    def simulation_protocol(self) -> QuestionSimulation | None:
        return self.question_simulation or self.benchmark

    @model_validator(mode="after")
    def unique_definitions(self):
        if self.question_simulation and (self.benchmark or not self.question_simulation.sequential_outcome):
            raise ValueError("Question forecasts require the sequential protocol without a separate benchmark configuration.")
        protocol = self.simulation_protocol
        if self.sample_count < 2 and self.benchmark is None:
            raise ValueError("Interactive experiments require at least two trajectories")
        if protocol:
            if self.decision_policy != "weighted" or {c.id for c in self.outcome_definitions} != {"yes", "no"}:
                raise ValueError("Benchmark simulation requires the shared weighted policy and binary outcomes")
            boundaries = [date.fromisoformat(value) for value in protocol.round_boundaries]
            intraday = False
            if self.steps == 1 and len(boundaries) == 2 and boundaries[0] == boundaries[1] and protocol.as_of:
                start = datetime.fromisoformat(protocol.as_of.replace("Z", "+00:00"))
                end = datetime.fromisoformat(protocol.outcome_deadline.replace("Z", "+00:00"))
                intraday = (start.utcoffset() is not None and end.utcoffset() is not None and start < end
                            and start.astimezone(timezone.utc).date() == end.astimezone(timezone.utc).date() == boundaries[0])
            if len(boundaries) != self.steps + 1 or (not intraday and any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
                raise ValueError("Benchmark requires one strictly increasing date boundary per transition")
            if self.start_date != boundaries[0].isoformat():
                raise ValueError("Benchmark start date must match its first boundary")
        if self.step_unit == "auto" and self.step_size != 1:
            raise ValueError("A custom step length requires a time unit of days, weeks, months, or years")
        if self.start_date:
            date.fromisoformat(self.start_date)
        for values in (self.outcome_definitions, self.event_questions):
            ids = [v.id for v in values]
            if len(ids) != len(set(ids)):
                raise ValueError("Outcome and event question IDs must be unique")
        if self.outcome_definitions and self.outcome_dimensions:
            raise ValueError("Provide either outcome_dimensions or outcome_definitions")
        if len({dimension.id for dimension in self.outcome_dimensions}) != len(self.outcome_dimensions):
            raise ValueError("Outcome dimension IDs must be unique")
        if len(self.outcome_definitions) == 1:
            raise ValueError("Provide at least two outcome categories, or leave them empty for a frozen automatic plan")
        if any(v.id == "other" for v in self.outcome_definitions):
            raise ValueError("'other' is reserved for ambiguous or uncovered outcomes")
        import json
        from app.domain.protocol import ScenarioProtocol
        protocol=ScenarioProtocol.model_validate(self.scenario_protocol)
        if protocol.rates and self.step_unit == "auto":
            raise ValueError("Select a time unit before configuring per-step changes")
        encoded=json.dumps(protocol.model_dump(),allow_nan=False)
        if len(encoded)>60000: raise ValueError("Scenario protocol is too large")
        self.scenario_protocol=protocol.model_dump()
        return self


class ReassessmentRequest(LLMOptions):
    outcome_dimensions: list[OutcomeDimension] = Field(default_factory=list, max_length=3)
    max_runtime_seconds: int | None = Field(default=180, ge=30, le=86400)


class QuestionAnswer(BaseModel):
    question_id: str
    answer: Literal["yes", "no", "unclear"]
    event_refs: list[str] = Field(default_factory=list)
    reason: str = ""
    evidence_kind: Literal['occurrence', 'explicit_negative', 'not_observed', 'unknown', 'state_rule'] = 'unknown'
    evidence_quote: str = Field(default='', max_length=2000)
    evidence_scope: Literal['event', 'terminal', 'whole_horizon'] = 'event'
    trace_answer: Literal['matched', 'not_matched', 'unclear'] | None = None
    trace_reason: str = Field(default='', max_length=2000)


class TerminalJudgment(BaseModel):
    dimension_id: str
    category_id: str
    matched_category_ids: list[str] = Field(default_factory=list)
    event_refs: list[str] = Field(default_factory=list, max_length=8)
    evidence_quote: str = Field(default="", max_length=1200)
    reason: str = Field(default="", max_length=1200)


class SampleClassification(BaseModel):
    sample_id: int
    category_id: str = "other"
    reason: str = ""
    event_refs: list[str] = Field(default_factory=list)
    questions: list[QuestionAnswer] = Field(default_factory=list)
    matched_category_ids: list[str] = Field(default_factory=list)
    terminal_judgments: list[TerminalJudgment] = Field(default_factory=list)


class ClassificationBatch(BaseModel):
    samples: list[SampleClassification]


class EvaluationObservation(BaseModel):
    """Held-out outcome for an empirical frequency forecast."""
    case_key: str = Field(min_length=1, max_length=200)
    probabilities: dict[str, float]
    observed_category: str

    @model_validator(mode="after")
    def distribution(self):
        if self.observed_category not in self.probabilities:
            raise ValueError("Observed category is absent from forecast")
        if any(not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError("Probabilities must lie in [0, 1]")
        if abs(sum(self.probabilities.values()) - 1) > 1e-6:
            raise ValueError("Probabilities must sum to one")
        return self
