"""Explicit domain mechanisms; no executable code or inferred physical laws."""

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator, ConfigDict

from app.domain.ensemble import OutcomeCondition


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StateRate(ProtocolModel):
    path: str = Field(pattern=r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")
    delta_per_step: float = Field(allow_inf_nan=False)


class ScheduledObservation(ProtocolModel):
    step: int = Field(ge=1, le=12)
    description: str = Field(min_length=1, max_length=1500)
    updates: dict[str, Any] = Field(default_factory=dict)


class ActionMechanism(ProtocolModel):
    action_type: str = Field(pattern=r"^[A-Za-z0-9_-]{1,60}$")
    allowed_actor_ids: list[str] = Field(default_factory=list)
    preconditions: list[OutcomeCondition] = Field(default_factory=list, max_length=30)
    effects: list[StateRate] = Field(default_factory=list, max_length=30)


class ScenarioProtocol(ProtocolModel):
    rates: list[StateRate] = Field(default_factory=list, max_length=50)
    scheduled_observations: list[ScheduledObservation] = Field(
        default_factory=list, max_length=50
    )
    decision_due_steps: dict[str, list[int]] = Field(default_factory=dict)
    action_mechanisms: list[ActionMechanism] = Field(
        default_factory=list, max_length=30
    )
    idle_policy: Literal["review", "advance_if_unchanged"] = "review"
    assumptions: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def unique_actions(self):
        ids = [a.action_type for a in self.action_mechanisms]
        if len(ids) != len(set(ids)):
            raise ValueError("Action mechanism IDs must be unique")
        if any(
            not 1 <= s <= 12
            for steps in self.decision_due_steps.values()
            for s in steps
        ):
            raise ValueError("Decision steps must be between one and twelve")
        return self
