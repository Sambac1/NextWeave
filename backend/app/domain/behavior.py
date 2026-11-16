"""Contextual behavioral hypotheses; stance labels never define a persona."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

VERSION = "contextual-behavior-1"


class BehaviorStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimension: str = Field(default="", max_length=100)
    description: str = Field(min_length=1, max_length=900)
    scope: str = Field(default="", max_length=300)
    counterpart: str = Field(default="", max_length=150)
    conditions: str = Field(default="", max_length=400)
    timeline_refs: list[str] = Field(default_factory=list, max_length=12)
    status: Literal["observed", "inferred", "assumption"] = "inferred"


class BehavioralModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = VERSION
    contextual_roles: list[BehaviorStatement] = Field(
        default_factory=list, max_length=8
    )
    capabilities: list[BehaviorStatement] = Field(default_factory=list, max_length=8)
    decision_patterns: list[BehaviorStatement] = Field(
        default_factory=list, max_length=8
    )
    objectives: list[BehaviorStatement] = Field(default_factory=list, max_length=8)
    calibration: Literal["not_empirically_calibrated"] = "not_empirically_calibrated"


def grounded_behavior(raw, allowed_refs=(), role_detail="") -> BehavioralModel:
    """Validate hypotheses individually. Citation presence is not trait proof."""
    raw = raw if isinstance(raw, dict) else {}
    allowed = set(allowed_refs)
    model = BehavioralModel()
    for kind in ("contextual_roles", "capabilities", "decision_patterns", "objectives"):
        statements = []
        values = raw.get(kind)
        for value in values[:8] if isinstance(values, list) else []:
            if not isinstance(value, dict):
                continue
            data = {
                k: v for k, v in value.items() if k in BehaviorStatement.model_fields
            }
            refs = value.get("timeline_refs")
            data["timeline_refs"] = (
                list(
                    dict.fromkeys(
                        r for r in refs if isinstance(r, str) and r in allowed
                    )
                )[:12]
                if isinstance(refs, list)
                else []
            )
            # An automatic profile cannot promote its own trait to observation.
            data["status"] = "inferred" if data["timeline_refs"] else "assumption"
            try:
                statements.append(BehaviorStatement.model_validate(data))
            except (TypeError, ValueError):
                continue
        setattr(model, kind, statements)
    if not model.contextual_roles and role_detail.strip():
        model.contextual_roles = [
            BehaviorStatement(
                description=role_detail[:900],
                timeline_refs=list(allowed_refs)[:12],
                status="inferred" if allowed else "assumption",
            )
        ]
    return model


def behavior_from_record(record: dict) -> BehavioralModel:
    state = record.get("public_state")
    state = state if isinstance(state, dict) else {}
    source = state.get("source_profile")
    source = source if isinstance(source, dict) else {}
    detail = str(state.get("role_detail") or source.get("role_in_event") or "")
    return grounded_behavior(
        record.get("behavioral_model"),
        record.get("relevant_timeline_refs") or [],
        detail,
    )
