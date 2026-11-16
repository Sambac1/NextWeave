"""Public issue changes, separate from actor intentions and event frequency labels."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


ProcessStateCode = Literal['unknown','proposed','discussed','conditional','accepted','announced','started','ongoing','completed','failed','rejected','blocked','paused','cancelled']


class ProcessTopic(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,60}$')
    title: str = Field(min_length=1, max_length=120)


class IssueUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    issue_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,60}$')
    title: str = Field(min_length=1, max_length=120)
    before: str = Field(default='', max_length=500)
    after: str = Field(min_length=1, max_length=500)
    kind: Literal['proposal','response','decision','implementation','outcome','unchanged']
    state_code: ProcessStateCode = 'unknown'
    actor_ids: list[str] = Field(default_factory=list, max_length=12)
    event_refs: list[str] = Field(min_length=1, max_length=8)
    evidence_quote: str = Field(min_length=8, max_length=1200)
    cause_action_ids: list[str] = Field(default_factory=list, max_length=8)
    pending_actor_ids: list[str] = Field(default_factory=list, max_length=12)


class ProcessIssue(BaseModel):
    issue_id: str
    title: str
    state: str
    actor_ids: list[str] = Field(default_factory=list)
    event_refs: list[str] = Field(default_factory=list)
    pending_actor_ids: list[str] = Field(default_factory=list)
    updated_round: int = 0
    last_response_round: dict[str, int] = Field(default_factory=dict)
