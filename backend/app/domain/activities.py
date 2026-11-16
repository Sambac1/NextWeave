"""Domain-neutral labels for behaviors described by settled public events."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

ActivityFamily = Literal['proposal','negotiation','agreement','communication','implementation','resource','outcome']
ActivityStage = Literal['proposed','discussed','conditional','accepted','rejected','announced','started','ongoing','completed','failed','paused','cancelled']


class ActivityAnnotation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    method: Literal["settlement_annotation", "legacy_structure_rule"] = "settlement_annotation"
    family: ActivityFamily
    stage: ActivityStage
    quote: str = Field(min_length=8, max_length=1800)
