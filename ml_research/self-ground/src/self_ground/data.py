from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MinimalPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    domain: str
    concept: str
    template_family: str
    x_pos: str
    x_neg: str
    x_para: str
    x_decoy: str
    held_constant: list[str]
    changed_variable: str
    control_purity: float = Field(ge=0.0, le=1.0)


class FeatureEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_id: str
    operation: Literal["ablate", "amplify", "patch"]
    delta_pos: float
    delta_neg: float
    delta_para: float
    delta_decoy: float
    necessity: float
    sufficiency: float
    collateral: float
    specificity: float
    cleanliness: float


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    feature_id: str
    template_family: str
    metrics: FeatureEffect
    metadata: dict[str, Any]
