from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClaimSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "mechanismlab.claim.v1"
    claim_id: str
    claim_type: str
    title: str
    claim_text: str
    project: str | None = None
    concept: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    internal_object: dict[str, Any] = Field(default_factory=dict)
    intervention: dict[str, Any] = Field(default_factory=dict)
    expected_effect: dict[str, Any] = Field(default_factory=dict)
    controls_required: list[str] = Field(default_factory=list)
    promotion_rules: dict[str, Any] = Field(default_factory=dict)
    falsification_conditions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
