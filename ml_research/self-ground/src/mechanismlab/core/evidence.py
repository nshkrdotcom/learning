from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "mechanismlab.thresholds.v1"
    min_effect_abs: float = 1e-6
    min_tasks_for_candidate: int = 1
    min_controls_for_candidate: int = 1
    min_replications_for_replicated: int = 2
    require_compatibility: bool = True
    require_controls_for_candidate: bool = True
    max_warning_rate_for_candidate: float | None = None
    project_specific: dict[str, Any] = Field(default_factory=dict)
