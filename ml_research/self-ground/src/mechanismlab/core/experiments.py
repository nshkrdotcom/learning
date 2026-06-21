from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "mechanismlab.experiment.v1"
    experiment_id: str
    claim_id: str
    status: str = "planned"
    hypothesis: str
    backend_requirements: dict[str, Any] = Field(default_factory=dict)
    task_suite: dict[str, Any] = Field(default_factory=dict)
    object_selection: dict[str, Any] = Field(default_factory=dict)
    interventions: list[dict[str, Any]] = Field(default_factory=list)
    evaluators: list[dict[str, Any]] = Field(default_factory=list)
    required_controls: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    failure_criteria: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
