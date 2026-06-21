from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "mechanismlab.run.v1"
    run_id: str
    claim_id: str | None = None
    experiment_id: str | None = None
    project: str | None = None
    git: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    backends: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
