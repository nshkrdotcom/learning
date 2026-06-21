from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from mechanismlab.core.claims import ClaimSpec
from mechanismlab.core.experiments import ExperimentSpec


class BackendManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str | None = None
    kind: str
    available: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ModelBackend(Protocol):
    name: str

    def manifest(self) -> BackendManifest: ...


class RepresentationBackend(Protocol):
    name: str

    def manifest(self) -> BackendManifest: ...


class InterventionBackend(Protocol):
    name: str

    def manifest(self) -> BackendManifest: ...


class EvaluationBackend(Protocol):
    name: str

    def manifest(self) -> BackendManifest: ...


class ProjectPlugin(Protocol):
    name: str

    def claim_templates(self) -> dict[str, ClaimSpec]: ...

    def experiment_templates(self) -> dict[str, ExperimentSpec]: ...
