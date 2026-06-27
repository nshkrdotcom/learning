from __future__ import annotations

import importlib.metadata as metadata
import platform
from typing import Any

from pydantic import BaseModel, Field


class ClaimBearingSupport(BaseModel):
    supported: bool
    required_conformance: list[str] = Field(default_factory=list)


class AdapterCapabilityManifest(BaseModel):
    adapter_name: str
    adapter_version: str
    package: dict[str, str | None]
    capabilities: dict[str, bool | str]
    claim_bearing: ClaimBearingSupport
    limitations: list[str] = Field(default_factory=list)


class BackendVersionManifest(BaseModel):
    adapter_name: str
    adapter_version: str
    package_versions: dict[str, str | None]
    python_version: str
    platform: str
    cuda_available: bool
    cuda_version: str | None = None
    device: str


class AdapterConformanceResult(BaseModel):
    adapter_name: str
    status: str
    checks: list[dict[str, Any]] = Field(default_factory=list)
    manifest: dict[str, Any] | None = None
    backend_versions: dict[str, Any] | None = None
    model_identity: dict[str, Any] | None = None
    dictionary_identity: dict[str, Any] | None = None
    tensor_space: dict[str, Any] | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def add_check(self, name: str, status: str, **details: Any) -> None:
        self.checks.append({"name": name, "status": status, **details})
        if status == "fail" and self.status != "fail":
            self.status = "fail"


class ClaimBearingGateResult(BaseModel):
    supported: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def backend_version_manifest(
    *,
    adapter_name: str,
    adapter_version: str,
    packages: list[str],
    device: str,
) -> BackendVersionManifest:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(torch.version.cuda) if torch.version.cuda else None
    except Exception:
        cuda_available = False
        cuda_version = None

    return BackendVersionManifest(
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        package_versions={package: package_version(package) for package in packages},
        python_version=platform.python_version(),
        platform=platform.platform(),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        device=device,
    )

