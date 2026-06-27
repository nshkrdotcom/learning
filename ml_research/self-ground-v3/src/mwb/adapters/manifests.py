from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mwb.refs import stable_ref


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
    manifest_ref: str | None = None
    backend_version_ref: str | None = None
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


def adapter_manifest_ref(manifest: AdapterCapabilityManifest | dict[str, Any]) -> str:
    payload = (
        manifest.model_dump(mode="json")
        if isinstance(manifest, AdapterCapabilityManifest)
        else manifest
    )
    return stable_ref(
        "adapter_manifest",
        payload.get("adapter_name"),
        payload.get("adapter_version"),
        payload.get("package", {}),
        payload.get("capabilities", {}),
        payload.get("claim_bearing", {}),
    )


def backend_version_ref(manifest: BackendVersionManifest | dict[str, Any]) -> str:
    payload = (
        manifest.model_dump(mode="json")
        if isinstance(manifest, BackendVersionManifest)
        else manifest
    )
    return stable_ref(
        "backend",
        payload.get("adapter_name"),
        payload.get("adapter_version"),
        payload.get("package_versions", {}),
        payload.get("python_version"),
        payload.get("platform"),
        payload.get("device"),
    )


def result_from_manifests(
    *,
    adapter_name: str,
    status: str,
    manifest: AdapterCapabilityManifest,
    backend_versions: BackendVersionManifest,
) -> AdapterConformanceResult:
    manifest_payload = manifest.model_dump(mode="json")
    backend_payload = backend_versions.model_dump(mode="json")
    manifest_payload["manifest_ref"] = adapter_manifest_ref(manifest_payload)
    backend_payload["backend_version_ref"] = backend_version_ref(backend_payload)
    return AdapterConformanceResult(
        adapter_name=adapter_name,
        status=status,
        manifest_ref=manifest_payload["manifest_ref"],
        backend_version_ref=backend_payload["backend_version_ref"],
        manifest=manifest_payload,
        backend_versions=backend_payload,
    )


def write_conformance_artifacts(
    result: AdapterConformanceResult,
    *,
    output_dir: Path | None,
    stem: str,
) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.manifest is not None:
        (output_dir / "manifest.json").write_text(
            json.dumps(result.manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if result.backend_versions is not None:
        (output_dir / "backend_versions.json").write_text(
            json.dumps(result.backend_versions, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (output_dir / f"{stem}_conformance.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


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
