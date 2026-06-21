from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from mechanismlab.backends.base import BackendManifest
from mechanismlab.core.runs import RunManifest


class OptionalDependencyUnavailable(RuntimeError):
    def __init__(self, package_name: str) -> None:
        super().__init__(f"optional dependency is not installed: {package_name}")
        self.package_name = package_name


class TrackerBackend(Protocol):
    name: str

    def start_run(self, run: RunManifest) -> None: ...

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def log_artifact(self, path: Path, name: str | None = None) -> None: ...

    def finish(self, status: str | None = None) -> None: ...


def unavailable_manifest(package_name: str, *, kind: str = "tracker") -> BackendManifest:
    return BackendManifest(
        name=package_name,
        kind=kind,
        available=False,
        details={"reason": f"optional dependency {package_name!r} is not installed"},
    )
