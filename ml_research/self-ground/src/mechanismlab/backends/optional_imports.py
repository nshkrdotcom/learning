from __future__ import annotations

import importlib.metadata
import importlib.util

from mechanismlab.backends.base import BackendManifest


def optional_package_manifest(package_name: str, *, kind: str) -> BackendManifest:
    spec = importlib.util.find_spec(package_name)
    version = None
    if spec is not None:
        try:
            version = importlib.metadata.version(package_name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            try:
                version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                version = None
    return BackendManifest(
        name=package_name,
        version=version,
        kind=kind,
        available=spec is not None,
        details={"import_checked_with": "importlib.util.find_spec"},
    )
