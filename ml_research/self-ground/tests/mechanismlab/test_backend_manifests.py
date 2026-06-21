from __future__ import annotations

import importlib.util

from mechanismlab.backends import BackendManifest, optional_package_manifest


def test_missing_optional_package_manifest_is_unavailable() -> None:
    manifest = optional_package_manifest("definitely_missing_mechanismlab_pkg", kind="test")

    assert isinstance(manifest, BackendManifest)
    assert manifest.available is False
    assert manifest.kind == "test"


def test_optional_package_manifest_uses_find_spec_without_import(monkeypatch) -> None:
    calls = []

    def fake_find_spec(name):
        calls.append(name)
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    manifest = optional_package_manifest("wandb", kind="tracker")

    assert manifest.available is False
    assert calls == ["wandb"]
