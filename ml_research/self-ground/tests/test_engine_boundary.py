from __future__ import annotations

from pathlib import Path

import pytest

from self_ground.engine_boundary import (
    LEGACY_FEATURE_SPACE_PROXY,
    RESIDUAL_SMOKE_DIAGNOSTIC,
    TRANSFORMER_LENS_BACKEND,
    is_claim_eligible_backend,
    validate_engine_backend,
)
from self_ground.hooking import ENGINE_BACKEND


def test_forbidden_self_ground_engine_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="self_ground_generic_engine"):
        validate_engine_backend("self_ground_generic_engine")


def test_thin_hooking_call_site_names_transformerlens_backend() -> None:
    assert ENGINE_BACKEND == TRANSFORMER_LENS_BACKEND


def test_diagnostic_and_legacy_backends_are_not_claim_eligible() -> None:
    assert is_claim_eligible_backend(TRANSFORMER_LENS_BACKEND)
    assert not is_claim_eligible_backend(RESIDUAL_SMOKE_DIAGNOSTIC)
    assert not is_claim_eligible_backend(LEGACY_FEATURE_SPACE_PROXY)


def test_optional_intervention_frameworks_are_not_core_dependencies() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    dependencies_block = pyproject.split("[project.scripts]", 1)[0]
    assert "nnsight" not in dependencies_block
    assert "pyvene" not in dependencies_block
