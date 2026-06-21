from __future__ import annotations

import json
from types import ModuleType

from self_ground.ravel_adapter.saebench_probe import probe_saebench_ravel_bridge


def test_probe_writes_not_installed_blocker_artifact(tmp_path) -> None:
    def missing_importer(name: str) -> ModuleType:
        raise ModuleNotFoundError(f"No module named {name!r}")

    result = probe_saebench_ravel_bridge(out_dir=tmp_path, importer=missing_importer)

    assert result.status == "not_installed"
    assert result.imported_modules == []
    assert result.blockers
    artifact = json.loads((tmp_path / "probe_result.json").read_text())
    assert artifact["status"] == "not_installed"
    assert artifact["packages_attempted"]
    assert (tmp_path / "README.md").exists()


def test_probe_records_detected_entrypoints_for_installed_like_module(tmp_path) -> None:
    module = ModuleType("sae_bench")

    def run_ravel_eval(dataset, sae, activations, cause_attribute, isolation_attribute):
        return {
            "dataset": dataset,
            "sae": sae,
            "activations": activations,
            "cause": cause_attribute,
            "isolation": isolation_attribute,
        }

    module.run_ravel_eval = run_ravel_eval

    def importer(name: str) -> ModuleType:
        if name == "sae_bench":
            return module
        raise ModuleNotFoundError(f"No module named {name!r}")

    result = probe_saebench_ravel_bridge(out_dir=tmp_path, importer=importer)

    assert result.status == "bridge_feasible"
    assert result.imported_modules == ["sae_bench"]
    assert "sae_bench.run_ravel_eval" in result.detected_entrypoints
    assert result.can_accept_custom_dataset is True
    assert result.can_accept_custom_sae is True
    assert result.can_accept_precomputed_activations is True
    assert result.can_compute_cause_isolation is True


def test_probe_records_api_incompatibility_when_entrypoint_shape_is_wrong(tmp_path) -> None:
    module = ModuleType("sae_bench")

    def run_eval():
        return None

    module.run_eval = run_eval

    def importer(name: str) -> ModuleType:
        if name == "sae_bench":
            return module
        raise ModuleNotFoundError(f"No module named {name!r}")

    result = probe_saebench_ravel_bridge(out_dir=tmp_path, importer=importer)

    assert result.status == "blocked_api_incompatible"
    assert result.detected_entrypoints == ["sae_bench.run_eval"]
    assert any("custom dataset" in str(blocker) for blocker in result.blockers)


def test_probe_importer_exceptions_are_serialized(tmp_path) -> None:
    def broken_importer(name: str) -> ModuleType:
        raise RuntimeError(f"boom {name}")

    result = probe_saebench_ravel_bridge(out_dir=tmp_path, importer=broken_importer)

    assert result.status == "not_installed"
    assert result.blockers[0]["exception_class"] == "RuntimeError"
