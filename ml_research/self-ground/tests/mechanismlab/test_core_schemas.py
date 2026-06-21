from __future__ import annotations

import importlib
import pkgutil

from mechanismlab.core import (
    ArtifactContract,
    ClaimSpec,
    EvidenceThresholds,
    ExperimentSpec,
    RunManifest,
    load_claim_spec,
    load_experiment_spec,
    load_run_manifest,
    write_model,
)


def test_core_schemas_round_trip_json(tmp_path) -> None:
    claim = ClaimSpec(
        claim_id="claim.test",
        claim_type="unit",
        title="Test claim",
        claim_text="A test claim.",
    )
    experiment = ExperimentSpec(
        experiment_id="experiment.test",
        claim_id=claim.claim_id,
        hypothesis="test hypothesis",
        required_artifacts=["artifact.json"],
    )
    run = RunManifest(run_id="run.test", claim_id=claim.claim_id)
    contract = ArtifactContract(required=["artifact.json"])
    thresholds = EvidenceThresholds()

    write_model(claim, tmp_path / "claim.json")
    write_model(experiment, tmp_path / "experiment.json")
    write_model(run, tmp_path / "run.json")
    write_model(contract, tmp_path / "contract.json")
    write_model(thresholds, tmp_path / "thresholds.json")

    assert load_claim_spec(tmp_path / "claim.json") == claim
    assert load_experiment_spec(tmp_path / "experiment.json") == experiment
    assert load_run_manifest(tmp_path / "run.json") == run


def test_generic_mechanismlab_modules_do_not_import_self_ground() -> None:
    package = importlib.import_module("mechanismlab")
    prefix = package.__name__ + "."
    for module in pkgutil.walk_packages(package.__path__, prefix):
        name = module.name
        if name.startswith("mechanismlab.projects"):
            continue
        imported = importlib.import_module(name)
        assert "self_ground" not in getattr(imported, "__dict__", {})
