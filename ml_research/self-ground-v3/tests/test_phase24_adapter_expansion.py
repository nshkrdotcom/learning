import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.adapters.conformance import claim_bearing_gate
from mwb.adapters.manifests import AdapterConformanceResult
from mwb.adapters.neuronpedia import NeuronpediaAdapter
from mwb.adapters.nnsight import NNsightAdapter
from mwb.adapters.pyvene import PyVeneAdapter
from mwb.artifacts import ArtifactRegistry
from mwb.cli import app
from mwb.project import ProjectManager
from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_optional_adapter_manifests_declare_real_capabilities_and_no_claim_bearing() -> None:
    nnsight = NNsightAdapter().capability_manifest()
    pyvene = PyVeneAdapter().capability_manifest()
    neuronpedia = NeuronpediaAdapter().capability_manifest()

    assert nnsight.adapter_name == "NNsightAdapter"
    assert nnsight.capabilities["activation_access"] is True
    assert nnsight.capabilities["activation_edit"] is True
    assert nnsight.capabilities["gradient_capture"] is True
    assert nnsight.capabilities["remote_tracing"] == "conditional"
    assert nnsight.capabilities["nnterp_normalized_names"] == "conditional"
    assert nnsight.claim_bearing.supported is False
    assert "diagnostic-only until exact intervention conformance passes" in nnsight.limitations

    assert pyvene.adapter_name == "PyVeneAdapter"
    assert pyvene.capabilities["intervention_configs"] is True
    assert pyvene.capabilities["static_interventions"] is True
    assert pyvene.capabilities["trainable_interventions"] == "conditional"
    assert pyvene.capabilities["causal_abstraction"] is True
    assert pyvene.claim_bearing.supported is False

    assert neuronpedia.adapter_name == "NeuronpediaAdapter"
    assert neuronpedia.capabilities["read_feature_metadata"] is True
    assert neuronpedia.capabilities["write_feature_metadata"] is False
    assert neuronpedia.capabilities["feature_ref_roundtrip"] is True
    assert neuronpedia.claim_bearing.supported is False


def test_missing_optional_backends_emit_diagnostic_only_results(monkeypatch) -> None:
    adapter = NNsightAdapter()
    monkeypatch.setattr(adapter, "backend_available", lambda: False)

    result = adapter.run_conformance(
        model_name="gpt2",
        module_path="transformer.h.0.mlp",
        device="cpu",
        dry_run=False,
    )
    gate = claim_bearing_gate(
        [result],
        required_adapters=["NNsightAdapter"],
        required_refs={
            "model_ref": "model_gpt2",
            "backend_version_ref": result.backend_version_ref,
        },
    )

    assert result.status == "diagnostic_only"
    assert result.checks[0]["name"] == "optional_dependency_available"
    assert result.checks[0]["status"] == "fail"
    assert "nnsight" in result.errors[0]
    assert gate.supported is False
    assert "adapter conformance not pass: NNsightAdapter=diagnostic_only" in gate.blockers


def test_manifest_without_claim_bearing_support_blocks_claim_gate() -> None:
    manifest = PyVeneAdapter().capability_manifest()
    result = AdapterConformanceResult(
        adapter_name="PyVeneAdapter",
        status="pass",
        manifest=manifest.model_dump(mode="json"),
        backend_version_ref="backend_pyvene",
    )

    gate = claim_bearing_gate(
        [result],
        required_adapters=["PyVeneAdapter"],
        required_refs={"backend_version_ref": result.backend_version_ref},
    )

    assert gate.supported is False
    assert "adapter not claim-bearing: PyVeneAdapter" in gate.blockers


def test_pyvene_dry_run_conformance_materializes_intervention_contract() -> None:
    result = PyVeneAdapter().run_conformance(
        model_name="gpt2",
        module_path="transformer.h.0.mlp",
        intervention_kind="resample_ablation",
        device="cpu",
        dry_run=True,
    )

    assert result.status == "diagnostic_only"
    assert result.model_identity is not None
    assert result.model_identity["backend"] == "pyvene"
    assert any(check["name"] == "intervention_config_roundtrip" for check in result.checks)
    assert result.manifest["claim_bearing"]["supported"] is False


def test_neuronpedia_dry_run_metadata_ref_is_read_only() -> None:
    result = NeuronpediaAdapter().run_conformance(
        model_id="gemma-2-2b",
        sae_id="20-gemmascope-res-16k",
        feature_index=123,
        dry_run=True,
    )

    assert result.status == "diagnostic_only"
    assert result.dictionary_identity is not None
    assert result.dictionary_identity["provider"] == "Neuronpedia"
    assert result.artifact_refs == ["neuronpedia://gemma-2-2b/20-gemmascope-res-16k/123"]
    assert any(check["name"] == "feature_metadata_ref_roundtrip" for check in result.checks)


def test_optional_adapter_cli_writes_persisted_manifest_and_rebuilds_sqlite(
    tmp_path: Path, monkeypatch
) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "adapter",
            "conformance",
            "nnsight",
            "--model",
            "gpt2",
            "--module-path",
            "transformer.h.0.mlp",
            "--device",
            "cpu",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["adapter_name"] == "NNsightAdapter"
    assert payload["status"] == "diagnostic_only"
    assert payload["manifest_ref"].startswith("adapter_manifest_")
    assert payload["backend_version_ref"].startswith("backend_")

    adapter_dir = tmp_path / ".mechanism" / "adapters" / "nnsight"
    assert (adapter_dir / "manifest.json").exists()
    assert (adapter_dir / "backend_versions.json").exists()
    assert (adapter_dir / "nnsight_conformance.json").exists()

    restored = rebuild_sqlite_index(project, output_path=tmp_path / "rebuilt.sqlite")

    assert restored["counts"]["adapter_manifests"] == 1
    assert restored["counts"]["backend_versions"] == 1
    indexed_manifest = fetch_payload(
        tmp_path / "rebuilt.sqlite", "adapter_manifests", payload["manifest_ref"]
    )
    indexed_backend = fetch_payload(
        tmp_path / "rebuilt.sqlite", "backend_versions", payload["backend_version_ref"]
    )
    assert indexed_manifest["adapter_name"] == "NNsightAdapter"
    assert indexed_backend["adapter_name"] == "NNsightAdapter"


def test_optional_pyvene_and_neuronpedia_cli_commands_write_conformance(
    tmp_path: Path, monkeypatch
) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    runner = CliRunner()

    pyvene = runner.invoke(
        app,
        [
            "adapter",
            "conformance",
            "pyvene",
            "--model",
            "gpt2",
            "--module-path",
            "transformer.h.0.mlp",
            "--intervention-kind",
            "resample_ablation",
            "--device",
            "cpu",
            "--dry-run",
        ],
    )
    neuronpedia = runner.invoke(
        app,
        [
            "adapter",
            "conformance",
            "neuronpedia",
            "--model-id",
            "gemma-2-2b",
            "--sae-id",
            "20-gemmascope-res-16k",
            "--feature-index",
            "123",
            "--dry-run",
        ],
    )

    assert pyvene.exit_code == 0, pyvene.output
    assert neuronpedia.exit_code == 0, neuronpedia.output
    assert json.loads(pyvene.output)["adapter_name"] == "PyVeneAdapter"
    assert json.loads(neuronpedia.output)["adapter_name"] == "NeuronpediaAdapter"
    assert (tmp_path / ".mechanism" / "adapters" / "pyvene" / "pyvene_conformance.json").exists()
    assert (
        tmp_path
        / ".mechanism"
        / "adapters"
        / "neuronpedia"
        / "neuronpedia_conformance.json"
    ).exists()


def test_artifact_registry_records_external_artifact_pointers(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")

    lfs_pointer = tmp_path / "weights.safetensors"
    lfs_pointer.write_text(
        "\n".join(
            [
                "version https://git-lfs.github.com/spec/v1",
                "oid sha256:abc123",
                "size 42",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dvc_pointer = tmp_path / "dataset.csv.dvc"
    dvc_pointer.write_text(
        "outs:\n- md5: 8d777f385d3dfec8815d20f7496026dc\n  size: 1024\n  path: dataset.csv\n",
        encoding="utf-8",
    )
    annex_pointer = tmp_path / "large.pt"
    annex_pointer.symlink_to(".git/annex/objects/ab/cd/SHA256E-s123--deadbeef.pt")

    registry = ArtifactRegistry(project)
    lfs_record = registry.register_path(lfs_pointer, role="model_weights")
    dvc_record = registry.register_path(dvc_pointer, role="dataset")
    annex_record = registry.register_path(annex_pointer, role="activation_cache")

    assert lfs_record.materialized is False
    assert lfs_record.pointer == {
        "backend": "git_lfs",
        "oid": "sha256:abc123",
        "size": 42,
    }
    assert dvc_record.materialized is False
    assert dvc_record.pointer["backend"] == "dvc"
    assert dvc_record.pointer["path"] == "dataset.csv"
    assert annex_record.materialized is False
    assert annex_record.pointer["backend"] == "git_annex"
    assert annex_record.pointer["key"] == "SHA256E-s123--deadbeef.pt"

    stored = fetch_payload(project.sqlite_path, "artifacts", lfs_record.artifact_ref)
    assert stored["pointer"]["backend"] == "git_lfs"
    assert stored["materialized"] is False
