from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helpers_project import populate_project, runner

from mechledger.cli import app


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def activation_record(record_id: str = "REC-ACT-001") -> dict[str, Any]:
    return {
        "record_id": record_id,
        "record_type": "ActivationRecord",
        "activation_id": "ACT001",
        "run_id": "RUN_E001",
        "model": "pythia-70m",
        "tokenizer": "EleutherAI/pythia-70m",
        "hook_name": "blocks.2.hook_resid_post",
        "layer": 2,
        "component_type": "residual_stream",
        "batch_index": 0,
        "token_position": 3,
        "generation_step": None,
        "shape": [1, 10, 512],
        "dtype": "float32",
        "device": "cuda:0",
        "tensor_artifact_path": "artifacts/activation.pt",
        "tensor_hash": "sha256:activation",
        "artifact_storage_backend": "git",
        "content_hash_status": "computed",
        "summary_stats": {"mean": 0.0, "std": 1.0, "norm": 12.0, "max_abs": 3.0},
        "source_paths": ["research/experiments/E001_test.md"],
        "linked_runs": ["RUN_E001"],
        "linked_claims": ["C001"],
        "linked_decisions": ["D001"],
        "artifact_paths": ["artifacts/activation.pt"],
    }


def weight_analysis_record(
    record_type: str = "WeightAnalysisRun",
    record_id: str = "REC-WGT-001",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "record_type": record_type,
        "run_id": "RUN_E001",
        "model": "pythia-70m",
        "target_weights": [
            {
                "weight_tensor_id": "WT001",
                "model": "pythia-70m",
                "name": "blocks.2.attn.W_O",
                "component_type": "W_O",
                "layer": 2,
                "head": 1,
                "shape": [8, 64, 512],
                "dtype": "float32",
                "tensor_artifact_path": "artifacts/weights.pt",
                "tensor_hash": "sha256:weights",
                "artifact_storage_backend": "git_annex",
                "content_hash_status": "external_unverified",
            }
        ],
        "analysis_type": "ov_composition",
        "analysis_config_path": "research/records/weight_config.yaml",
        "result_artifact_path": "artifacts/weight_result.json",
        "visualization_artifact_path": None,
        "linked_claims": ["C001"],
        "linked_runs": ["RUN_E001"],
        "linked_decisions": ["D001"],
        "source_paths": ["research/records/weight_config.yaml"],
        "artifact_paths": ["artifacts/weights.pt", "artifacts/weight_result.json"],
    }


def circuit_graph_record(
    record_type: str = "CircuitGraph",
    record_id: str = "REC-CIR-001",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "record_type": record_type,
        "graph_id": "G001",
        "run_id": "RUN_E001",
        "graph_type": "attribution_graph",
        "nodes_artifact_path": "artifacts/nodes.json",
        "edges_artifact_path": "artifacts/edges.json",
        "root_metric": "specificity_gap",
        "pruning_threshold": 0.05,
        "graph_layout_artifact_path": None,
        "validation_artifacts": ["artifacts/validation.json"],
        "linked_runs": ["RUN_E001"],
        "linked_claims": ["C001"],
        "linked_decisions": ["D001"],
        "source_paths": ["research/experiments/E001_test.md"],
        "artifact_paths": ["artifacts/nodes.json", "artifacts/edges.json"],
    }


def cross_model_record(
    record_type: str = "CrossModelComparison",
    record_id: str = "REC-XMC-001",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "record_type": record_type,
        "comparison_id": "CMP001",
        "model_ids": ["pythia-70m", "pythia-160m"],
        "comparison_type": "feature_matching",
        "source_run_ids": ["RUN_E001"],
        "result_artifact_path": "artifacts/cross_model.json",
        "summary_metrics": {"overlap": 0.42},
        "linked_claims": ["C001"],
        "linked_runs": ["RUN_E001"],
        "linked_decisions": ["D001"],
        "source_paths": ["research/experiments/E001_test.md"],
        "artifact_paths": ["artifacts/cross_model.json"],
    }


def extension_record(
    record_type: str = "FeatureCorrespondenceRecord",
    record_id: str = "REC-EXT-001",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "record_type": record_type,
        "source_paths": ["artifacts/result.json"],
        "linked_runs": ["RUN_E001"],
        "linked_claims": ["C001"],
        "linked_decisions": ["D001"],
        "artifact_paths": ["artifacts/result.json"],
    }


def _validate_record(tmp_path: Path, path: Path):
    return runner.invoke(
        app,
        ["records", "validate", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )


def test_records_validate_prd_typed_records_and_extension_records(tmp_path: Path) -> None:
    populate_project(tmp_path)
    fixtures = [
        activation_record(),
        weight_analysis_record("WeightAnalysisRun", "REC-WGT-001"),
        weight_analysis_record("WeightAnalysisRecord", "REC-WGT-002"),
        circuit_graph_record("CircuitGraph", "REC-CIR-001"),
        circuit_graph_record("CircuitGraphRecord", "REC-CIR-002"),
        cross_model_record("CrossModelComparison", "REC-XMC-001"),
        cross_model_record("CrossModelComparisonRecord", "REC-XMC-002"),
        extension_record("FeatureCorrespondenceRecord", "REC-EXT-001"),
        extension_record("TrainingDynamicsRecord", "REC-EXT-002"),
        extension_record("RemoteJobMetadataRecord", "REC-EXT-003"),
    ]
    for payload in fixtures:
        path = _write_json(tmp_path / "research/records" / f"{payload['record_id']}.json", payload)
        result = _validate_record(tmp_path, path)
        assert result.exit_code == 0, result.output

    listed = runner.invoke(
        app,
        ["records", "list"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    shown = runner.invoke(
        app,
        ["records", "show", "REC-ACT-001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert listed.exit_code == 0 and "REC-EXT-003" in listed.output
    shown_payload = json.loads(shown.output)
    assert shown_payload["record_type"] == "ActivationRecord"
    assert shown_payload["activation_id"] == "ACT001"
    assert shown_payload["schema_status"] == "prd_defined_typed"


def test_prd_typed_records_reject_missing_specific_ids_and_invalid_fields(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)

    missing_id = activation_record("REC-BAD-MISSING-ID")
    missing_id.pop("activation_id")
    missing_id_path = _write_json(tmp_path / "research/records/missing_id.json", missing_id)
    missing_id_result = _validate_record(tmp_path, missing_id_path)
    assert missing_id_result.exit_code == 2
    assert "missing_id.json" in missing_id_result.output
    assert "ActivationRecord" in missing_id_result.output
    assert "activation_id" in missing_id_result.output
    assert "Suggested fix" in missing_id_result.output

    invalid_enum = weight_analysis_record(record_id="REC-BAD-ENUM")
    invalid_enum["target_weights"][0]["component_type"] = "attention_output"
    invalid_enum_path = _write_json(tmp_path / "research/records/invalid_enum.json", invalid_enum)
    invalid_enum_result = _validate_record(tmp_path, invalid_enum_path)
    assert invalid_enum_result.exit_code == 2
    assert "component_type" in invalid_enum_result.output
    assert "W_O" in invalid_enum_result.output

    invalid_shape = activation_record("REC-BAD-SHAPE")
    invalid_shape["shape"] = ["not-an-int"]
    invalid_shape_path = _write_json(
        tmp_path / "research/records/invalid_shape.json", invalid_shape
    )
    invalid_shape_result = _validate_record(tmp_path, invalid_shape_path)
    assert invalid_shape_result.exit_code == 2
    assert "shape" in invalid_shape_result.output

    missing_stats = activation_record("REC-BAD-STATS")
    missing_stats["summary_stats"].pop("max_abs")
    missing_stats_path = _write_json(
        tmp_path / "research/records/missing_stats.json", missing_stats
    )
    missing_stats_result = _validate_record(tmp_path, missing_stats_path)
    assert missing_stats_result.exit_code == 2
    assert "summary_stats.max_abs" in missing_stats_result.output


def test_ro_crate_bundle_and_dashboard_surface_typed_record_metadata(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    _write_json(tmp_path / "research/records/activation.json", activation_record())

    out = tmp_path / "bundles/ro-crate"
    first = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert first.exit_code == 0, first.output
    first_bytes = (out / "ro-crate-metadata.json").read_bytes()
    second = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert second.exit_code == 0, second.output
    assert first_bytes == (out / "ro-crate-metadata.json").read_bytes()
    crate = json.loads(first_bytes)
    record_entity = next(
        entity
        for entity in crate["@graph"]
        if entity.get("@id") == "research/records/activation.json#ACT001"
    )
    assert record_entity["recordType"] == "ActivationRecord"
    assert record_entity["recordSpecificId"] == "ACT001"
    assert record_entity["linkedRuns"] == [".mechledger/runs/RUN_E001/"]
    assert record_entity["linkedDecisions"] == ["research/logs/decision_log.md#D001"]
    assert record_entity["artifactPaths"] == ["artifacts/activation.pt"]
    assert record_entity["evidenceRole"] == "platform-record-metadata"

    manifest = tmp_path / "bundles/manifest.json"
    bundled = runner.invoke(
        app,
        ["export", "bundle", "--out", str(manifest), "--manifest-only"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert bundled.exit_code == 0, bundled.output
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["platform_records"] == [
        {
            "file": "research/records/activation.json",
            "record_id": "REC-ACT-001",
            "record_specific_id": "ACT001",
            "record_type": "ActivationRecord",
            "schema_status": "prd_defined_typed",
        }
    ]

    dashboard_out = tmp_path / ".mechledger/dashboard/data.json"
    dashboard = runner.invoke(
        app,
        ["dashboard", "data", "--out", str(dashboard_out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert dashboard.exit_code == 0, dashboard.output
    dashboard_payload = json.loads(dashboard_out.read_text(encoding="utf-8"))
    assert dashboard_payload["platform_records_by_type"] == {"ActivationRecord": 1}
    assert dashboard_payload["platform_records_are_evidence"] is False
