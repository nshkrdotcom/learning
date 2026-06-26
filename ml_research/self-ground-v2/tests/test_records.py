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
            "artifact_paths": ["artifacts/activation.pt"],
            "canonical_record_type": "ActivationRecord",
            "evidence_role": "platform-record-metadata",
            "file": "research/records/activation.json",
            "linked_claims": ["C001"],
            "linked_decisions": ["D001"],
            "linked_runs": ["RUN_E001"],
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
    assert dashboard_payload["platform_records_by_schema_status"] == {"prd_defined_typed": 1}
    assert dashboard_payload["platform_records_are_evidence"] is False


def test_ro_crate_surfaces_all_platform_record_types_and_aliases(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    fixtures = [
        activation_record("REC-ACT-001"),
        weight_analysis_record("WeightAnalysisRecord", "REC-WGT-ALIAS"),
        circuit_graph_record("CircuitGraphRecord", "REC-CIR-ALIAS"),
        cross_model_record("CrossModelComparisonRecord", "REC-XMC-ALIAS"),
        extension_record("RemoteJobMetadataRecord", "REC-EXT-REMOTE"),
    ]
    for payload in fixtures:
        _write_json(tmp_path / "research/records" / f"{payload['record_id']}.json", payload)

    out = tmp_path / "bundles/ro-crate"
    result = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    crate = json.loads((out / "ro-crate-metadata.json").read_text(encoding="utf-8"))
    records = {
        entity["recordId"]: entity
        for entity in crate["@graph"]
        if entity.get("evidenceRole") == "platform-record-metadata"
    }
    assert records["REC-ACT-001"]["recordSpecificId"] == "ACT001"
    assert records["REC-WGT-ALIAS"]["canonicalRecordType"] == "WeightAnalysisRun"
    assert records["REC-WGT-ALIAS"]["recordSpecificId"] == "REC-WGT-ALIAS"
    assert records["REC-CIR-ALIAS"]["canonicalRecordType"] == "CircuitGraph"
    assert records["REC-XMC-ALIAS"]["canonicalRecordType"] == "CrossModelComparison"
    assert records["REC-EXT-REMOTE"]["schemaStatus"] == "extension_record"
    assert records["REC-EXT-REMOTE"]["artifactPaths"] == ["artifacts/result.json"]


def test_records_list_show_include_normalized_platform_metadata(tmp_path: Path) -> None:
    populate_project(tmp_path)
    _write_json(
        tmp_path / "research/records/weight_alias.json",
        weight_analysis_record("WeightAnalysisRecord", "REC-WGT-ALIAS"),
    )

    listed = runner.invoke(
        app,
        ["records", "list"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    shown = runner.invoke(
        app,
        ["records", "show", "REC-WGT-ALIAS"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert listed.exit_code == 0, listed.output
    assert "WeightAnalysisRun" in listed.output
    assert "extension_record" not in listed.output
    shown_payload = json.loads(shown.output)
    assert shown_payload["record_type"] == "WeightAnalysisRecord"
    assert shown_payload["canonical_record_type"] == "WeightAnalysisRun"
    assert shown_payload["schema_status"] == "prd_defined_typed"
    assert shown_payload["record_specific_id"] == "REC-WGT-ALIAS"
    assert shown_payload["linked_runs"] == ["RUN_E001"]
    assert shown_payload["linked_claims"] == ["C001"]
    assert shown_payload["linked_decisions"] == ["D001"]
    assert shown_payload["artifact_paths"] == [
        "artifacts/weight_result.json",
        "artifacts/weights.pt",
        "research/records/weight_config.yaml",
    ]


def test_platform_records_validate_prd_edge_cases(tmp_path: Path) -> None:
    populate_project(tmp_path)

    valid_string_position = activation_record("REC-ACT-STRING-POS")
    valid_string_position["token_position"] = "final"
    ok = _validate_record(
        tmp_path,
        _write_json(tmp_path / "research/records/string_pos.json", valid_string_position),
    )
    assert ok.exit_code == 0, ok.output

    bads: list[tuple[str, dict[str, Any], list[str]]] = []
    missing_record_id = activation_record("REC-IGNORED")
    missing_record_id.pop("record_id")
    bads.append(("missing_record_id", missing_record_id, ["record_id", "Suggested fix"]))
    missing_record_type = activation_record("REC-MISSING-TYPE")
    missing_record_type.pop("record_type")
    bads.append(("missing_record_type", missing_record_type, ["record_type", "Rule:"]))
    invalid_record_type = activation_record("REC-BAD-TYPE")
    invalid_record_type["record_type"] = "ActivationSource"
    bads.append(("invalid_record_type", invalid_record_type, ["Unknown", "record_type"]))
    invalid_content_hash = activation_record("REC-BAD-HASH")
    invalid_content_hash["content_hash_status"] = "unchecked"
    bads.append(("invalid_content_hash", invalid_content_hash, ["content_hash_status"]))
    invalid_analysis = weight_analysis_record(record_id="REC-BAD-ANALYSIS")
    invalid_analysis["analysis_type"] = "pca"
    bads.append(("invalid_analysis", invalid_analysis, ["analysis_type"]))
    invalid_graph = circuit_graph_record(record_id="REC-BAD-GRAPH")
    invalid_graph["graph_type"] = "networkx_graph"
    bads.append(("invalid_graph", invalid_graph, ["graph_type"]))
    invalid_comparison = cross_model_record(record_id="REC-BAD-COMP")
    invalid_comparison["comparison_type"] = "alignment"
    bads.append(("invalid_comparison", invalid_comparison, ["comparison_type"]))
    empty_weights = weight_analysis_record(record_id="REC-EMPTY-WEIGHTS")
    empty_weights["target_weights"] = []
    bads.append(("empty_weights", empty_weights, ["target_weights"]))
    empty_model_ids = cross_model_record(record_id="REC-EMPTY-MODELS")
    empty_model_ids["model_ids"] = []
    bads.append(("empty_model_ids", empty_model_ids, ["model_ids"]))
    empty_source_runs = cross_model_record(record_id="REC-EMPTY-SOURCE")
    empty_source_runs["source_run_ids"] = []
    bads.append(("empty_source_runs", empty_source_runs, ["source_run_ids"]))
    extra_prd = activation_record("REC-EXTRA")
    extra_prd["activation_source"] = "not allowed"
    bads.append(("extra_prd", extra_prd, ["activation_source", "remove unsupported"]))
    extra_extension = extension_record(record_id="REC-EXT-EXTRA")
    extra_extension["computed_result"] = {"not": "allowed"}
    bads.append(("extra_extension", extra_extension, ["computed_result", "remove unsupported"]))

    for name, payload, expected in bads:
        result = _validate_record(
            tmp_path,
            _write_json(tmp_path / "research/records" / f"{name}.json", payload),
        )
        assert result.exit_code == 2, result.output
        for text in expected:
            assert text in result.output


def test_platform_record_bundle_metadata_is_full_and_metadata_only(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    _write_json(tmp_path / "research/records/activation.json", activation_record())
    tensor_path = tmp_path / "artifacts/activation.pt"
    tensor_path.parent.mkdir(exist_ok=True)
    tensor_path.write_bytes(b"tensor bytes should not be swept")

    manifest = tmp_path / "bundles/manifest.json"
    result = runner.invoke(
        app,
        ["export", "bundle", "--out", str(manifest), "--manifest-only", "--include-artifacts"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["platform_records"] == [
        {
            "artifact_paths": ["artifacts/activation.pt"],
            "canonical_record_type": "ActivationRecord",
            "evidence_role": "platform-record-metadata",
            "file": "research/records/activation.json",
            "linked_claims": ["C001"],
            "linked_decisions": ["D001"],
            "linked_runs": ["RUN_E001"],
            "record_id": "REC-ACT-001",
            "record_specific_id": "ACT001",
            "record_type": "ActivationRecord",
            "schema_status": "prd_defined_typed",
        }
    ]
    assert "artifacts/activation.pt" not in {
        file_entry["path"] for file_entry in payload["files"]
    }
