from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from ruamel.yaml import YAML

from mechledger.inspection import read_structured_record, record_paths, relative_to_root
from mechledger.project import Project

ArtifactStorageBackend: TypeAlias = Literal["git", "dvc", "git_annex", "external"]
ContentHashStatus: TypeAlias = Literal["computed", "external_unverified"]
WeightComponentType: TypeAlias = Literal[
    "W_Q",
    "W_K",
    "W_V",
    "W_O",
    "W_OV",
    "W_QK",
    "W_in",
    "W_out",
    "W_E",
    "W_U",
    "layernorm",
    "lora_delta",
    "adapter_weight",
    "custom",
]
WeightAnalysisType: TypeAlias = Literal[
    "svd",
    "norm_analysis",
    "qk_composition",
    "ov_composition",
    "weight_delta",
    "spectral",
    "rank_analysis",
    "lora_delta_analysis",
    "custom",
]
CircuitGraphType: TypeAlias = Literal[
    "manual",
    "activation_patch_graph",
    "attribution_graph",
    "transcoder_graph",
    "acdc_graph",
    "sparse_feature_circuit",
    "hybrid",
]
CrossModelComparisonType: TypeAlias = Literal[
    "feature_matching",
    "circuit_matching",
    "behavior_matching",
    "activation_geometry",
    "intervention_transfer",
    "checkpoint_dynamics",
]
ExtensionRecordType: TypeAlias = Literal[
    "FeatureCorrespondenceRecord",
    "TrainingDynamicsRecord",
    "RemoteJobMetadataRecord",
]

PRD_RECORD_TYPES = {
    "ActivationRecord",
    "WeightAnalysisRun",
    "WeightAnalysisRecord",
    "CircuitGraph",
    "CircuitGraphRecord",
    "CrossModelComparison",
    "CrossModelComparisonRecord",
}
EXTENSION_RECORD_TYPES = {
    "FeatureCorrespondenceRecord",
    "TrainingDynamicsRecord",
    "RemoteJobMetadataRecord",
}
RECORD_TYPES = PRD_RECORD_TYPES | EXTENSION_RECORD_TYPES
SCHEMA_STATUS_PRD = "prd_defined_typed"
SCHEMA_STATUS_EXTENSION = "extension_record"


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SharedRecordFields(StrictRecord):
    record_id: str
    record_type: str
    source_paths: list[str] = Field(default_factory=list)
    linked_runs: list[str] = Field(default_factory=list)
    linked_claims: list[str] = Field(default_factory=list)
    linked_decisions: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)


class SummaryStats(StrictRecord):
    mean: float | None
    std: float | None
    norm: float | None
    max_abs: float | None


class ActivationRecord(SharedRecordFields):
    record_type: Literal["ActivationRecord"]
    activation_id: str
    run_id: str
    model: str
    tokenizer: str | None
    hook_name: str
    layer: int | None
    component_type: str
    batch_index: int | None
    token_position: int | str | None
    generation_step: int | None
    shape: list[int]
    dtype: str
    device: str
    tensor_artifact_path: str | None
    tensor_hash: str | None
    artifact_storage_backend: ArtifactStorageBackend | None
    content_hash_status: ContentHashStatus | None
    summary_stats: SummaryStats

    @field_validator("shape")
    @classmethod
    def shape_is_integer_list(cls, value: list[int]) -> list[int]:
        if not all(isinstance(item, int) for item in value):
            raise ValueError("shape must be a list of integers")
        return value


class WeightTensorSpec(StrictRecord):
    weight_tensor_id: str
    model: str
    name: str
    component_type: WeightComponentType
    layer: int | None
    head: int | None
    shape: list[int]
    dtype: str
    tensor_artifact_path: str | None
    tensor_hash: str | None
    artifact_storage_backend: ArtifactStorageBackend | None
    content_hash_status: ContentHashStatus | None

    @field_validator("shape")
    @classmethod
    def shape_is_integer_list(cls, value: list[int]) -> list[int]:
        if not all(isinstance(item, int) for item in value):
            raise ValueError("shape must be a list of integers")
        return value


class WeightAnalysisRecord(SharedRecordFields):
    record_type: Literal["WeightAnalysisRun", "WeightAnalysisRecord"]
    run_id: str
    model: str
    target_weights: list[WeightTensorSpec]
    analysis_type: WeightAnalysisType
    analysis_config_path: str
    result_artifact_path: str
    visualization_artifact_path: str | None
    linked_claims: list[str]

    @field_validator("target_weights")
    @classmethod
    def target_weights_nonempty(cls, value: list[WeightTensorSpec]) -> list[WeightTensorSpec]:
        if not value:
            raise ValueError("target_weights must contain at least one WeightTensorSpec")
        return value


class CircuitGraphRecord(SharedRecordFields):
    record_type: Literal["CircuitGraph", "CircuitGraphRecord"]
    graph_id: str
    run_id: str
    graph_type: CircuitGraphType
    nodes_artifact_path: str
    edges_artifact_path: str
    root_metric: str | None
    pruning_threshold: float | None
    graph_layout_artifact_path: str | None
    validation_artifacts: list[str]


class CrossModelComparisonRecord(SharedRecordFields):
    record_type: Literal["CrossModelComparison", "CrossModelComparisonRecord"]
    comparison_id: str
    model_ids: list[str]
    comparison_type: CrossModelComparisonType
    source_run_ids: list[str]
    result_artifact_path: str
    summary_metrics: dict[str, Any]
    linked_claims: list[str]

    @field_validator("model_ids", "source_run_ids")
    @classmethod
    def nonempty_list(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must contain at least one item")
        return value


class ExtensionRecord(SharedRecordFields):
    record_type: ExtensionRecordType
    source_paths: list[str]
    linked_runs: list[str]
    linked_claims: list[str]
    linked_decisions: list[str]
    artifact_paths: list[str]


PlatformRecord: TypeAlias = (
    ActivationRecord
    | WeightAnalysisRecord
    | CircuitGraphRecord
    | CrossModelComparisonRecord
    | ExtensionRecord
)


def validate_record(path: Path) -> PlatformRecord:
    payload = read_structured_record(path)
    record_type = payload.get("record_type")
    if not isinstance(record_type, str):
        raise ValueError(_unknown_record_type_message(path, payload, record_type))
    model = _model_for_type(record_type)
    if model is None:
        raise ValueError(_unknown_record_type_message(path, payload, record_type))
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(_validation_message(path, payload, exc)) from exc


def list_records(project: Project) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in record_paths(project.root):
        record = validate_record(path)
        records.append(record_export_payload(record, project=project, path=path))
    records.sort(key=lambda item: str(item["record_id"]))
    return records


def show_record(project: Project, record_id: str) -> dict[str, object]:
    record = next((item for item in list_records(project) if item["record_id"] == record_id), None)
    if record is None:
        raise ValueError(f"Unknown record: {record_id}")
    return record


def write_record(path: Path, record: PlatformRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    yaml = YAML()
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(record.model_dump(mode="json"), handle)


def record_export_payload(
    record: PlatformRecord,
    *,
    project: Project | None = None,
    path: Path | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = record.model_dump(mode="json")
    payload["schema_status"] = (
        SCHEMA_STATUS_EXTENSION if isinstance(record, ExtensionRecord) else SCHEMA_STATUS_PRD
    )
    payload["canonical_record_type"] = canonical_record_type(record.record_type)
    payload["record_specific_id"] = record_specific_id(record)
    payload["linked_runs"] = sorted(record_linked_runs(record))
    payload["linked_claims"] = sorted(record.linked_claims)
    payload["linked_decisions"] = sorted(record.linked_decisions)
    payload["artifact_paths"] = sorted(record_artifact_paths(record))
    if project is not None and path is not None:
        payload["file"] = relative_to_root(project, path)
    return payload


def canonical_record_type(record_type: str) -> str:
    if record_type == "WeightAnalysisRecord":
        return "WeightAnalysisRun"
    if record_type == "CircuitGraphRecord":
        return "CircuitGraph"
    if record_type == "CrossModelComparisonRecord":
        return "CrossModelComparison"
    return record_type


def record_specific_id(record: PlatformRecord) -> str:
    if isinstance(record, ActivationRecord):
        return record.activation_id
    if isinstance(record, WeightAnalysisRecord):
        return record.run_id
    if isinstance(record, CircuitGraphRecord):
        return record.graph_id
    if isinstance(record, CrossModelComparisonRecord):
        return record.comparison_id
    return record.record_id


def record_linked_runs(record: PlatformRecord) -> set[str]:
    runs = set(record.linked_runs)
    if isinstance(record, ActivationRecord | WeightAnalysisRecord | CircuitGraphRecord):
        runs.add(record.run_id)
    if isinstance(record, CrossModelComparisonRecord):
        runs.update(record.source_run_ids)
    return {run for run in runs if run}


def record_artifact_paths(record: PlatformRecord) -> set[str]:
    paths = set(record.artifact_paths)
    if isinstance(record, ActivationRecord):
        paths.update(item for item in [record.tensor_artifact_path] if item)
    elif isinstance(record, WeightAnalysisRecord):
        paths.update([record.analysis_config_path, record.result_artifact_path])
        paths.update(item for item in [record.visualization_artifact_path] if item)
        for spec in record.target_weights:
            if spec.tensor_artifact_path:
                paths.add(spec.tensor_artifact_path)
    elif isinstance(record, CircuitGraphRecord):
        paths.update([record.nodes_artifact_path, record.edges_artifact_path])
        paths.update(item for item in [record.graph_layout_artifact_path] if item)
        paths.update(record.validation_artifacts)
    elif isinstance(record, CrossModelComparisonRecord):
        paths.add(record.result_artifact_path)
    return {path for path in paths if path}


def _model_for_type(record_type: str) -> type[BaseModel] | None:
    if record_type == "ActivationRecord":
        return ActivationRecord
    if record_type in {"WeightAnalysisRun", "WeightAnalysisRecord"}:
        return WeightAnalysisRecord
    if record_type in {"CircuitGraph", "CircuitGraphRecord"}:
        return CircuitGraphRecord
    if record_type in {"CrossModelComparison", "CrossModelComparisonRecord"}:
        return CrossModelComparisonRecord
    if record_type in EXTENSION_RECORD_TYPES:
        return ExtensionRecord
    return None


def _unknown_record_type_message(
    path: Path,
    payload: dict[str, Any],
    record_type: object,
) -> str:
    object_id = str(payload.get("record_id") or "<unknown>")
    return (
        f"ERROR {path} {object_id}\n"
        "Rule: platform_record.record_type\n"
        f"Unknown or missing record_type: {record_type!r}\n"
        f"Suggested fix: use one of {', '.join(sorted(RECORD_TYPES))}."
    )


def _validation_message(path: Path, payload: dict[str, Any], exc: ValidationError) -> str:
    record_type = str(payload.get("record_type") or "<unknown>")
    object_id = _payload_object_id(payload)
    field_messages = []
    failed_fields = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        failed_fields.append(loc)
        field_messages.append(f"- {loc}: {error.get('msg')}")
    return (
        f"ERROR {path} {object_id}\n"
        "Rule: platform_record.validation\n"
        f"Record type: {record_type}\n"
        f"Failed fields: {', '.join(failed_fields)}\n"
        + "\n".join(field_messages)
        + "\n"
        f"Suggested fix: {_suggested_fix(record_type, exc)}"
    )


def _payload_object_id(payload: dict[str, Any]) -> str:
    for field in ("activation_id", "graph_id", "comparison_id", "run_id", "record_id"):
        value = payload.get(field)
        if value:
            return str(value)
    return "<unknown>"


def _suggested_fix(record_type: str, exc: ValidationError) -> str:
    errors = exc.errors()
    missing = [
        ".".join(str(item) for item in error.get("loc", ()))
        for error in errors
        if error.get("type") == "missing"
    ]
    if missing:
        return f"add required field(s): {', '.join(missing)}."
    literal_fields = [
        ".".join(str(item) for item in error.get("loc", ()))
        for error in errors
        if error.get("type") == "literal_error"
    ]
    if literal_fields:
        return f"use PRD enum value(s) for: {', '.join(literal_fields)}."
    extra = [
        ".".join(str(item) for item in error.get("loc", ()))
        for error in errors
        if error.get("type") == "extra_forbidden"
    ]
    if extra:
        return f"remove unsupported field(s) from {record_type}: {', '.join(extra)}."
    return "make the record match the PRD schema for its record_type."
