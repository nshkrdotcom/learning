from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ruamel.yaml import YAML

from mechledger.inspection import read_structured_record, record_paths, relative_to_root
from mechledger.project import Project

RECORD_TYPES = {
    "ActivationRecord",
    "CircuitGraphRecord",
    "WeightAnalysisRecord",
    "CrossModelComparisonRecord",
    "FeatureCorrespondenceRecord",
    "TrainingDynamicsRecord",
    "RemoteJobMetadataRecord",
}


class PlatformRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    record_id: str
    record_type: Literal[
        "ActivationRecord",
        "CircuitGraphRecord",
        "WeightAnalysisRecord",
        "CrossModelComparisonRecord",
        "FeatureCorrespondenceRecord",
        "TrainingDynamicsRecord",
        "RemoteJobMetadataRecord",
    ]
    source_paths: list[str] = Field(default_factory=list)
    linked_runs: list[str] = Field(default_factory=list)
    linked_claims: list[str] = Field(default_factory=list)
    linked_decisions: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)


def validate_record(path: Path) -> PlatformRecord:
    try:
        payload = read_structured_record(path)
        return PlatformRecord.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid platform record: {exc}") from exc


def list_records(project: Project) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in record_paths(project.root):
        record = validate_record(path)
        payload = record.model_dump(mode="json")
        payload["file"] = relative_to_root(project, path)
        records.append(payload)
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
