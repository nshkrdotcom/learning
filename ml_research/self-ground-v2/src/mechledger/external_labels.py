from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ruamel.yaml import YAML

from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.core.diagnostics import Diagnostic, DiagnosticSeverity
from mechledger.project import Project, now_utc

SEMANTIC_EXCLUDE = {"imported_at", "semantic_hash", "linked_claims"}


class ExternalLabel(BaseModel):
    model_config = ConfigDict(extra="allow")

    label_id: str
    source: str
    source_url: str | None = None
    source_model: str | None = None
    label_text: str
    feature_id: str
    model: str | None = None
    layer_or_hook: str | None = None
    sae_release: str | None = None
    sae_id: str | None = None
    created_at: str | None = None
    imported_at: str | None = None
    confidence: float | None = None
    license: str | None = None
    linked_claims: list[str] = Field(default_factory=list)
    notes: str | None = None
    semantic_hash: str | None = None


def import_labels(project: Project, source: Path) -> list[ExternalLabel]:
    imported = [
        validate_payload(payload, source=source, line=line)
        for payload, line in _load_source(source)
    ]
    existing = {label.label_id: label for label in read_labels(project)}
    for label in imported:
        label.imported_at = label.imported_at or now_utc()
        label.semantic_hash = semantic_hash(label)
        existing[label.label_id] = label
    write_labels(project, sorted(existing.values(), key=lambda item: item.label_id))
    return imported


def validate_file(source: Path) -> list[ExternalLabel]:
    return [
        validate_payload(payload, source=source, line=line)
        for payload, line in _load_source(source)
    ]


def read_labels(project: Project) -> list[ExternalLabel]:
    path = registry_path(project)
    if not path.exists():
        return []
    labels: list[ExternalLabel] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            labels.append(ExternalLabel.model_validate(payload))
        except json.JSONDecodeError as exc:
            raise ValueError(_label_json_message(path, line_number, exc.msg)) from exc
        except ValidationError as exc:
            raise ValueError(
                _label_validation_message(path, line_number, payload, exc)
            ) from exc
    labels.sort(key=lambda item: item.label_id)
    return labels


def write_labels(project: Project, labels: list[ExternalLabel]) -> None:
    path = registry_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(label.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        for label in sorted(labels, key=lambda item: item.label_id)
    ]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def show_label(project: Project, label_id: str) -> ExternalLabel:
    label = next((item for item in read_labels(project) if item.label_id == label_id), None)
    if label is None:
        raise ValueError(f"Unknown label: {label_id}")
    return label


def link_label_to_claim(project: Project, label_id: str, claim_id: str) -> ExternalLabel:
    claim_ledger = parse_claim_ledger(project.resolve(project.config.default_claim_ledger))
    if claim_id not in claim_ledger.claims:
        raise ValueError(f"Unknown claim: {claim_id}")
    labels = read_labels(project)
    label = next((item for item in labels if item.label_id == label_id), None)
    if label is None:
        raise ValueError(f"Unknown label: {label_id}")
    label.linked_claims = sorted({*label.linked_claims, claim_id})
    write_labels(project, labels)
    return label


def registry_path(project: Project) -> Path:
    return project.root / "research/literature/external_labels.jsonl"


def validate_payload(
    payload: dict[str, Any],
    *,
    source: Path,
    line: int | None = None,
) -> ExternalLabel:
    try:
        label = ExternalLabel.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(_label_validation_message(source, line, payload, exc)) from exc
    label.semantic_hash = semantic_hash(label)
    return label


def semantic_hash(label: ExternalLabel) -> str:
    payload = label.model_dump(mode="json")
    canonical = {key: value for key, value in payload.items() if key not in SEMANTIC_EXCLUDE}
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_source(source: Path) -> list[tuple[dict[str, Any], int | None]]:
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Label import path does not exist: {source}")
    if source.suffix == ".jsonl":
        rows: list[tuple[dict[str, Any], int | None]] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(_label_json_message(source, line_number, exc.msg)) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    _label_state_message(
                        source,
                        line_number,
                        None,
                        "external_label.jsonl.row_type",
                        "JSONL label row must be an object.",
                        "write one JSON object per line.",
                    )
                )
            rows.append((payload, line_number))
        return rows
    if source.suffix in {".yaml", ".yml", ".json"}:
        if source.suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            yaml = YAML(typ="safe")
            payload = yaml.load(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return [(payload, None)]
        if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
            return [(item, None) for item in payload]
        raise ValueError(f"{source}: label file must contain an object or list of objects.")
    raise ValueError("External labels import supports .jsonl, .json, .yaml, and .yml.")


def _label_validation_message(
    path: Path,
    line: int | None,
    payload: dict[str, Any],
    exc: ValidationError,
) -> str:
    object_id = str(payload.get("label_id") or "<unknown>")
    failed_fields = [
        ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        for error in exc.errors()
    ]
    details = "\n".join(
        f"- {field}: {error.get('msg')}"
        for field, error in zip(failed_fields, exc.errors(), strict=False)
    )
    return Diagnostic(
        severity=DiagnosticSeverity.ERROR,
        code="external_label.validation",
        message=f"Invalid external label.\nFailed fields: {', '.join(failed_fields)}\n{details}",
        file=str(path),
        line=line,
        object_id=object_id,
        suggested_fix="make the label record match the ExternalLabelRecord metadata schema.",
    ).format()


def _label_json_message(path: Path, line: int, message: str) -> str:
    return _label_state_message(
        path,
        line,
        None,
        "external_label.json.invalid",
        f"Malformed JSON: {message}",
        "fix the JSON syntax for this label row.",
    )


def _label_state_message(
    path: Path,
    line: int | None,
    object_id: str | None,
    rule: str,
    message: str,
    suggested_fix: str,
) -> str:
    return Diagnostic(
        severity=DiagnosticSeverity.ERROR,
        code=rule,
        message=message,
        file=str(path),
        line=line,
        object_id=object_id,
        suggested_fix=suggested_fix,
    ).format()
