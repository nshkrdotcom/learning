from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from mechledger.alias import resolve_run_id
from mechledger.artifacts import load_manifest, write_manifest
from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.inspection import sha256_file, write_json
from mechledger.project import Project, now_utc


class RedactionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    redaction_id: str
    target_type: Literal["run", "artifact", "environment", "stdout", "stderr", "other"]
    target_path: str
    reason: str
    created_at: str
    created_by: str | None
    original_hash: str | None
    redacted_hash: str | None
    placeholder_path: str | None


def redact_run(project: Project, run_alias: str, *, reason: str) -> tuple[RedactionRecord, str]:
    if not reason.strip():
        raise ValueError("--reason must be non-empty.")
    run_id = resolve_run_id(project, run_alias)
    run_dir = project.runs_dir / run_id
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"Run is missing run.json: {run_json}")
    record_path = run_dir / "redaction_record.json"
    if record_path.exists():
        return RedactionRecord.model_validate(_read_json(record_path)), "already_redacted"
    record = RedactionRecord(
        redaction_id=_redaction_id("run", run_id),
        target_type="run",
        target_path=f".mechledger/runs/{run_id}/",
        reason=reason.strip(),
        created_at=now_utc(),
        created_by=os.environ.get("USER"),
        original_hash=sha256_file(run_json),
        redacted_hash=None,
        placeholder_path=None,
    )
    write_json(record_path, record.model_dump(mode="json"))
    return record, "redacted"


def redact_artifact(
    project: Project,
    path: Path,
    *,
    reason: str,
    run_alias: str | None = None,
) -> tuple[RedactionRecord, str]:
    if not reason.strip():
        raise ValueError("--reason must be non-empty.")
    target = _resolve_project_local_path(project, path)
    if not target.exists() and not (Path(str(target) + ".redacted")).exists():
        raise FileNotFoundError(f"Artifact path does not exist: {path}")
    run_id, artifact = _registered_artifact(project, target, run_alias)
    if artifact is None or run_id is None:
        raise ValueError(f"Artifact is not registered in a run manifest: {path}")
    run_dir = project.runs_dir / run_id
    record_path = run_dir / "redaction_record.json"
    placeholder = Path(str(target) + ".redacted")
    if _artifact_already_redacted(record_path, placeholder, artifact):
        return RedactionRecord.model_validate(_read_json(record_path)), "already_redacted"
    original_hash = sha256_file(target) if target.exists() and target.is_file() else None
    redaction_id = _redaction_id("artifact", f"{run_id}:{artifact['artifact_id']}")
    placeholder_rel = _rel(project, placeholder)
    record_rel = _rel(project, record_path)
    placeholder_text = (
        "MechLedger redacted artifact placeholder\n"
        f"redaction_id: {redaction_id}\n"
        f"redaction_record: {record_rel}\n"
        f"original_path: {_rel(project, target)}\n"
    )
    record = RedactionRecord(
        redaction_id=redaction_id,
        target_type="artifact",
        target_path=_rel(project, target),
        reason=reason.strip(),
        created_at=now_utc(),
        created_by=os.environ.get("USER"),
        original_hash=original_hash,
        redacted_hash=None,
        placeholder_path=placeholder_rel,
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text(placeholder_text, encoding="utf-8")
    record.redacted_hash = sha256_file(placeholder)
    write_json(record_path, record.model_dump(mode="json"))
    if target.exists():
        target.unlink()
    _mark_artifact_redacted(project, run_id, artifact["artifact_id"], record)
    _append_redaction_debt(project, run_id, artifact, record)
    return record, "redacted"


def _registered_artifact(
    project: Project, target: Path, run_alias: str | None
) -> tuple[str | None, dict[str, Any] | None]:
    run_ids = [resolve_run_id(project, run_alias)] if run_alias else []
    if not run_ids:
        run_ids = sorted(path.name for path in project.runs_dir.iterdir() if path.is_dir())
    target_resolved = target.resolve()
    for run_id in run_ids:
        manifest = load_manifest(project.runs_dir / run_id)
        for artifact in manifest.get("artifacts", []):
            candidates = [
                artifact.get("resolved_path"),
                str(project.root / str(artifact.get("project_relative_path") or "")),
                str(project.root / str(artifact.get("original_path") or "")),
            ]
            for candidate in candidates:
                if candidate and Path(candidate).resolve() == target_resolved:
                    return run_id, artifact
    return None, None


def _mark_artifact_redacted(
    project: Project, run_id: str, artifact_id: str, record: RedactionRecord
) -> None:
    run_dir = project.runs_dir / run_id
    manifest = load_manifest(run_dir)
    for artifact in manifest.get("artifacts", []):
        if artifact.get("artifact_id") != artifact_id:
            continue
        artifact["redaction_status"] = "redacted"
        artifact["review_status"] = "redacted"
        artifact["placeholder_path"] = record.placeholder_path
        artifact["redaction_record_path"] = f".mechledger/runs/{run_id}/redaction_record.json"
        artifact["content_hash_status"] = "redacted"
        artifact["redacted_at"] = record.created_at
    write_manifest(run_dir, manifest)
    with (run_dir / "artifacts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "artifact_redacted",
                    "artifact_id": artifact_id,
                    "redaction_id": record.redaction_id,
                    "placeholder_path": record.placeholder_path,
                },
                sort_keys=True,
            )
            + "\n"
        )


def _append_redaction_debt(
    project: Project, run_id: str, artifact: dict[str, Any], record: RedactionRecord
) -> None:
    if artifact.get("claim_relevance") not in {"supporting", "required"}:
        return
    run_dir = project.runs_dir / run_id
    report_path = run_dir / "scientific_debt_report.json"
    payload = (
        _read_json(report_path)
        if report_path.exists()
        else _empty_debt_report(project, run_id)
    )
    claim_id = _claim_for_run(project, run_id)
    debt_id = f"REDACTED-{artifact.get('artifact_id')}"
    debts = payload.setdefault("debts", [])
    if any(debt.get("debt_id") == debt_id for debt in debts):
        return
    debts.append(
        {
            "debt_id": debt_id,
            "debt_type": "redacted_supporting_evidence",
            "severity": "serious",
            "claim_id": claim_id,
            "run_id": run_id,
            "experiment_id": payload.get("experiment_id"),
            "evidence_paths": [record.placeholder_path or record.target_path],
            "message": "Supporting or required evidence was redacted.",
            "required_resolution": "Replace evidence or review claim support.",
            "status": "open",
            "waiver_decision_id": None,
            "created_at": now_utc(),
            "resolved_at": None,
            "assessment_id": "redaction",
        }
    )
    payload["warnings"] = [
        debt
        for debt in debts
        if debt.get("status") == "open" and debt.get("severity") != "blocking"
    ]
    payload["blockers"] = [
        debt
        for debt in debts
        if debt.get("status") == "open" and debt.get("severity") == "blocking"
    ]
    write_json(report_path, payload)
    md_lines = [
        f"# Scientific Debt Report for {run_id}",
        "",
        payload.get("summary") or "",
        "",
        "## Debts",
    ]
    for debt in debts:
        md_lines.append(
            f"- {debt.get('debt_id')} [{debt.get('severity')}/{debt.get('status')}] "
            f"{debt.get('debt_type')}: {debt.get('message')}"
        )
    (run_dir / "scientific_debt_report.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )


def _claim_for_run(project: Project, run_id: str) -> str | None:
    ledger = parse_claim_ledger(project.resolve(project.config.default_claim_ledger))
    for claim in sorted(ledger.claims.values(), key=lambda item: item.claim_id):
        if run_id in claim.linked_runs:
            return claim.claim_id
    return None


def _artifact_already_redacted(
    record_path: Path, placeholder: Path, artifact: dict[str, Any]
) -> bool:
    return (
        record_path.exists()
        and placeholder.exists()
        and artifact.get("redaction_status") == "redacted"
    )


def _resolve_project_local_path(project: Project, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (project.root / path).resolve()
    try:
        resolved.relative_to(project.root.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing to redact path outside project: {path}") from exc
    return resolved


def _redaction_id(target_type: str, object_id: str) -> str:
    safe = "".join(char if char.isalnum() else "-" for char in object_id).strip("-")
    return f"RED-{target_type.upper()}-{safe[:80]}"


def _empty_debt_report(project: Project, run_id: str) -> dict[str, Any]:
    run_json = _read_json(project.runs_dir / run_id / "run.json")
    return {
        "report_id": f"SDR-{run_id}",
        "run_id": run_id,
        "experiment_id": run_json.get("experiment_id"),
        "generated_at": now_utc(),
        "evaluated_assessments": ["redaction"],
        "threshold_sources": [],
        "clean_candidate_support": False,
        "summary": "redaction debt remains",
        "debts": [],
        "warnings": [],
        "blockers": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object.")
    return payload


def _rel(project: Project, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
