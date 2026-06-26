from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from mechledger.alias import resolve_run_id
from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.core.decision_log import parse_decision_log
from mechledger.draftguard import check_draft_files
from mechledger.inspection import sha256_file, write_json
from mechledger.prediction import canonical_prediction_hash
from mechledger.project import Project, now_utc


class TamperRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    tamper_id: str
    object_type: str
    object_id: str
    expected_hash: str | None
    observed_hash: str | None
    detected_at: str
    severity: Literal["info", "warning", "blocking"]
    consequence: Literal[
        "block_scoring",
        "invalidate_experiment",
        "require_review",
        "log_only",
    ]
    resolution_status: Literal[
        "unresolved",
        "accepted_as_new_version",
        "reverted",
        "waived",
    ] = "unresolved"
    resolution_note: str | None = None
    resolution_decision_id: str | None = None


def check_integrity(project: Project, run_alias: str | None = None) -> list[TamperRecord]:
    run_id = resolve_run_id(project, run_alias) if run_alias else None
    existing = _read_records(project_record_path(project))
    baselines = _read_baselines(project)
    current = _collect_findings(project, baselines, run_id=run_id)
    merged = _merge_records(existing, current)
    _write_records(project_record_path(project), merged)
    _write_baselines(project, baselines)
    if run_id:
        run_records = [
            record
            for record in merged
            if run_id in record.object_id or f"/{run_id}/" in record.object_id
        ]
        _write_records(run_record_path(project, run_id), run_records)
    return [record for record in merged if record.resolution_status == "unresolved"]


def resolve_tamper_record(
    project: Project,
    tamper_id: str,
    *,
    decision_id: str,
    status: str,
    note: str,
) -> TamperRecord:
    if status not in {
        "accepted_as_new_version",
        "reverted",
        "waived",
    }:
        raise ValueError("--status must be accepted_as_new_version, reverted, or waived.")
    if status in {"accepted_as_new_version", "waived"}:
        decision_log = parse_decision_log(project.resolve(project.config.default_decision_log))
        decision = decision_log.decisions.get(decision_id)
        if decision is None or decision.status != "accepted":
            raise ValueError(f"Resolving {status} requires an accepted decision: {decision_id}")
    records = _read_records(project_record_path(project))
    for index, record in enumerate(records):
        if record.tamper_id != tamper_id:
            continue
        updated = record.model_copy(
            update={
                "resolution_status": status,
                "resolution_note": note,
                "resolution_decision_id": decision_id,
            }
        )
        records[index] = updated
        _write_records(project_record_path(project), records)
        return updated
    raise ValueError(f"Unknown tamper record: {tamper_id}")


def unresolved_tamper_count(project: Project) -> int:
    return sum(
        1
        for record in _read_records(project_record_path(project))
        if record.resolution_status == "unresolved"
    )


def project_record_path(project: Project) -> Path:
    return project.mechledger_dir / "integrity/tamper_records.jsonl"


def run_record_path(project: Project, run_id: str) -> Path:
    return project.runs_dir / run_id / "integrity_tamper_records.jsonl"


def _collect_findings(
    project: Project, baselines: dict[str, Any], *, run_id: str | None
) -> list[TamperRecord]:
    findings: list[TamperRecord] = []
    claims = parse_claim_ledger(project.resolve(project.config.default_claim_ledger))
    findings.extend(_prediction_findings(project))
    findings.extend(_claim_proposal_findings(project, baselines, claims.claims, run_id=run_id))
    findings.extend(_decision_status_findings(project, baselines))
    findings.extend(_draft_findings(project))
    findings.extend(_run_ledger_proposal_findings(project, baselines, run_id=run_id))
    return findings


def _prediction_findings(project: Project) -> list[TamperRecord]:
    findings = []
    for path in sorted(
        list((project.root / "research/predictions").glob("**/*.json"))
        + list((project.root / "predictions").glob("**/*.json"))
    ):
        payload = _read_json(path)
        expected = payload.get("locked_content_hash")
        if not expected:
            continue
        observed = canonical_prediction_hash(payload)
        if observed != expected:
            prediction_id = str(payload.get("prediction_id") or path.name)
            findings.append(
                _record(
                    "prediction",
                    prediction_id,
                    str(expected),
                    observed,
                    severity="blocking",
                    consequence="block_scoring",
                )
            )
    return findings


def _claim_proposal_findings(
    project: Project,
    baselines: dict[str, Any],
    claims: dict[str, Any],
    *,
    run_id: str | None,
) -> list[TamperRecord]:
    findings: list[TamperRecord] = []
    artifact_baselines = baselines.setdefault("claim_proposal_artifacts", {})
    for proposal_path in sorted(project.runs_dir.glob("*/claim_update_proposal.json")):
        proposal_run_id = proposal_path.parent.name
        if run_id and proposal_run_id != run_id:
            continue
        proposal = _read_json(proposal_path)
        claim_id = proposal.get("target_claim_id")
        expected_claim_hash = proposal.get("expected_claim_block_hash")
        if claim_id and expected_claim_hash and claim_id in claims:
            observed = claims[str(claim_id)].block_hash
            if observed != expected_claim_hash:
                findings.append(
                    _record(
                        "claim_yaml_block",
                        str(claim_id),
                        str(expected_claim_hash),
                        observed,
                        severity="blocking",
                        consequence="require_review",
                    )
                )
        expected_artifact_hashes = proposal.get("expected_artifact_hashes") or {}
        for raw_path in sorted(proposal.get("supporting_artifact_paths") or []):
            artifact_path = (project.root / str(raw_path)).resolve()
            observed = sha256_file(artifact_path) if artifact_path.exists() else None
            key = f"{proposal.get('proposal_id') or proposal_path}:{raw_path}"
            expected = expected_artifact_hashes.get(raw_path) or artifact_baselines.get(key)
            if expected is None:
                artifact_baselines[key] = observed
                continue
            if observed != expected:
                findings.append(
                    _record(
                        "artifact",
                        f"{proposal_run_id}:{raw_path}",
                        expected,
                        observed,
                        severity="blocking",
                        consequence="require_review",
                    )
                )
    return findings


def _decision_status_findings(project: Project, baselines: dict[str, Any]) -> list[TamperRecord]:
    findings: list[TamperRecord] = []
    decision_baselines = baselines.setdefault("decision_status", {})
    decisions = parse_decision_log(project.resolve(project.config.default_decision_log)).decisions
    for decision_id, decision in sorted(decisions.items()):
        observed = _hash_json({"decision_id": decision_id, "status": decision.status})
        expected = decision_baselines.get(decision_id)
        if expected is None:
            decision_baselines[decision_id] = observed
            continue
        if observed != expected:
            findings.append(
                _record(
                    "decision_status",
                    decision_id,
                    expected,
                    observed,
                    severity="blocking",
                    consequence="require_review",
                )
            )
    return findings


def _draft_findings(project: Project) -> list[TamperRecord]:
    paper_root = project.root / "research/paper"
    files = sorted(
        path
        for pattern in ("**/*.md", "**/*.markdown", "**/*.tex")
        for path in paper_root.glob(pattern)
        if path.is_file()
    )
    if not files:
        return []
    result = check_draft_files(
        files, claim_ledger_path=project.resolve(project.config.default_claim_ledger)
    )
    findings = []
    for violation in result.violations:
        if violation.violation_type != "unknown_claim":
            continue
        object_id = f"{violation.file}:{violation.line}:{violation.claim_id}"
        findings.append(
            _record(
                "draft_claim_tag",
                object_id,
                None,
                violation.claim_id,
                severity="blocking",
                consequence="require_review",
            )
        )
    return findings


def _run_ledger_proposal_findings(
    project: Project, baselines: dict[str, Any], *, run_id: str | None
) -> list[TamperRecord]:
    findings: list[TamperRecord] = []
    proposal_baselines = baselines.setdefault("run_ledger_proposals", {})
    for path in sorted(project.runs_dir.glob("*/run_ledger_row.csv")):
        proposal_run_id = path.parent.name
        if run_id and proposal_run_id != run_id:
            continue
        observed = sha256_file(path)
        expected = proposal_baselines.get(proposal_run_id)
        if expected is None:
            proposal_baselines[proposal_run_id] = observed
            continue
        if observed != expected:
            findings.append(
                _record(
                    "run_ledger_proposal",
                    proposal_run_id,
                    expected,
                    observed,
                    severity="blocking",
                    consequence="require_review",
                )
            )
    return findings


def _merge_records(
    existing: list[TamperRecord], current: list[TamperRecord]
) -> list[TamperRecord]:
    by_id = {record.tamper_id: record for record in existing}
    for record in current:
        previous = by_id.get(record.tamper_id)
        if previous and previous.resolution_status != "unresolved":
            continue
        if previous:
            by_id[record.tamper_id] = previous.model_copy(
                update={"observed_hash": record.observed_hash}
            )
        else:
            by_id[record.tamper_id] = record
    return sorted(by_id.values(), key=lambda item: item.tamper_id)


def _record(
    object_type: str,
    object_id: str,
    expected_hash: str | None,
    observed_hash: str | None,
    *,
    severity: Literal["info", "warning", "blocking"],
    consequence: Literal[
        "block_scoring",
        "invalidate_experiment",
        "require_review",
        "log_only",
    ],
) -> TamperRecord:
    tamper_id = _tamper_id(object_type, object_id, expected_hash)
    return TamperRecord(
        tamper_id=tamper_id,
        object_type=object_type,
        object_id=object_id,
        expected_hash=expected_hash,
        observed_hash=observed_hash,
        detected_at=now_utc(),
        severity=severity,
        consequence=consequence,
    )


def _tamper_id(object_type: str, object_id: str, expected_hash: str | None) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "object_type": object_type,
                "object_id": object_id,
                "expected_hash": expected_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"TP-{digest}"


def _read_records(path: Path) -> list[TamperRecord]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(TamperRecord.model_validate(json.loads(line)))
    return sorted(records, key=lambda item: item.tamper_id)


def _write_records(path: Path, records: list[TamperRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(record.model_dump(mode="json"), sort_keys=True)
        for record in sorted(records, key=lambda item: item.tamper_id)
    ]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _read_baselines(project: Project) -> dict[str, Any]:
    path = project.mechledger_dir / "integrity/baselines.json"
    if not path.exists():
        return {}
    return _read_json(path)


def _write_baselines(project: Project, baselines: dict[str, Any]) -> None:
    write_json(project.mechledger_dir / "integrity/baselines.json", baselines)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object.")
    return payload


def _hash_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
