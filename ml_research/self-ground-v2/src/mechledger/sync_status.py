from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.core.run_ledger import DEFAULT_RUN_LEDGER_COLUMNS
from mechledger.project import Project

ALIAS_REFERENCE = re.compile(r"^(latest(?::[0-9]+)?|#[0-9]+)$")


class SyncFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_type: str
    object_id: str
    path: str
    message: str
    severity: str = "blocking"


@dataclass(frozen=True)
class LocalRun:
    run_id: str
    run_dir: Path
    run_json: Path
    hash: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class RawAlias:
    run_id: str
    timestamp: str
    experiment_id: str
    slug: str
    line_number: int


def evaluate_sync(project: Project) -> list[SyncFinding]:
    local_runs = _local_runs(project)
    ledger_rows = _run_ledger_rows(project)
    claims = parse_claim_ledger(project.resolve(project.config.default_claim_ledger))
    aliases, malformed = _raw_aliases(project)
    findings: list[SyncFinding] = []
    local_ids = set(local_runs)
    ledger_ids = {row["run_id"] for row in ledger_rows if row.get("run_id")}

    for run_id in sorted(local_ids - ledger_ids):
        run = local_runs[run_id][0]
        findings.append(
            _finding(
                "local_run_missing_from_ledger",
                run_id,
                _rel(project, run.run_dir),
                "Local run directory is not represented in committed run_ledger.csv.",
            )
        )
    for run_id in sorted(ledger_ids - local_ids):
        findings.append(
            _finding(
                "ledger_run_missing_locally",
                run_id,
                project.config.default_run_ledger,
                "Committed run ledger row has no local run directory in this clone.",
            )
        )
    for claim in sorted(claims.claims.values(), key=lambda item: item.claim_id):
        for run_id in sorted(claim.linked_runs):
            if ALIAS_REFERENCE.fullmatch(run_id):
                findings.append(
                    _finding(
                        "claim_uses_run_alias",
                        f"{claim.claim_id}:{run_id}",
                        _rel(project, Path(claim.file)),
                        "Canonical claim ledger linked_runs must use full run IDs, not aliases.",
                    )
                )
            elif run_id not in ledger_ids:
                findings.append(
                    _finding(
                        "claim_run_missing_from_ledger",
                        f"{claim.claim_id}:{run_id}",
                        _rel(project, Path(claim.file)),
                        "Claim references a run absent from committed run_ledger.csv.",
                    )
                )
    for run_id, runs in sorted(local_runs.items()):
        if len(runs) > 1:
            paths = ", ".join(_rel(project, run.run_json) for run in runs)
            findings.append(
                _finding(
                    "duplicate_run_id",
                    run_id,
                    paths,
                    "Multiple local run.json files declare the same canonical run_id.",
                )
            )
            hashes = {run.hash for run in runs}
            if len(hashes) > 1:
                findings.append(
                    _finding(
                        "run_json_hash_mismatch",
                        run_id,
                        paths,
                        "Duplicate local run.json files for the same run_id differ.",
                    )
                )
    ledger_counts: dict[str, int] = {}
    for row in ledger_rows:
        ledger_counts[row["run_id"]] = ledger_counts.get(row["run_id"], 0) + 1
    for run_id, count in sorted(ledger_counts.items()):
        if count > 1:
            findings.append(
                _finding(
                    "duplicate_run_id",
                    run_id,
                    project.config.default_run_ledger,
                    "Committed run ledger contains duplicate run_id rows.",
                )
            )
    alias_counts: dict[str, list[RawAlias]] = {}
    for alias in aliases:
        alias_counts.setdefault(alias.run_id, []).append(alias)
        if alias.run_id not in local_ids:
            findings.append(
                _finding(
                    "alias_points_to_absent_run",
                    alias.run_id,
                    ".mechledger/alias_cache.txt",
                    "Alias cache record points to a local run directory absent on this clone.",
                )
            )
    for run_id, records in sorted(alias_counts.items()):
        if len(records) > 1:
            findings.append(
                _finding(
                    "duplicate_run_id",
                    run_id,
                    ".mechledger/alias_cache.txt",
                    "Alias cache contains duplicate canonical run_id records.",
                )
            )
    for line_number, line in malformed:
        findings.append(
            SyncFinding(
                finding_type="malformed_alias_cache_line",
                object_id=f"line:{line_number}",
                path=".mechledger/alias_cache.txt",
                message=f"Malformed alias cache line ignored: {line}",
                severity="warning",
            )
        )
    findings.sort(key=lambda item: (item.finding_type, item.object_id, item.path))
    return findings


def sync_counts(findings: list[SyncFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.finding_type] = counts.get(finding.finding_type, 0) + 1
    return dict(sorted(counts.items()))


def blocking_findings(findings: list[SyncFinding]) -> list[SyncFinding]:
    return [finding for finding in findings if finding.severity == "blocking"]


def format_sync_status(findings: list[SyncFinding]) -> str:
    counts = sync_counts(findings)
    lines = [f"blocking_findings: {len(blocking_findings(findings))}"]
    known = {
        "local_run_missing_from_ledger",
        "ledger_run_missing_locally",
        "claim_run_missing_from_ledger",
        "duplicate_run_id",
        "run_json_hash_mismatch",
        "alias_points_to_absent_run",
        "claim_uses_run_alias",
        "malformed_alias_cache_line",
    }
    for finding_type in sorted(known | set(counts)):
        lines.append(f"{finding_type}: {counts.get(finding_type, 0)}")
    return "\n".join(lines) + "\n"


def format_sync_diff(findings: list[SyncFinding]) -> str:
    if not findings:
        return "No sync findings.\n"
    lines = []
    for finding in findings:
        lines.append(
            f"{finding.finding_type}\t{finding.object_id}\t{finding.path}\t"
            f"{finding.message}"
        )
    return "\n".join(lines) + "\n"


def _local_runs(project: Project) -> dict[str, list[LocalRun]]:
    runs: dict[str, list[LocalRun]] = {}
    for run_json in sorted(project.runs_dir.glob("*/run.json")):
        payload = _read_json(run_json)
        run_id = str(payload.get("run_id") or run_json.parent.name)
        record = LocalRun(
            run_id=run_id,
            run_dir=run_json.parent,
            run_json=run_json,
            hash=hash_file(run_json),
            payload=payload,
        )
        runs.setdefault(run_id, []).append(record)
    return runs


def _run_ledger_rows(project: Project) -> list[dict[str, str]]:
    path = project.resolve(project.config.default_run_ledger)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DEFAULT_RUN_LEDGER_COLUMNS:
            raise ValueError(f"{path}: run ledger header is invalid.")
        return [dict(row) for row in reader]


def _raw_aliases(project: Project) -> tuple[list[RawAlias], list[tuple[int, str]]]:
    path = project.mechledger_dir / "alias_cache.txt"
    if not path.exists():
        return [], []
    aliases: list[RawAlias] = []
    malformed: list[tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split("\t")
        if len(parts) != 4 or not parts[0]:
            malformed.append((line_number, line))
            continue
        aliases.append(RawAlias(parts[0], parts[1], parts[2], parts[3], line_number))
    return aliases, malformed


def _finding(finding_type: str, object_id: str, path: str, message: str) -> SyncFinding:
    return SyncFinding(
        finding_type=finding_type,
        object_id=object_id,
        path=path,
        message=message,
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
