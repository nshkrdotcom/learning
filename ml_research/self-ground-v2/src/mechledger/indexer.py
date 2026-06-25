from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from mechledger.io import read_json
from mechledger.models import RUN_LEDGER_COLUMNS
from mechledger.parsers import (
    LedgerParseError,
    parse_claim_ledger,
    parse_decision_log,
    parse_experiment_spec,
)
from mechledger.paths import index_path


@dataclass(slots=True)
class ProjectIndex:
    claim_count_by_status: dict[str, int]
    decision_count: int
    experiment_count: int
    run_count: int
    debt_count_by_severity: dict[str, int]


@dataclass(slots=True)
class CheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def index_project(project_root: str | Path) -> ProjectIndex:
    project_root = Path(project_root)
    result = _load_project(project_root)
    db_path = index_path(project_root)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            drop table if exists claims;
            drop table if exists decisions;
            drop table if exists experiments;
            drop table if exists runs;
            create table claims (
                claim_id text primary key,
                status text,
                title text,
                file text,
                line int
            );
            create table decisions (
                decision_id text primary key,
                status text,
                title text,
                file text,
                line int
            );
            create table experiments (
                experiment_id text primary key,
                status text,
                title text,
                file text,
                line int
            );
            create table runs (
                run_id text primary key,
                status text,
                run_class text,
                experiment_id text
            );
            """
        )
        for claim in result["claims"].claims.values():
            conn.execute(
                "insert into claims values (?, ?, ?, ?, ?)",
                (claim.claim_id, claim.status, claim.title, str(claim.file), claim.line),
            )
        for decision in result["decisions"].decisions.values():
            conn.execute(
                "insert into decisions values (?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.status,
                    decision.title,
                    str(decision.file),
                    decision.line,
                ),
            )
        for spec in result["experiments"]:
            conn.execute(
                "insert into experiments values (?, ?, ?, ?, ?)",
                (spec.experiment_id, spec.status, spec.title, str(spec.file), spec.line),
            )
        for run in result["runs"]:
            conn.execute(
                "insert into runs values (?, ?, ?, ?)",
                (run["run_id"], run.get("status"), run.get("run_class"), run.get("experiment_id")),
            )
    claim_counts: dict[str, int] = {}
    for claim in result["claims"].claims.values():
        claim_counts[claim.status] = claim_counts.get(claim.status, 0) + 1
    debt_counts: dict[str, int] = {}
    for report in result["debt_reports"]:
        for debt in report.get("debts", []):
            severity = str(debt.get("severity"))
            debt_counts[severity] = debt_counts.get(severity, 0) + 1
    return ProjectIndex(
        claim_count_by_status=claim_counts,
        decision_count=len(result["decisions"].decisions),
        experiment_count=len(result["experiments"]),
        run_count=len(result["runs"]),
        debt_count_by_severity=debt_counts,
    )


def check_project(project_root: str | Path) -> CheckResult:
    try:
        index_project(project_root)
    except (LedgerParseError, OSError, csv.Error, ValueError) as exc:
        return CheckResult(ok=False, errors=[str(exc)])
    return CheckResult(ok=True)


def _load_project(project_root: Path) -> dict:
    claims = parse_claim_ledger(project_root / "research" / "logs" / "claim_ledger.md")
    decisions = parse_decision_log(project_root / "research" / "logs" / "decision_log.md")
    experiments = [
        parse_experiment_spec(path)
        for path in sorted((project_root / "research" / "experiments").glob("*.md"))
        if not path.name.startswith("TEMPLATE")
    ]
    _validate_run_ledger(project_root / "research" / "logs" / "run_ledger.csv")
    runs = []
    debt_reports = []
    for run_json in sorted((project_root / ".mechledger" / "runs").glob("*/run.json")):
        runs.append(read_json(run_json))
        report_path = run_json.parent / "scientific_debt_report.json"
        if report_path.exists():
            debt_reports.append(read_json(report_path))
    return {
        "claims": claims,
        "decisions": decisions,
        "experiments": experiments,
        "runs": runs,
        "debt_reports": debt_reports,
    }


def _validate_run_ledger(path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return
    if header != RUN_LEDGER_COLUMNS:
        raise ValueError(
            f"ERROR {path}:1\n"
            "Rule: run_ledger.header\n"
            "Run ledger header does not match the canonical schema."
        )
