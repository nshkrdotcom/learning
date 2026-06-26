from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mechledger.alias import rebuild_alias_cache
from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.core.decision_log import parse_decision_log
from mechledger.core.experiment_spec import parse_experiment_spec
from mechledger.core.research_log import parse_research_log
from mechledger.core.run_ledger import parse_run_ledger
from mechledger.project import Project

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
  claim_id TEXT PRIMARY KEY,
  status TEXT,
  title TEXT,
  file TEXT,
  line INTEGER,
  block_hash TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
  status TEXT,
  title TEXT,
  file TEXT,
  line INTEGER
);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY,
  status TEXT,
  title TEXT,
  file TEXT,
  line INTEGER
);
CREATE TABLE IF NOT EXISTS research_log_entries (
  entry_id TEXT PRIMARY KEY,
  date TEXT,
  file TEXT,
  line INTEGER
);
CREATE TABLE IF NOT EXISTS run_ledger_rows (
  run_id TEXT PRIMARY KEY,
  status TEXT,
  experiment_id TEXT,
  row_json TEXT
);
CREATE TABLE IF NOT EXISTS local_runs (
  run_id TEXT PRIMARY KEY,
  status TEXT,
  run_class TEXT,
  experiment_id TEXT,
  path TEXT,
  indexed_status TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT,
  run_id TEXT,
  path TEXT,
  claim_relevance TEXT,
  review_status TEXT,
  PRIMARY KEY(run_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS scientific_debt_records (
  debt_id TEXT,
  run_id TEXT,
  severity TEXT,
  status TEXT,
  debt_type TEXT,
  payload TEXT,
  PRIMARY KEY(run_id, debt_id)
);
CREATE TABLE IF NOT EXISTS draft_findings (
  finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
  file TEXT,
  line INTEGER,
  claim_id TEXT,
  violation_type TEXT,
  severity TEXT
);
CREATE TABLE IF NOT EXISTS aliases (
  run_id TEXT PRIMARY KEY,
  timestamp TEXT,
  experiment_id TEXT,
  slug TEXT
);
"""


def index_path(project: Project) -> Path:
    project.mechledger_dir.mkdir(parents=True, exist_ok=True)
    try:
        test = project.mechledger_dir / ".write_test"
        test.write_text("", encoding="utf-8")
        test.unlink()
        return project.mechledger_dir / "index.sqlite"
    except OSError:
        import hashlib
        import tempfile

        digest = hashlib.sha256(str(project.root).encode()).hexdigest()[:12]
        temp_root = Path(tempfile.gettempdir()) / f"mechledger_cache_{digest}"
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root / "index.sqlite"


def validate_project(project: Project) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    data: dict[str, Any] = {
        "claims": {},
        "decisions": {},
        "experiments": {},
        "research_entries": [],
        "run_ledger_rows": [],
        "local_runs": [],
    }
    try:
        claim_ledger = parse_claim_ledger(project.root / project.config.default_claim_ledger)
        data["claims"] = claim_ledger.claims
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    try:
        decisions = parse_decision_log(project.root / project.config.default_decision_log)
        data["decisions"] = decisions.decisions
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    for experiment_path in sorted((project.root / "research/experiments").glob("*.md")):
        if experiment_path.name.startswith("TEMPLATE_"):
            continue
        try:
            spec = parse_experiment_spec(experiment_path)
            data["experiments"][spec.experiment_id] = spec
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    try:
        data["research_entries"] = parse_research_log(
            project.root / project.config.default_research_log
        ).entries
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    try:
        data["run_ledger_rows"] = parse_run_ledger(
            project.root / project.config.default_run_ledger
        ).rows
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    for run_json in sorted(project.runs_dir.glob("*/run.json")):
        try:
            data["local_runs"].append(json.loads(run_json.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ERROR {run_json}\nRule: run.json.invalid\n{exc}")
        run_dir = run_json.parent
        for jsonl_name in ("events.jsonl", "metrics.jsonl", "artifacts.jsonl"):
            _validate_jsonl(run_dir / jsonl_name, errors)
        _validate_json(run_dir / "artifact_manifest.json", "artifact_manifest.invalid", errors)
    for report in sorted(project.runs_dir.glob("*/scientific_debt_report.json")):
        try:
            json.loads(report.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ERROR {report}\nRule: scientific_debt_report.invalid\n{exc}")
    return errors, data


def _validate_json(path: Path, rule: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(
            f"ERROR {path}\nRule: {rule}\nMissing required run metadata file."
        )
        return
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ERROR {path}\nRule: {rule}\n{exc}")


def _validate_jsonl(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(
            f"ERROR {path}\nRule: run.jsonl.missing\nMissing required run JSONL file."
        )
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                f"ERROR {path}:{line_number}\n"
                "Rule: run.jsonl.invalid\n"
                f"Malformed JSONL row: {exc.msg}\n"
                "Suggested fix: rewrite the row as one complete JSON object per line."
            )
            continue
        if not isinstance(row, dict):
            errors.append(
                f"ERROR {path}:{line_number}\n"
                "Rule: run.jsonl.object\n"
                "JSONL rows must be objects.\n"
                "Suggested fix: write an object like {\"event_type\": \"...\"}."
            )


def rebuild_index(project: Project) -> tuple[Path, list[str]]:
    errors, data = validate_project(project)
    if errors:
        return index_path(project), errors
    db_path = index_path(project)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for table in [
            "claims",
            "decisions",
            "experiments",
            "research_log_entries",
            "run_ledger_rows",
            "local_runs",
            "artifacts",
            "scientific_debt_records",
            "draft_findings",
            "aliases",
        ]:
            conn.execute(f"DELETE FROM {table}")
        for claim in data["claims"].values():
            conn.execute(
                "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?)",
                (
                    claim.claim_id,
                    claim.status.value,
                    claim.title or claim.heading_title,
                    claim.file,
                    claim.line,
                    claim.block_hash,
                ),
            )
        for decision in data["decisions"].values():
            conn.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.status,
                    decision.title,
                    decision.file,
                    decision.line,
                ),
            )
        for spec in data["experiments"].values():
            conn.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
                (spec.experiment_id, spec.status, spec.title, spec.file, spec.line),
            )
        for entry in data["research_entries"]:
            conn.execute(
                "INSERT INTO research_log_entries VALUES (?, ?, ?, ?)",
                (entry.entry_id, entry.date, entry.file, entry.line),
            )
        for row in data["run_ledger_rows"]:
            conn.execute(
                "INSERT INTO run_ledger_rows VALUES (?, ?, ?, ?)",
                (
                    row["run_id"],
                    row.get("status"),
                    row.get("phase"),
                    json.dumps(row, sort_keys=True),
                ),
            )
        for run in data["local_runs"]:
            indexed_status = _indexed_run_status(project, run)
            conn.execute(
                "INSERT INTO local_runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run["run_id"],
                    run.get("status"),
                    run.get("run_class"),
                    run.get("experiment_id"),
                    str(project.runs_dir / run["run_id"]),
                    indexed_status,
                ),
            )
            _index_artifacts_and_debt(conn, project, run["run_id"])
        conn.commit()
    rebuild_alias_cache(project)
    return db_path, []


def _index_artifacts_and_debt(conn: sqlite3.Connection, project: Project, run_id: str) -> None:
    manifest = project.runs_dir / run_id / "artifact_manifest.json"
    if manifest.exists():
        for artifact in json.loads(manifest.read_text(encoding="utf-8")).get("artifacts", []):
            conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
                (
                    artifact["artifact_id"],
                    run_id,
                    artifact.get("project_relative_path") or artifact.get("original_path"),
                    artifact.get("claim_relevance"),
                    artifact.get("review_status"),
                ),
            )
    report = project.runs_dir / run_id / "scientific_debt_report.json"
    if report.exists():
        payload = json.loads(report.read_text(encoding="utf-8"))
        for debt in payload.get("debts", []):
            conn.execute(
                "INSERT INTO scientific_debt_records VALUES (?, ?, ?, ?, ?, ?)",
                (
                    debt["debt_id"],
                    run_id,
                    debt["severity"],
                    debt["status"],
                    debt["debt_type"],
                    json.dumps(debt, sort_keys=True),
                ),
            )


def _indexed_run_status(project: Project, run: dict[str, Any]) -> str:
    if run.get("status") != "running":
        return run.get("status") or "unknown"
    heartbeat = project.runs_dir / run["run_id"] / "heartbeat.json"
    timestamp: str | None = None
    if heartbeat.exists():
        try:
            timestamp = str(
                json.loads(heartbeat.read_text(encoding="utf-8")).get(
                    "last_heartbeat_at"
                )
                or ""
            )
        except Exception:  # noqa: BLE001
            return "interrupted_indexed"
    else:
        timestamp = _latest_event_timestamp(project.runs_dir / run["run_id"]) or str(
            run.get("started_at") or ""
        )
    if not timestamp:
        return "interrupted_indexed"
    heartbeat_at = _parse_utc(timestamp)
    if heartbeat_at is None:
        return "interrupted_indexed"
    if (datetime.now(UTC) - heartbeat_at).total_seconds() > 120:
        return "interrupted_indexed"
    return "running"


def _latest_event_timestamp(run_dir: Path) -> str | None:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return None
    latest: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp = payload.get("timestamp")
        if isinstance(timestamp, str):
            latest = timestamp
    return latest


def _parse_utc(value: str) -> datetime | None:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
