from __future__ import annotations

import json
import sqlite3
import tomllib
from pathlib import Path
from typing import Any

SCHEMA_TABLES = {
    "projects",
    "events",
    "sessions",
    "cells",
    "objects",
    "object_versions",
    "tensor_spaces",
    "mechanistic_units",
    "example_bundles",
    "control_bundles",
    "artifacts",
    "artifact_versions",
    "lineage_edges",
    "evidence_edges",
    "runs",
    "run_events",
    "metrics",
    "hypotheses",
    "hypothesis_states",
    "hypothesis_transitions",
    "alternative_explanations",
    "prediction_locks",
    "preflight_reports",
    "verification_runs",
    "blocker_reports",
    "next_probe_plans",
    "mechanism_cards",
    "claims",
    "claim_evidence",
    "draft_claim_links",
    "decisions",
    "research_log_entries",
    "scientific_debt",
    "adapter_manifests",
    "backend_versions",
    "redaction_records",
    "exports",
}


def connect(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_schema(sqlite_path: Path) -> None:
    with connect(sqlite_path) as conn:
        conn.execute("pragma journal_mode=wal")
        conn.execute(
            """
            create table if not exists projects (
                project_ref text primary key,
                name text not null,
                root text not null,
                mechanism_dir text not null,
                schema_version integer not null,
                created_at text not null
            )
            """
        )
        conn.execute(
            """
            create table if not exists events (
                id integer primary key autoincrement,
                event_type text not null,
                created_at text not null,
                payload_json text not null
            )
            """
        )

        for table in sorted(SCHEMA_TABLES - {"projects", "events"}):
            conn.execute(
                f"""
                create table if not exists {table} (
                    ref text primary key,
                    payload_json text not null,
                    created_at text
                )
                """
            )


def insert_project(sqlite_path: Path, project: dict[str, Any]) -> None:
    with connect(sqlite_path) as conn:
        conn.execute(
            """
            insert or ignore into projects
            (project_ref, name, root, mechanism_dir, schema_version, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                project["project_ref"],
                project["name"],
                project["root"],
                project["mechanism_dir"],
                project["schema_version"],
                project["created_at"],
            ),
        )


def insert_event(sqlite_path: Path, event: dict[str, Any]) -> None:
    with connect(sqlite_path) as conn:
        conn.execute(
            "insert into events (event_type, created_at, payload_json) values (?, ?, ?)",
            (
                event["event_type"],
                event["created_at"],
                json.dumps(event["payload"], sort_keys=True),
            ),
        )


def insert_payload(sqlite_path: Path, table: str, ref: str, payload: dict[str, Any]) -> None:
    if table not in SCHEMA_TABLES:
        raise ValueError(f"unsupported table: {table}")
    with connect(sqlite_path) as conn:
        conn.execute(
            f"""
            insert or replace into {table} (ref, payload_json, created_at)
            values (?, ?, ?)
            """,
            (
                ref,
                json.dumps(payload, sort_keys=True),
                str(payload.get("created_at", "")) or None,
            ),
        )


def fetch_payload(sqlite_path: Path, table: str, ref: str) -> dict[str, Any]:
    if table not in SCHEMA_TABLES:
        raise ValueError(f"unsupported table: {table}")
    with connect(sqlite_path) as conn:
        row = conn.execute(f"select payload_json from {table} where ref = ?", (ref,)).fetchone()
    if row is None:
        raise KeyError(f"{table}:{ref}")
    return json.loads(row["payload_json"])


def existing_tables(sqlite_path: Path) -> set[str]:
    with connect(sqlite_path) as conn:
        return {
            row[0]
            for row in conn.execute("select name from sqlite_master where type='table'")
        }


def rebuild_sqlite_index(project: Any, *, output_path: Path | None = None) -> dict[str, Any]:
    """Rebuild a SQLite index from file-backed .mechanism records."""
    sqlite_path = output_path or project.mechanism_dir / "workbench.rebuilt.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()
    initialize_schema(sqlite_path)
    counts = {table: 0 for table in SCHEMA_TABLES}

    project_record = _project_record(project)
    insert_project(sqlite_path, project_record)
    counts["projects"] = 1

    for event in _read_jsonl(project.events_path):
        if {"event_type", "created_at", "payload"} <= set(event):
            insert_event(sqlite_path, event)
            counts["events"] += 1

    sessions_dir = project.mechanism_dir / "sessions"
    for session_dir in sorted(sessions_dir.glob("sess_*")) if sessions_dir.exists() else []:
        session_json = session_dir / "session.json"
        if session_json.exists():
            session = json.loads(session_json.read_text(encoding="utf-8"))
            insert_payload(sqlite_path, "sessions", session["session_ref"], session)
            counts["sessions"] += 1
        for cell in _read_jsonl(session_dir / "cells.jsonl"):
            insert_payload(sqlite_path, "cells", cell["cell_ref"], cell)
            counts["cells"] += 1
        for event in _read_jsonl(session_dir / "namespace_objects.jsonl"):
            if event.get("event") != "object_registered":
                continue
            payload = event.get("object_payload") or {
                "wb_ref": event["object_ref"],
                "wb_type": event.get("object_type", "WorkbenchObject"),
                "wb_version": "1",
                "parents": event.get("parents", []),
                "metadata": {},
            }
            insert_payload(sqlite_path, "objects", str(event["object_ref"]), payload)
            counts["objects"] += 1
            counts["lineage_edges"] += _insert_lineage_edges_from_event(sqlite_path, event)
        for artifact in _read_jsonl(session_dir / "artifacts.jsonl"):
            ref = artifact.get("artifact_ref") or artifact.get("ref")
            if ref:
                insert_payload(sqlite_path, "artifacts", str(ref), artifact)
                counts["artifacts"] += 1

    indexed_alternative_refs: set[str] = set()
    runs_dir = project.mechanism_dir / "runs"
    for run_dir in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        counts["runs"] += _insert_json_file(sqlite_path, run_dir / "run_manifest.json", "runs")
        counts["blocker_reports"] += _insert_json_file(
            sqlite_path, run_dir / "blocker_report.json", "blocker_reports"
        )
        counts["next_probe_plans"] += _insert_json_file(
            sqlite_path, run_dir / "next_probe.json", "next_probe_plans"
        )
        counts["mechanism_cards"] += _insert_json_file(
            sqlite_path, run_dir / "mechanism_card.json", "mechanism_cards"
        )
        counts["metrics"] += _insert_json_file(
            sqlite_path, run_dir / "control_metrics.json", "metrics"
        )
        counts["scientific_debt"] += _insert_json_file(
            sqlite_path, run_dir / "scientific_debt.json", "scientific_debt"
        )
        counts["alternative_explanations"] += _insert_json_file_once(
            sqlite_path,
            run_dir / "alternative_explanations.json",
            "alternative_explanations",
            indexed_alternative_refs,
        )

    hypotheses_dir = project.mechanism_dir / "hypotheses"
    for path in sorted(hypotheses_dir.glob("*.json")) if hypotheses_dir.exists() else []:
        if path.name.endswith("_lifecycle.json"):
            counts["hypothesis_states"] += _insert_json_file(
                sqlite_path, path, "hypothesis_states"
            )
        elif path.name.endswith("_alternatives.json"):
            counts["alternative_explanations"] += _insert_json_file_once(
                sqlite_path,
                path,
                "alternative_explanations",
                indexed_alternative_refs,
            )
        else:
            counts["hypotheses"] += _insert_json_file(sqlite_path, path, "hypotheses")
    transition_paths = (
        sorted(hypotheses_dir.glob("*_transitions.jsonl")) if hypotheses_dir.exists() else []
    )
    for path in transition_paths:
        for receipt in _read_jsonl(path):
            ref = receipt.get("wb_ref") or receipt.get("receipt_ref")
            if ref:
                insert_payload(sqlite_path, "hypothesis_transitions", str(ref), receipt)
                counts["hypothesis_transitions"] += 1

    for edge in _read_jsonl(project.mechanism_dir / "graph" / "evidence_edges.jsonl"):
        ref = edge.get("wb_ref") or edge.get("edge_ref")
        if ref:
            insert_payload(sqlite_path, "evidence_edges", str(ref), edge)
            counts["evidence_edges"] += 1

    from mwb.ledgers import index_ledgers, parse_research_ledgers

    parsed_ledgers = parse_research_ledgers(project.root / "research" / "logs")
    if not parsed_ledgers["errors"]:
        ledger_counts = index_ledgers(sqlite_path, parsed_ledgers)
        counts["claims"] += ledger_counts["claims"]
        counts["decisions"] += ledger_counts["decisions"]
        counts["research_log_entries"] += ledger_counts["research_log_entries"]
        counts["runs"] += ledger_counts["run_ledger_rows"]

    return {"status": "ok", "sqlite_path": str(sqlite_path), "counts": counts}


def _project_record(project: Any) -> dict[str, Any]:
    project_toml = project.mechanism_dir / "project.toml"
    created_at = ""
    if project_toml.exists():
        config = tomllib.loads(project_toml.read_text(encoding="utf-8"))
        created_at = str(config.get("project", {}).get("created_at", ""))
    return {
        "project_ref": project.project_ref,
        "name": project.name,
        "root": str(project.root),
        "mechanism_dir": str(project.mechanism_dir.relative_to(project.root)),
        "schema_version": project.schema_version,
        "created_at": created_at,
    }


def _insert_json_file(sqlite_path: Path, path: Path, table: str) -> int:
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    ref = (
        payload.get("wb_ref")
        or payload.get("run_ref")
        or payload.get("source_run_ref")
        or payload.get("mechanism_card_ref")
        or path.parent.name
    )
    insert_payload(sqlite_path, table, str(ref), payload)
    return 1


def _insert_json_file_once(
    sqlite_path: Path,
    path: Path,
    table: str,
    seen_refs: set[str],
) -> int:
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    ref = (
        payload.get("wb_ref")
        or payload.get("run_ref")
        or payload.get("source_run_ref")
        or payload.get("mechanism_card_ref")
        or path.parent.name
    )
    ref_text = str(ref)
    if ref_text in seen_refs:
        return 0
    seen_refs.add(ref_text)
    insert_payload(sqlite_path, table, ref_text, payload)
    return 1


def _insert_lineage_edges_from_event(sqlite_path: Path, event: dict[str, Any]) -> int:
    count = 0
    object_ref = str(event["object_ref"])
    cell_ref = str(event["cell_ref"])
    relation = "mutated_in_cell" if event.get("event") == "object_mutated" else "created_in_cell"
    edge = {
        "src_ref": cell_ref,
        "dst_ref": object_ref,
        "relation": relation,
        "session_ref": event.get("session_ref"),
        "cell_ref": cell_ref,
        "created_at": event.get("created_at"),
    }
    insert_payload(sqlite_path, "lineage_edges", stable_edge_ref(edge), edge)
    count += 1
    for parent_ref in event.get("parents", []):
        parent_edge = {
            "src_ref": str(parent_ref),
            "dst_ref": object_ref,
            "relation": "parent",
            "session_ref": event.get("session_ref"),
            "cell_ref": cell_ref,
            "created_at": event.get("created_at"),
        }
        insert_payload(sqlite_path, "lineage_edges", stable_edge_ref(parent_edge), parent_edge)
        count += 1
    return count


def stable_edge_ref(edge: dict[str, Any]) -> str:
    from mwb.refs import stable_ref

    return stable_ref("edge", edge["src_ref"], edge["dst_ref"], edge["relation"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
