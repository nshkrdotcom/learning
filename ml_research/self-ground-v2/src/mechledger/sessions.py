from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mechledger.core.decision_log import parse_decision_log
from mechledger.inspection import relative_to_root, sha256_file, write_json
from mechledger.project import Project, now_utc

SESSION_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def start_session(project: Project, title: str) -> dict[str, Any]:
    if not title.strip():
        raise ValueError("Session title must not be empty.")
    session_id = _new_session_id(project, title)
    record = {
        "session_id": session_id,
        "title": title,
        "created_at": now_utc(),
        "closed_at": None,
        "status": "open",
        "linked_claims": [],
        "linked_runs": [],
        "linked_experiments": [],
        "linked_decisions": [],
        "notes": [],
        "attached_paths": [],
        "generated_outputs": [],
        "review_decision_id": None,
        "review_status": None,
        "warnings": [],
    }
    _write_session(project, record)
    return record


def add_session_note(project: Project, session_id: str, text: str) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("Session note text must not be empty.")
    record = load_session(project, session_id)
    record["notes"].append({"created_at": now_utc(), "text": text})
    _write_session(project, record)
    return record


def attach_session_path(project: Project, session_id: str, path: Path) -> dict[str, Any]:
    record = load_session(project, session_id)
    source = path if path.is_absolute() else project.root / path
    attachment: dict[str, Any] = {
        "path": relative_to_root(project, source),
        "exists": source.exists(),
        "inside_project": _inside_project(project, source),
        "sha256": None,
        "byte_size": None,
    }
    if source.exists() and source.is_file():
        attachment["sha256"] = sha256_file(source)
        attachment["byte_size"] = source.stat().st_size
    record["attached_paths"].append(attachment)
    _write_session(project, record)
    return record


def close_session_record(project: Project, session_id: str) -> dict[str, Any]:
    record = load_session(project, session_id)
    if record.get("status") == "open":
        record["status"] = "closed"
        record["closed_at"] = now_utc()
    _write_session(project, record)
    _write_summary(project, record)
    return record


def review_session(
    project: Project,
    session_id: str,
    *,
    accept: bool,
    reject: bool,
    decision_id: str | None,
) -> dict[str, Any]:
    if accept == reject:
        raise ValueError("Use exactly one of --accept or --reject.")
    record = load_session(project, session_id)
    if accept:
        if not decision_id:
            raise ValueError("Accepting a session requires an accepted decision.")
        decision_log = parse_decision_log(project.resolve(project.config.default_decision_log))
        decision = decision_log.decisions.get(decision_id)
        if decision is None or decision.status != "accepted":
            raise ValueError(
                f"Session acceptance requires an accepted decision: {decision_id}"
            )
        record["status"] = "accepted"
        record["review_status"] = "accepted"
        record["review_decision_id"] = decision_id
    else:
        record["status"] = "rejected"
        record["review_status"] = "rejected"
        record["review_decision_id"] = decision_id
    _write_session(project, record)
    _write_summary(project, record)
    return record


def list_sessions(project: Project) -> list[dict[str, Any]]:
    records = []
    for path in sorted(_session_root(project).glob("*/session.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    records.sort(key=lambda item: str(item.get("session_id")))
    return records


def load_session(project: Project, session_id: str) -> dict[str, Any]:
    path = _session_path(project, session_id)
    if not path.exists():
        raise FileNotFoundError(f"Unknown session: {session_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: session JSON must be an object.")
    return payload


def session_path(project: Project, session_id: str) -> Path:
    return _session_path(project, session_id)


def _new_session_id(project: Project, title: str) -> str:
    base = SESSION_ID_SAFE.sub("-", title.lower()).strip("-")[:32] or "session"
    stamp = now_utc().replace(":", "").replace("-", "").replace("T", "-").replace("Z", "")
    candidate = f"S{stamp}-{base}"
    root = _session_root(project)
    if not (root / candidate).exists():
        return candidate
    counter = 2
    while (root / f"{candidate}-{counter}").exists():
        counter += 1
    return f"{candidate}-{counter}"


def _session_root(project: Project) -> Path:
    return project.mechledger_dir / "copilot"


def _session_path(project: Project, session_id: str) -> Path:
    return _session_root(project) / session_id / "session.json"


def _write_session(project: Project, record: dict[str, Any]) -> None:
    write_json(_session_path(project, str(record["session_id"])), record)


def _write_summary(project: Project, record: dict[str, Any]) -> None:
    path = _session_root(project) / str(record["session_id"]) / "summary.md"
    lines = [
        f"# Session {record['session_id']}",
        "",
        f"Title: {record.get('title')}",
        f"Status: {record.get('status')}",
        f"Created: {record.get('created_at')}",
        f"Closed: {record.get('closed_at') or 'not closed'}",
        "",
        "## Notes",
    ]
    for note in record.get("notes") or []:
        lines.append(f"- {note.get('created_at')}: {note.get('text')}")
    lines.extend(["", "## Attachments"])
    for attachment in record.get("attached_paths") or []:
        lines.append(
            f"- {attachment.get('path')} sha256={attachment.get('sha256') or 'missing'}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _inside_project(project: Project, path: Path) -> bool:
    try:
        path.resolve().relative_to(project.root.resolve())
    except ValueError:
        return False
    return True
