from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from mechledger.core.decision_log import parse_decision_log
from mechledger.core.research_log import parse_research_log
from mechledger.inspection import collect_project
from mechledger.project import Project, now_utc

QUESTION_HEADING = re.compile(r"^##\s+(Q[0-9]+[A-Za-z0-9_-]*)\s+-\s+(.+?)\s*$")


def list_questions(project: Project) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    research_log = parse_research_log(project.resolve(project.config.default_research_log))
    for entry in research_log.entries:
        for index, text in enumerate(entry.open_questions, start=1):
            questions.append(
                {
                    "question_id": f"{entry.entry_id}-Q{index}",
                    "text": text,
                    "status": "open",
                    "priority": "normal",
                    "linked_claims": entry.linked_claims,
                    "linked_runs": entry.linked_runs,
                    "linked_experiments": [],
                    "linked_decisions": entry.linked_decisions,
                    "created_at": entry.date,
                    "resolved_at": None,
                    "resolution": None,
                    "resolution_decision_id": None,
                    "source": "research_log",
                }
            )
    questions.extend(_read_question_file(project))
    questions.sort(key=lambda item: str(item.get("question_id")))
    return questions


def add_question(
    project: Project,
    *,
    text: str,
    claim: str | None = None,
    experiment: str | None = None,
    run: str | None = None,
    priority: str = "normal",
) -> dict[str, Any]:
    if priority not in {"low", "normal", "high"}:
        raise ValueError("--priority must be one of low, normal, high.")
    if not text.strip():
        raise ValueError("--text must not be empty.")
    snapshot = collect_project(project)
    if claim and claim not in snapshot.claims:
        raise ValueError(f"Unknown claim: {claim}")
    if experiment and experiment not in snapshot.experiments:
        raise ValueError(f"Unknown experiment: {experiment}")
    if run and run not in snapshot.runs:
        raise ValueError(f"Unknown run: {run}")
    existing = _read_question_file(project)
    question_id = _next_question_id(existing)
    record = {
        "question_id": question_id,
        "text": text,
        "status": "open",
        "priority": priority,
        "linked_claims": [claim] if claim else [],
        "linked_runs": [run] if run else [],
        "linked_experiments": [experiment] if experiment else [],
        "linked_decisions": [],
        "created_at": now_utc(),
        "resolved_at": None,
        "resolution": None,
        "resolution_decision_id": None,
        "source": "open_questions",
    }
    existing.append(record)
    _write_question_file(project, existing)
    return record


def resolve_question(
    project: Project,
    question_id: str,
    *,
    decision: str,
    resolution: str,
) -> dict[str, Any]:
    records = _read_question_file(project)
    record = next((item for item in records if item.get("question_id") == question_id), None)
    if record is None:
        raise ValueError(f"Unknown canonical question: {question_id}")
    decision_log = parse_decision_log(project.resolve(project.config.default_decision_log))
    decision_record = decision_log.decisions.get(decision)
    if decision_record is None or decision_record.status != "accepted":
        raise ValueError(f"Resolving question requires an accepted decision: {decision}")
    record["status"] = "resolved"
    record["resolved_at"] = now_utc()
    record["resolution"] = resolution
    record["resolution_decision_id"] = decision
    record["linked_decisions"] = sorted(
        {*(record.get("linked_decisions") or []), decision}
    )
    _write_question_file(project, records)
    return record


def link_question(
    project: Project,
    question_id: str,
    *,
    claim: str | None = None,
    experiment: str | None = None,
    run: str | None = None,
    decision: str | None = None,
) -> dict[str, Any]:
    if not any([claim, experiment, run, decision]):
        raise ValueError("Provide at least one link: --claim, --experiment, --run, or --decision.")
    records = _read_question_file(project)
    record = next((item for item in records if item.get("question_id") == question_id), None)
    if record is None:
        raise ValueError(f"Unknown canonical question: {question_id}")
    snapshot = collect_project(project)
    if claim and claim not in snapshot.claims:
        raise ValueError(f"Unknown claim: {claim}")
    if experiment and experiment not in snapshot.experiments:
        raise ValueError(f"Unknown experiment: {experiment}")
    if run and run not in snapshot.runs:
        raise ValueError(f"Unknown run: {run}")
    if decision and decision not in snapshot.decisions:
        raise ValueError(f"Unknown decision: {decision}")
    _append_unique(record, "linked_claims", claim)
    _append_unique(record, "linked_experiments", experiment)
    _append_unique(record, "linked_runs", run)
    _append_unique(record, "linked_decisions", decision)
    _write_question_file(project, records)
    return record


def show_question(project: Project, question_id: str) -> dict[str, Any]:
    record = next(
        (item for item in list_questions(project) if item["question_id"] == question_id),
        None,
    )
    if record is None:
        raise ValueError(f"Unknown question: {question_id}")
    return record


def _question_path(project: Project) -> Path:
    return project.root / "research/logs/open_questions.md"


def _read_question_file(project: Project) -> list[dict[str, Any]]:
    path = _question_path(project)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = QUESTION_HEADING.match(lines[index])
        if match is None:
            index += 1
            continue
        question_id, heading_text = match.groups()
        fence = index + 1
        while fence < len(lines) and not lines[fence].startswith("```yaml"):
            fence += 1
        if fence >= len(lines):
            raise ValueError(f"{path}:{index + 1}: missing YAML block for {question_id}.")
        end = fence + 1
        while end < len(lines) and not lines[end].startswith("```"):
            end += 1
        if end >= len(lines):
            raise ValueError(f"{path}:{fence + 1}: unclosed YAML block for {question_id}.")
        yaml = YAML(typ="safe")
        payload = yaml.load("\n".join(lines[fence + 1 : end])) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{fence + 1}: question YAML must be a mapping.")
        payload.setdefault("question_id", question_id)
        payload.setdefault("text", heading_text)
        records.append(dict(payload))
        index = end + 1
    records.sort(key=lambda item: str(item.get("question_id")))
    return records


def _write_question_file(project: Project, records: list[dict[str, Any]]) -> None:
    path = _question_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    lines = ["# Open Questions", ""]
    for record in sorted(records, key=lambda item: str(item.get("question_id"))):
        question_id = str(record["question_id"])
        lines.append(f"## {question_id} - {record['text']}")
        lines.append("")
        lines.append("```yaml")
        import io

        stream = io.StringIO()
        yaml.dump(record, stream)
        lines.extend(stream.getvalue().strip().splitlines())
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _next_question_id(records: list[dict[str, Any]]) -> str:
    maximum = 0
    for record in records:
        match = re.fullmatch(r"Q([0-9]+)", str(record.get("question_id") or ""))
        if match:
            maximum = max(maximum, int(match.group(1)))
    return f"Q{maximum + 1:03d}"


def _append_unique(record: dict[str, Any], key: str, value: str | None) -> None:
    if value is None:
        return
    record[key] = sorted({*(record.get(key) or []), value})
