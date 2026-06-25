from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mechledger.core.diagnostics import raise_diagnostic
from mechledger.core.markdown_yaml import load_yaml_block, require_list_of_strings

ENTRY_HEADING = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s*$")
REQUIRED_SECTIONS = [
    "Question",
    "Context",
    "Hypothesis",
    "Work done",
    "Result",
    "Interpretation",
    "Decision",
    "Open questions",
]


class ResearchLogEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    entry_id: str
    date: str
    linked_runs: list[str] = Field(default_factory=list)
    linked_claims: list[str] = Field(default_factory=list)
    linked_decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    copilot_session_id: str | None = None
    file: str
    line: int


class ResearchLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    entries: list[ResearchLogEntry]
    warnings: list[str] = Field(default_factory=list)


def parse_research_log(path: str | Path) -> ResearchLog:
    path = Path(path)
    if not path.exists():
        return ResearchLog(path=str(path), entries=[])
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[ResearchLogEntry] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        match = ENTRY_HEADING.match(line)
        if match is None:
            continue
        date = match.group(1)
        data, yaml_line, _ = load_yaml_block(
            path=path,
            lines=lines,
            heading_index=index,
            object_id=date,
            code_prefix="research_log",
        )
        entry_id = data.get("entry_id")
        if not entry_id:
            raise_diagnostic(
                file=str(path),
                line=yaml_line,
                object_id=date,
                code="research_log.yaml.required_field",
                message="Missing required field: entry_id",
                suggested_fix="add `entry_id: RYYYY-MM-DD-001`.",
            )
        if entry_id in seen:
            raise_diagnostic(
                file=str(path),
                line=index + 1,
                object_id=str(entry_id),
                code="research_log.id.duplicate",
                message="Research log entry ID is duplicated.",
                suggested_fix="give each entry a unique R-date-number ID.",
            )
        seen.add(str(entry_id))
        required = set(REQUIRED_SECTIONS)
        following = lines[index + 1 : _next_entry_index(lines, index + 1)]
        sections = {item[4:].strip() for item in following if item.startswith("### ")}
        missing = required - sections
        if missing:
            raise_diagnostic(
                file=str(path),
                line=index + 1,
                object_id=str(entry_id),
                code="research_log.section.required",
                message=f"Missing required sections: {', '.join(sorted(missing))}",
                suggested_fix="add the missing research-log prose sections.",
            )
        record_data = dict(data)
        for key in (
            "linked_runs",
            "linked_claims",
            "linked_decisions",
            "open_questions",
        ):
            record_data.pop(key, None)
        entries.append(
            ResearchLogEntry(
                **record_data,
                date=date,
                linked_runs=require_list_of_strings(
                    data.get("linked_runs"),
                    path=path,
                    line=yaml_line,
                    object_id=str(entry_id),
                    field="linked_runs",
                    code_prefix="research_log",
                ),
                linked_claims=require_list_of_strings(
                    data.get("linked_claims"),
                    path=path,
                    line=yaml_line,
                    object_id=str(entry_id),
                    field="linked_claims",
                    code_prefix="research_log",
                ),
                linked_decisions=require_list_of_strings(
                    data.get("linked_decisions"),
                    path=path,
                    line=yaml_line,
                    object_id=str(entry_id),
                    field="linked_decisions",
                    code_prefix="research_log",
                ),
                open_questions=require_list_of_strings(
                    data.get("open_questions"),
                    path=path,
                    line=yaml_line,
                    object_id=str(entry_id),
                    field="open_questions",
                    code_prefix="research_log",
                ),
                file=str(path),
                line=index + 1,
            )
        )
    return ResearchLog(path=str(path), entries=entries)


def _next_entry_index(lines: list[str], start: int) -> int:
    return next((i for i in range(start, len(lines)) if ENTRY_HEADING.match(lines[i])), len(lines))
