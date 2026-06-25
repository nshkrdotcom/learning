from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mechledger.core.diagnostics import raise_diagnostic
from mechledger.core.markdown_yaml import load_yaml_block, require_list_of_strings

DECISION_HEADING = re.compile(r"^##\s+(D[0-9]+[A-Za-z0-9_-]*)\s+(?:-|—)\s+(.+?)\s*$")
DECISION_STATUSES = {"proposed", "accepted", "superseded", "rejected"}


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    decision_id: str
    status: str
    title: str | None = None
    affected_experiments: list[str] = Field(default_factory=list)
    affected_claims: list[str] = Field(default_factory=list)
    decision_type: str | None = None
    copilot_session_id: str | None = None
    file: str
    line: int


class DecisionLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    decisions: dict[str, DecisionRecord]
    warnings: list[str] = Field(default_factory=list)


def parse_decision_log(path: str | Path) -> DecisionLog:
    path = Path(path)
    if not path.exists():
        return DecisionLog(path=str(path), decisions={})
    lines = path.read_text(encoding="utf-8").splitlines()
    decisions: dict[str, DecisionRecord] = {}
    warnings: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        match = DECISION_HEADING.match(line)
        if match is None:
            raise_diagnostic(
                file=str(path),
                line=index + 1,
                code="decision.heading.malformed",
                message="Malformed decision heading.",
                suggested_fix="Use `## D001 - Title`.",
            )
        decision_id, title = match.groups()
        if decision_id in decisions:
            raise_diagnostic(
                file=str(path),
                line=index + 1,
                object_id=decision_id,
                code="decision.id.duplicate",
                message="Decision ID is duplicated.",
                suggested_fix="give each decision a unique D-number.",
            )
        data, yaml_line, blanks = load_yaml_block(
            path=path,
            lines=lines,
            heading_index=index,
            object_id=decision_id,
            code_prefix="decision",
        )
        if blanks > 1:
            warnings.append(f"{path}:{index + 1} {decision_id}: extra blank lines before YAML")
        if data.get("decision_id") != decision_id:
            raise_diagnostic(
                file=str(path),
                line=yaml_line,
                object_id=decision_id,
                code="decision.id.mismatch",
                message="YAML `decision_id` must match heading decision ID.",
                suggested_fix="edit the YAML decision_id or heading so they match.",
            )
        status = data.get("status")
        if not status:
            raise_diagnostic(
                file=str(path),
                line=yaml_line,
                object_id=decision_id,
                code="decision.yaml.required_field",
                message="Missing required field: status",
                suggested_fix="add `status: proposed|accepted|superseded|rejected`.",
            )
        if status not in DECISION_STATUSES:
            raise_diagnostic(
                file=str(path),
                line=yaml_line,
                object_id=decision_id,
                code="decision.status.unknown",
                message=f"Unknown decision status: {status}",
                suggested_fix="use one of proposed, accepted, superseded, rejected.",
            )
        record_data = dict(data)
        for key in ("affected_experiments", "affected_claims", "title"):
            record_data.pop(key, None)
        record = DecisionRecord(
            **record_data,
            title=title,
            affected_experiments=require_list_of_strings(
                data.get("affected_experiments"),
                path=path,
                line=yaml_line,
                object_id=decision_id,
                field="affected_experiments",
                code_prefix="decision",
            ),
            affected_claims=require_list_of_strings(
                data.get("affected_claims"),
                path=path,
                line=yaml_line,
                object_id=decision_id,
                field="affected_claims",
                code_prefix="decision",
            ),
            file=str(path),
            line=index + 1,
        )
        decisions[decision_id] = record
    return DecisionLog(path=str(path), decisions=decisions, warnings=warnings)
