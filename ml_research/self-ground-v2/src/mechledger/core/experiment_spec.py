from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mechledger.core.diagnostics import raise_diagnostic
from mechledger.core.markdown_yaml import load_yaml_block, require_list_of_strings

EXPERIMENT_TITLE = re.compile(r"^#\s+(E[0-9]+[A-Za-z0-9_-]*):\s+(.+?)\s*$")
REQUIRED_HEADINGS = [
    "Status",
    "Research question",
    "Hypothesis",
    "Model / SAE / Hook",
    "Task",
    "Mechanism objects",
    "Claim format",
    "Intervention",
    "Metrics",
    "Baselines",
    "Controls",
    "Success criterion",
    "Failure criterion",
    "Prerequisites",
    "Expected artifacts",
    "Notes",
]


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiment_id: str
    title: str
    status: str
    claim_targets: list[str] = Field(default_factory=list)
    source_runs: list[str] = Field(default_factory=list)
    prerequisites: list[dict[str, Any]] = Field(default_factory=list)
    config_files: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    file: str
    line: int
    warnings: list[str] = Field(default_factory=list)


def parse_experiment_spec(path: str | Path) -> ExperimentSpec:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    title_match = next(
        (EXPERIMENT_TITLE.match(line) for line in lines if line.startswith("# ")), None
    )
    if title_match is None:
        raise_diagnostic(
            file=str(path),
            line=1,
            code="experiment.heading.malformed",
            message="ExperimentSpec must start with `# E001: Title`.",
            suggested_fix="add a top-level experiment heading.",
        )
    experiment_id, title = title_match.groups()
    heading_index = next(i for i, line in enumerate(lines) if line.startswith("# "))
    data, yaml_line, _ = load_yaml_block(
        path=path,
        lines=lines,
        heading_index=heading_index,
        object_id=experiment_id,
        code_prefix="experiment",
    )
    if data.get("experiment_id") != experiment_id:
        raise_diagnostic(
            file=str(path),
            line=yaml_line,
            object_id=experiment_id,
            code="experiment.id.mismatch",
            message="YAML `experiment_id` must match heading experiment ID.",
            suggested_fix="edit the YAML experiment_id or heading so they match.",
        )
    if not data.get("status"):
        raise_diagnostic(
            file=str(path),
            line=yaml_line,
            object_id=experiment_id,
            code="experiment.yaml.required_field",
            message="Missing required field: status",
            suggested_fix="add `status: draft|planned|running|completed|retired`.",
        )
    headings = {line[3:].strip() for line in lines if line.startswith("## ")}
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in headings]
    if missing:
        raise_diagnostic(
            file=str(path),
            line=1,
            object_id=experiment_id,
            code="experiment.heading.required",
            message=f"Missing required headings: {', '.join(missing)}",
            suggested_fix="add all required ExperimentSpec sections from the template.",
        )
    warnings: list[str] = []
    if not data.get("prerequisites") and _section_has_prose(lines, "Prerequisites"):
        warnings.append(
            f"{path}: {experiment_id}: prose prerequisites exist without YAML prerequisites"
        )
    prerequisites = data.get("prerequisites") or []
    if not isinstance(prerequisites, list) or not all(
        isinstance(item, dict) for item in prerequisites
    ):
        raise_diagnostic(
            file=str(path),
            line=yaml_line,
            object_id=experiment_id,
            code="experiment.yaml.prerequisites",
            message="`prerequisites` must be a list of mappings.",
            suggested_fix="write prerequisites as YAML objects with `type` and `id`/`path` fields.",
        )
    record_data = dict(data)
    for key in (
        "claim_targets",
        "source_runs",
        "config_files",
        "expected_artifacts",
        "prerequisites",
        "title",
    ):
        record_data.pop(key, None)
    return ExperimentSpec(
        **record_data,
        title=title,
        claim_targets=require_list_of_strings(
            data.get("claim_targets"),
            path=path,
            line=yaml_line,
            object_id=experiment_id,
            field="claim_targets",
            code_prefix="experiment",
        ),
        source_runs=require_list_of_strings(
            data.get("source_runs"),
            path=path,
            line=yaml_line,
            object_id=experiment_id,
            field="source_runs",
            code_prefix="experiment",
        ),
        config_files=require_list_of_strings(
            data.get("config_files"),
            path=path,
            line=yaml_line,
            object_id=experiment_id,
            field="config_files",
            code_prefix="experiment",
        ),
        expected_artifacts=require_list_of_strings(
            data.get("expected_artifacts"),
            path=path,
            line=yaml_line,
            object_id=experiment_id,
            field="expected_artifacts",
            code_prefix="experiment",
        ),
        prerequisites=prerequisites,
        file=str(path),
        line=heading_index + 1,
        warnings=warnings,
    )


def _section_has_prose(lines: list[str], heading: str) -> bool:
    marker = f"## {heading}"
    try:
        start = lines.index(marker) + 1
    except ValueError:
        return False
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return any(line.strip() for line in lines[start:end])
