from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from mechledger.hashing import canonical_claim_hash
from mechledger.models import (
    CLAIM_STATUSES,
    DECISION_STATUSES,
    ClaimLedger,
    ClaimRecord,
    DecisionLog,
    DecisionRecord,
    ExperimentSpec,
)


class LedgerParseError(ValueError):
    pass


CLAIM_HEADING_RE = re.compile(r"^###\s+(C[0-9]+[A-Za-z0-9_-]*)\s+(?:-|—)\s+(.+?)\s*$")
DECISION_HEADING_RE = re.compile(r"^##\s+(D[0-9]+[A-Za-z0-9_-]*)\s+(?:-|—)\s+(.+?)\s*$")
EXPERIMENT_HEADING_RE = re.compile(r"^#\s+(E[0-9]+[A-Za-z0-9_-]*):\s+(.+?)\s*$")
YAML_OPEN_RE = re.compile(r"^```\s*yaml\s*$|^```\s*yml\s*$")


def _error(path: Path, line: int | None, object_id: str | None, rule: str, message: str) -> None:
    location = str(path)
    if line is not None:
        location += f":{line}"
    if object_id:
        location += f" {object_id}"
    raise LedgerParseError(f"ERROR {location}\nRule: {rule}\n{message}")


def _load_yaml_block(
    path: Path, lines: list[str], heading_index: int, object_id: str
) -> tuple[dict[str, Any], int, int]:
    index = heading_index + 1
    blank_count = 0
    while index < len(lines) and not lines[index].strip():
        blank_count += 1
        index += 1
    if index >= len(lines) or not YAML_OPEN_RE.match(lines[index].strip()):
        _error(
            path,
            heading_index + 1,
            object_id,
            "yaml.block.missing",
            "Missing required fenced YAML block.",
        )
    start = index + 1
    end = start
    while end < len(lines) and not lines[end].strip().startswith("```"):
        end += 1
    if end >= len(lines):
        _error(path, index + 1, object_id, "yaml.block.unclosed", "YAML fence is not closed.")
    raw = "\n".join(lines[start:end])
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        _error(path, start + 1, object_id, "yaml.invalid", f"Invalid YAML: {exc}")
    if not isinstance(data, dict):
        _error(path, start + 1, object_id, "yaml.object", "YAML block must be a mapping.")
    return data, start + 1, blank_count


def _ensure_list(
    path: Path,
    line: int,
    object_id: str,
    data: dict[str, Any],
    field: str,
    *,
    required: bool,
) -> list[str]:
    if field not in data:
        if required:
            _error(
                path,
                line,
                object_id,
                "claim.yaml.required_field",
                f"Missing required field: {field}\nSuggested fix: add `{field}: []` or a list.",
            )
        return []
    value = data[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _error(path, line, object_id, f"{field}.type", f"`{field}` must be a list of strings.")
    return list(value)


def _ensure_optional_string_or_null(
    path: Path, line: int, object_id: str, data: dict[str, Any], field: str
) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        _error(path, line, object_id, f"{field}.type", f"`{field}` must be a string or null.")
    return value


def parse_claim_ledger(path: str | Path) -> ClaimLedger:
    path = Path(path)
    if not path.exists():
        _error(path, None, None, "claim_ledger.missing", "Missing claim ledger.")
    lines = path.read_text(encoding="utf-8").splitlines()
    claims: dict[str, ClaimRecord] = {}
    warnings: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith("### "):
            continue
        match = CLAIM_HEADING_RE.match(line)
        if not match:
            _error(path, index + 1, None, "claim.heading.malformed", "Expected `### C001 - Title`.")
        claim_id, title = match.groups()
        if claim_id in claims:
            _error(path, index + 1, claim_id, "claim.id.duplicate", "Duplicate claim ID.")
        data, yaml_line, blank_count = _load_yaml_block(path, lines, index, claim_id)
        if blank_count > 1:
            warnings.append(f"{path}:{index + 1} {claim_id}: more than one blank line before YAML.")
        if data.get("claim_id") != claim_id:
            _error(
                path,
                yaml_line,
                claim_id,
                "claim.id.mismatch",
                "YAML claim_id must equal heading claim ID.",
            )
        status = data.get("status")
        if not isinstance(status, str):
            _error(
                path,
                yaml_line,
                claim_id,
                "claim.yaml.required_field",
                "Missing required field: status",
            )
        if status not in CLAIM_STATUSES:
            _error(
                path, yaml_line, claim_id, "claim.status.unknown", f"Unknown claim status: {status}"
            )
        allowed = _ensure_list(path, index + 1, claim_id, data, "allowed", required=True)
        forbidden = _ensure_list(path, index + 1, claim_id, data, "forbidden", required=True)
        required_caveats = _ensure_list(
            path, yaml_line, claim_id, data, "required_caveats", required=False
        )
        debt_flags = _ensure_list(path, yaml_line, claim_id, data, "debt_flags", required=False)
        linked_experiments = _ensure_list(
            path, yaml_line, claim_id, data, "linked_experiments", required=False
        )
        linked_runs = _ensure_list(path, yaml_line, claim_id, data, "linked_runs", required=False)
        linked_decisions = _ensure_list(
            path, yaml_line, claim_id, data, "linked_decisions", required=False
        )
        copilot_session_id = _ensure_optional_string_or_null(
            path, yaml_line, claim_id, data, "copilot_session_id"
        )
        claims[claim_id] = ClaimRecord(
            claim_id=claim_id,
            title=str(data.get("title") or title),
            status=status,
            allowed=allowed,
            forbidden=forbidden,
            required_caveats=required_caveats,
            debt_flags=debt_flags,
            linked_experiments=linked_experiments,
            linked_runs=linked_runs,
            linked_decisions=linked_decisions,
            raw_yaml=data,
            block_hash=canonical_claim_hash(data),
            file=path,
            line=index + 1,
            scope=data.get("scope"),
            owner=data.get("owner"),
            updated_at=data.get("updated_at"),
            tags=_ensure_list(path, yaml_line, claim_id, data, "tags", required=False),
            copilot_session_id=copilot_session_id,
        )
    return ClaimLedger(path=path, claims=claims, warnings=warnings)


def parse_decision_log(path: str | Path) -> DecisionLog:
    path = Path(path)
    if not path.exists():
        return DecisionLog(path=path, decisions={})
    lines = path.read_text(encoding="utf-8").splitlines()
    decisions: dict[str, DecisionRecord] = {}
    warnings: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        match = DECISION_HEADING_RE.match(line)
        if not match:
            continue
        decision_id, title = match.groups()
        if decision_id in decisions:
            _error(path, index + 1, decision_id, "decision.id.duplicate", "Duplicate decision ID.")
        data, yaml_line, blank_count = _load_yaml_block(path, lines, index, decision_id)
        if blank_count > 1:
            warnings.append(
                f"{path}:{index + 1} {decision_id}: more than one blank line before YAML."
            )
        if data.get("decision_id") != decision_id:
            _error(
                path,
                yaml_line,
                decision_id,
                "decision.id.mismatch",
                "YAML decision_id must equal heading decision ID.",
            )
        status = data.get("status")
        if not isinstance(status, str):
            _error(path, yaml_line, decision_id, "decision.yaml.required_field", "Missing status.")
        if status not in DECISION_STATUSES:
            _error(
                path, yaml_line, decision_id, "decision.status.unknown", f"Unknown status: {status}"
            )
        affected_experiments = _ensure_list(
            path, yaml_line, decision_id, data, "affected_experiments", required=False
        )
        affected_claims = _ensure_list(
            path, yaml_line, decision_id, data, "affected_claims", required=False
        )
        decisions[decision_id] = DecisionRecord(
            decision_id=decision_id,
            title=title,
            status=status,
            raw_yaml=data,
            file=path,
            line=index + 1,
            affected_experiments=affected_experiments,
            affected_claims=affected_claims,
            decision_type=data.get("decision_type"),
            copilot_session_id=_ensure_optional_string_or_null(
                path, yaml_line, decision_id, data, "copilot_session_id"
            ),
        )
    return DecisionLog(path=path, decisions=decisions, warnings=warnings)


def _first_yaml_block(lines: list[str]) -> tuple[dict[str, Any] | None, int | None]:
    for index, line in enumerate(lines):
        if YAML_OPEN_RE.match(line.strip()):
            start = index + 1
            end = start
            while end < len(lines) and not lines[end].strip().startswith("```"):
                end += 1
            if end >= len(lines):
                raise LedgerParseError(
                    f"ERROR experiment spec:{index + 1}\nRule: yaml.block.unclosed"
                )
            raw = "\n".join(lines[start:end])
            data = yaml.safe_load(raw) or {}
            if not isinstance(data, dict):
                raise LedgerParseError("ExperimentSpec YAML block must be a mapping.")
            return data, start + 1
    return None, None


def parse_experiment_spec(path: str | Path) -> ExperimentSpec:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    heading_id: str | None = None
    title: str | None = None
    heading_line = 1
    for index, line in enumerate(lines):
        match = EXPERIMENT_HEADING_RE.match(line)
        if match:
            heading_id, title = match.groups()
            heading_line = index + 1
            break
    if heading_id is None or title is None:
        _error(path, 1, None, "experiment.heading.malformed", "Expected `# E001: Title`.")
    data, yaml_line = _first_yaml_block(lines)
    warnings: list[str] = []
    if data is None:
        prerequisites_text = _section_text(lines, "Prerequisites")
        if prerequisites_text.strip():
            warnings.append(
                f"Experiment {heading_id} has prose prerequisites but no "
                "machine-readable YAML prerequisites."
            )
        experiment_id = heading_id
        status = _section_text(lines, "Status").strip() or "draft"
        claim_targets: list[str] = []
        source_runs: list[str] = []
        prerequisites: list[dict[str, Any]] = []
        config_files: list[str] = []
        expected_artifacts: list[str] = []
    else:
        experiment_id = data.get("experiment_id")
        if experiment_id != heading_id:
            _error(
                path,
                yaml_line,
                heading_id,
                "experiment.id.mismatch",
                "YAML experiment_id must equal heading experiment ID.",
            )
        status = str(data.get("status", "draft"))
        claim_targets = _string_list(
            data.get("claim_targets", []), "claim_targets", path, yaml_line
        )
        source_runs = _string_list(data.get("source_runs", []), "source_runs", path, yaml_line)
        config_files = _string_list(data.get("config_files", []), "config_files", path, yaml_line)
        expected_artifacts = _string_list(
            data.get("expected_artifacts", []), "expected_artifacts", path, yaml_line
        )
        prerequisites_raw = data.get("prerequisites", [])
        if not isinstance(prerequisites_raw, list) or not all(
            isinstance(item, dict) for item in prerequisites_raw
        ):
            _error(
                path,
                yaml_line,
                heading_id,
                "experiment.prerequisites.type",
                "Must be list of maps.",
            )
        prerequisites = list(prerequisites_raw)

    required_headings = [
        "Status",
        "Research question",
        "Hypothesis",
        "Metrics",
        "Controls",
        "Success criterion",
        "Failure criterion",
    ]
    present = set(_headings(lines))
    for required in required_headings:
        if required not in present:
            _error(
                path,
                heading_line,
                heading_id,
                "experiment.heading.required",
                f"Missing required heading: ## {required}",
            )

    return ExperimentSpec(
        experiment_id=experiment_id,
        title=title,
        status=status,
        file=path,
        line=heading_line,
        raw_yaml=data,
        claim_targets=claim_targets,
        source_runs=source_runs,
        prerequisites=prerequisites,
        config_files=config_files,
        expected_artifacts=expected_artifacts,
        machine_warnings=warnings,
    )


def _string_list(value: Any, field: str, path: Path, line: int | None) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _error(path, line, None, f"experiment.{field}.type", f"{field} must be list[str].")
    return list(value)


def _headings(lines: list[str]) -> list[str]:
    headings: list[str] = []
    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            headings.append(line[3:].strip())
    return headings


def _section_text(lines: list[str], heading: str) -> str:
    for index, line in enumerate(lines):
        if line.strip() == f"## {heading}":
            chunk: list[str] = []
            for next_line in lines[index + 1 :]:
                if next_line.startswith("## "):
                    break
                chunk.append(next_line)
            return "\n".join(chunk)
    return ""
