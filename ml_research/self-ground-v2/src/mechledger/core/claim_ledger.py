from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from mechledger.core.claim_status import ClaimStatus

CLAIM_HEADING = re.compile(r"^###\s+(C[0-9]+[A-Za-z0-9_-]*)\s+(?:-|—)\s+(.+?)\s*$")
YAML_FENCE = re.compile(r"^```\s*yaml\s*$")
ORDER_INSENSITIVE_FIELDS = {
    "allowed",
    "forbidden",
    "required_caveats",
    "debt_flags",
    "linked_experiments",
    "linked_runs",
    "linked_decisions",
    "tags",
}


class ClaimLedgerParseError(ValueError):
    pass


class ClaimRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim_id: str
    title: str | None = None
    status: ClaimStatus
    allowed: list[str]
    forbidden: list[str]
    scope: str | None = None
    required_caveats: list[str] = Field(default_factory=list)
    debt_flags: list[str] = Field(default_factory=list)
    linked_experiments: list[str] = Field(default_factory=list)
    linked_runs: list[str] = Field(default_factory=list)
    linked_decisions: list[str] = Field(default_factory=list)
    owner: str | None = None
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    copilot_session_id: str | None = None
    file: str
    line: int
    heading_title: str
    block_hash: str

    @field_validator(
        "allowed",
        "forbidden",
        "required_caveats",
        "debt_flags",
        "linked_experiments",
        "linked_runs",
        "linked_decisions",
        "tags",
    )
    @classmethod
    def strings_only(cls, value: list[str]) -> list[str]:
        if not all(isinstance(item, str) for item in value):
            raise ValueError("must be a list of strings")
        return value


class ClaimLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    claims: dict[str, ClaimRecord]
    warnings: list[str] = Field(default_factory=list)


def parse_claim_ledger(path: str | Path) -> ClaimLedger:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    claims: dict[str, ClaimRecord] = {}
    warnings: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith("### "):
            continue
        match = CLAIM_HEADING.match(line)
        if match is None:
            _raise(
                path,
                index + 1,
                None,
                "claim.heading.malformed",
                "Malformed claim heading.",
                "use `### C001 - Title`.",
            )
        claim_id, title = match.groups()
        if claim_id in claims:
            _raise(
                path,
                index + 1,
                claim_id,
                "claim.id.duplicate",
                "Claim ID is duplicated.",
                "give each claim a unique C-number.",
            )
        data, yaml_line, blank_count = _extract_yaml(path, lines, index, claim_id)
        if blank_count > 1:
            warnings.append(f"{path}:{index + 1} {claim_id}: extra blank lines before YAML block")
        if data.get("claim_id") != claim_id:
            _raise(
                path,
                yaml_line,
                claim_id,
                "claim.id.mismatch",
                "YAML `claim_id` must equal heading claim ID.",
                "edit the YAML claim_id or heading so they match.",
            )
        for field in ("claim_id", "status", "allowed", "forbidden"):
            if field not in data:
                _raise(
                    path,
                    index + 1,
                    claim_id,
                    "claim.yaml.required_field",
                    f"Missing required field: {field}",
                    f"add `{field}: []` or a valid `{field}` value.",
                )
        try:
            record = ClaimRecord.model_validate(
                {
                    **data,
                    "file": str(path),
                    "line": index + 1,
                    "heading_title": title,
                    "block_hash": canonical_claim_hash(data),
                }
            )
        except ValidationError as exc:
            _raise(
                path,
                yaml_line,
                claim_id,
                "claim.yaml.validation",
                str(exc),
                "make the YAML block match PRD Section 10.3-10.5.",
            )
        claims[claim_id] = record
    return ClaimLedger(path=str(path), claims=claims, warnings=warnings)


def canonical_claim_hash(data: dict[str, Any]) -> str:
    canonical = _strip_empty(_canonicalize(data))
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonicalize(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {item_key: _canonicalize(value[item_key], item_key) for item_key in sorted(value)}
    if isinstance(value, list):
        normalized = [_canonicalize(item) for item in value]
        if key in ORDER_INSENSITIVE_FIELDS and all(_is_scalar(item) for item in normalized):
            return sorted(
                {_scalar_key(item): item for item in normalized}.values(), key=_scalar_key
            )
        return normalized
    return value


def _strip_empty(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            stripped = _strip_empty(item)
            if stripped not in (None, "", [], {}):
                result[key] = stripped
        return result
    if isinstance(value, list):
        return [
            stripped for item in value if (stripped := _strip_empty(item)) not in (None, "", [], {})
        ]
    return value


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _scalar_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _extract_yaml(
    path: Path,
    lines: list[str],
    heading_index: int,
    object_id: str,
) -> tuple[dict[str, Any], int, int]:
    index = heading_index + 1
    blanks = 0
    while index < len(lines) and not lines[index].strip():
        blanks += 1
        index += 1
    if index >= len(lines) or YAML_FENCE.match(lines[index].strip()) is None:
        _raise(
            path,
            heading_index + 1,
            object_id,
            "claim.yaml.missing",
            "Missing required fenced YAML block.",
            "add a ```yaml block immediately after the claim heading.",
        )
    start = index + 1
    end = start
    while end < len(lines) and lines[end].strip() != "```":
        end += 1
    if end >= len(lines):
        _raise(path, index + 1, object_id, "claim.yaml.unclosed", "YAML block is unclosed.")
    yaml = YAML(typ="rt")
    try:
        data = yaml.load("\n".join(lines[start:end])) or {}
    except YAMLError as exc:
        _raise(
            path,
            start + 1,
            object_id,
            "claim.yaml.invalid",
            str(exc),
            "fix the fenced YAML syntax.",
        )
    if not isinstance(data, dict):
        _raise(path, start + 1, object_id, "claim.yaml.type", "YAML block must be a mapping.")
    return dict(data), start + 1, blanks


def _raise(
    path: Path,
    line: int | None,
    object_id: str | None,
    rule: str,
    message: str,
    fix: str | None = None,
) -> None:
    location = str(path)
    if line is not None:
        location += f":{line}"
    if object_id:
        location += f" {object_id}"
    text = f"ERROR {location}\nRule: {rule}\n{message}"
    if fix:
        text += f"\nSuggested fix: {fix}"
    raise ClaimLedgerParseError(text)
