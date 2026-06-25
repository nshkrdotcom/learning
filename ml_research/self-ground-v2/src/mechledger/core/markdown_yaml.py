from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from mechledger.core.diagnostics import raise_diagnostic

YAML_FENCE = re.compile(r"^```\s*yaml\s*$")


def load_yaml_block(
    *,
    path: Path,
    lines: list[str],
    heading_index: int,
    object_id: str,
    code_prefix: str,
) -> tuple[dict[str, Any], int, int]:
    index = heading_index + 1
    blanks = 0
    while index < len(lines) and not lines[index].strip():
        blanks += 1
        index += 1
    if index >= len(lines) or YAML_FENCE.match(lines[index].strip()) is None:
        raise_diagnostic(
            file=str(path),
            line=heading_index + 1,
            object_id=object_id,
            code=f"{code_prefix}.yaml.missing",
            message="Missing required fenced YAML block.",
            suggested_fix="add a ```yaml block immediately after the heading.",
        )
    start = index + 1
    end = start
    while end < len(lines) and lines[end].strip() != "```":
        end += 1
    if end >= len(lines):
        raise_diagnostic(
            file=str(path),
            line=index + 1,
            object_id=object_id,
            code=f"{code_prefix}.yaml.unclosed",
            message="YAML block is unclosed.",
            suggested_fix="add a closing ``` fence.",
        )
    yaml = YAML(typ="rt")
    try:
        data = yaml.load("\n".join(lines[start:end])) or {}
    except YAMLError as exc:
        raise_diagnostic(
            file=str(path),
            line=start + 1,
            object_id=object_id,
            code=f"{code_prefix}.yaml.invalid",
            message=str(exc),
            suggested_fix="fix the fenced YAML syntax.",
        )
    if not isinstance(data, dict):
        raise_diagnostic(
            file=str(path),
            line=start + 1,
            object_id=object_id,
            code=f"{code_prefix}.yaml.type",
            message="YAML block must be a mapping.",
            suggested_fix="replace the YAML block with key/value fields.",
        )
    return dict(data), start + 1, blanks


def require_list_of_strings(
    value: Any,
    *,
    path: Path,
    line: int,
    object_id: str,
    field: str,
    code_prefix: str,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise_diagnostic(
            file=str(path),
            line=line,
            object_id=object_id,
            code=f"{code_prefix}.yaml.{field}",
            message=f"`{field}` must be a list of strings.",
            suggested_fix=f"write `{field}: []` or a YAML list of strings.",
        )
    return value
