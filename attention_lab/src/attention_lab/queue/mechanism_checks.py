from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ACTIVITY_THRESHOLD = 1e-6
QKV_FAMILY_PREFIXES = ("multi_qkv_", "qkv_shift_")


@dataclass(frozen=True)
class MechanismCheckResult:
    active: bool | None
    note: str


def mechanism_check_name(attention_type: str, queue_config: dict[str, Any] | None = None) -> str | None:
    queue_config = queue_config or {}
    explicit = queue_config.get("mechanism_check")
    if explicit:
        return str(explicit)
    if attention_type == "standard":
        return None
    if attention_type in {"cp_bilinear", "cp_trilinear"}:
        return "cp_gradient_norm"
    if attention_type.startswith(QKV_FAMILY_PREFIXES):
        return "qkv_track_activity"
    return None


def load_diagnostic_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_mechanism_activity(
    *,
    attention_type: str,
    diagnostics_path: str | Path,
    queue_config: dict[str, Any] | None = None,
    threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
) -> MechanismCheckResult:
    queue_config = queue_config or {}
    check_name = mechanism_check_name(attention_type, queue_config)
    if check_name is None:
        return MechanismCheckResult(None, "standard attention has no mechanism activity requirement")

    rows = load_diagnostic_rows(diagnostics_path)
    if not rows:
        if queue_config.get("allow_missing_diagnostics", False):
            return MechanismCheckResult(None, "missing diagnostics allowed by queue.allow_missing_diagnostics")
        return MechanismCheckResult(None, f"missing diagnostics for mechanism_check={check_name}")

    if check_name == "cp_gradient_norm":
        return _check_cp_gradient_norm(rows, threshold)
    if check_name == "qkv_track_activity":
        return _check_qkv_track_activity(rows, threshold)
    return MechanismCheckResult(None, f"unknown mechanism_check={check_name}")


def _check_cp_gradient_norm(rows: list[dict[str, Any]], threshold: float) -> MechanismCheckResult:
    saw_value = False
    for row in rows:
        value = row.get("cp_gradient_norm")
        if value is None:
            continue
        saw_value = True
        if float(value) > threshold:
            return MechanismCheckResult(True, "cp_gradient_norm above threshold")
    if saw_value:
        return MechanismCheckResult(False, "cp_gradient_norm never exceeded threshold")
    return MechanismCheckResult(None, "diagnostics did not contain cp_gradient_norm")


def _check_qkv_track_activity(rows: list[dict[str, Any]], threshold: float) -> MechanismCheckResult:
    saw_value = False
    for row in rows:
        values = [
            _as_float(row.get("track_gradient_norm")),
            _as_float(row.get("branch_off_logit_delta")),
            _as_float(row.get("track_output_delta")),
            _max_nested_float(row.get("per_track_gradient_norm")),
        ]
        for value in values:
            if value is None:
                continue
            saw_value = True
            if value > threshold:
                return MechanismCheckResult(True, "qkv track activity above threshold")
    if saw_value:
        return MechanismCheckResult(False, "qkv track activity never exceeded threshold")
    return MechanismCheckResult(None, "diagnostics did not contain qkv track activity fields")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _max_nested_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        numbers = [float(item) for item in value.values()]
        return max(numbers) if numbers else None
    if isinstance(value, (list, tuple)):
        numbers = [float(item) for item in value]
        return max(numbers) if numbers else None
    return float(value)
