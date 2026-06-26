from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from mechledger.assessments.common import AssessmentConditionResult
from mechledger.core.debt import DebtSeverity

CalibrationMode = Literal["none", "baseline-intended-direction", "baseline-margin"]


class TaskCalibrationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_baseline_intended_direction: bool = True
    min_abs_baseline_margin: float | None = None
    min_tasks_per_family: int = 3
    required_families: list[str]
    allow_family_drop: bool = False
    reason: str


class TaskCalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_tasks_before: int
    total_tasks_after: int
    kept_task_ids: list[str]
    excluded: list[dict[str, Any]]
    valid_by_family_before: dict[str, int]
    valid_by_family_after: dict[str, int]
    passes_minimum: bool
    missing_required_families: list[str]
    exclusions_by_reason: dict[str, int]
    rule: TaskCalibrationRule


def apply_task_calibration(
    *,
    tasks: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    rule: TaskCalibrationRule,
) -> TaskCalibrationResult:
    baseline_by_id = {str(row.get("task_id")): row for row in baseline_rows if row.get("task_id")}
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task.get("id")
        family = task.get("family")
        if not task_id or not family:
            excluded.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "reason": "missing_task_metadata",
                }
            )
            continue
        if task.get("valid") is False or task.get("token_validation_valid") is False:
            excluded.append(
                {"task_id": task_id, "family": family, "reason": "token_validation_failed"}
            )
            continue
        baseline = baseline_by_id.get(str(task_id))
        if baseline is None:
            excluded.append(
                {"task_id": task_id, "family": family, "reason": "missing_baseline_row"}
            )
            continue
        if rule.require_baseline_intended_direction and not bool(
            baseline.get("intended_direction_pass")
        ):
            excluded.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "reason": "baseline_wrong_direction",
                    "baseline_prompt_contrast": baseline.get("baseline_prompt_contrast"),
                }
            )
            continue
        margin = _margin(baseline)
        if rule.min_abs_baseline_margin is not None and (
            margin is None or abs(margin) < rule.min_abs_baseline_margin
        ):
            excluded.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "reason": "baseline_margin_below_threshold",
                    "baseline_prompt_contrast": baseline.get("baseline_prompt_contrast"),
                    "threshold": rule.min_abs_baseline_margin,
                }
            )
            continue
        kept.append(task)
    before_counts = _family_counts(tasks, rule.required_families)
    after_counts = _family_counts(kept, rule.required_families)
    missing = _missing_required_families(before_counts, after_counts, rule)
    exclusion_counts = Counter(str(row["reason"]) for row in excluded)
    return TaskCalibrationResult(
        total_tasks_before=len(tasks),
        total_tasks_after=len(kept),
        kept_task_ids=[str(task["id"]) for task in kept],
        excluded=excluded,
        valid_by_family_before=before_counts,
        valid_by_family_after=after_counts,
        passes_minimum=not missing,
        missing_required_families=missing,
        exclusions_by_reason={key: int(exclusion_counts[key]) for key in sorted(exclusion_counts)},
        rule=rule,
    )


def evaluate_baseline_calibration(metrics: dict[str, Any]) -> AssessmentConditionResult:
    rate = metrics.get("intended_direction_pass_rate")
    recorded = rate is not None or metrics.get("baseline_contrast") is not None
    passed = recorded and (
        bool(metrics.get("intended_direction_pass"))
        if metrics.get("intended_direction_pass") is not None
        else float(rate if rate is not None else 0.0) >= 0.5
    )
    return AssessmentConditionResult(
        condition_id="baseline_calibration_recorded",
        condition_type="baseline_calibration_recorded",
        passed=bool(passed),
        parameters={
            "intended_direction_pass_rate": rate,
            "min_pass_rate": 0.5,
            "threshold": 0.5,
        },
        failure_message="Baseline calibration is missing or below the default pass-rate floor.",
        default_consequence="scientific_debt",
        debt_type="missing_baseline_calibration",
        severity=DebtSeverity.SERIOUS,
    )


def evaluate_positive_control(metrics: dict[str, Any]) -> AssessmentConditionResult:
    rate = metrics.get("positive_control_pass_rate")
    passed = rate is not None and float(rate) >= 0.9
    missing = rate is None
    return AssessmentConditionResult(
        condition_id="positive_control_pass_rate",
        condition_type="positive_control_passed",
        passed=bool(passed),
        parameters={
            "positive_control_pass_rate": rate,
            "min_pass_rate": 0.9,
            "threshold": 0.9,
        },
        failure_message="Positive-control pass rate is missing or below 0.9.",
        default_consequence="blocker",
        debt_type="missing_positive_control" if missing else "failed_positive_control",
        severity=DebtSeverity.BLOCKING,
    )


def _margin(row: dict[str, Any]) -> float | None:
    value = row.get("baseline_prompt_contrast")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _family_counts(tasks: list[dict[str, Any]], required_families: list[str]) -> dict[str, int]:
    counts = Counter(str(task.get("family")) for task in tasks if task.get("family"))
    families = set(required_families) | set(counts)
    return {family: int(counts.get(family, 0)) for family in sorted(families)}


def _missing_required_families(
    before_counts: dict[str, int],
    after_counts: dict[str, int],
    rule: TaskCalibrationRule,
) -> list[str]:
    missing: list[str] = []
    for family in rule.required_families:
        before = int(before_counts.get(family, 0))
        after = int(after_counts.get(family, 0))
        if rule.allow_family_drop and before == 0:
            continue
        if rule.allow_family_drop and after == 0:
            continue
        if after < rule.min_tasks_per_family:
            missing.append(family)
    return missing
