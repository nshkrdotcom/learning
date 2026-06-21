from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from self_ground.behavioral_tasks import TASK_FAMILY_ORDER
from self_ground.io import write_config, write_jsonl

CalibrationMode = Literal["none", "baseline-intended-direction", "baseline-margin"]


class TaskCalibrationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_baseline_intended_direction: bool = True
    min_abs_baseline_margin: float | None = None
    min_tasks_per_family: int = 3
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


def task_calibration_rule_from_mode(
    *,
    mode: CalibrationMode,
    min_abs_baseline_margin: float | None = None,
    min_tasks_per_family: int = 3,
    allow_family_drop: bool = False,
) -> TaskCalibrationRule | None:
    if mode == "none":
        return None
    if min_tasks_per_family < 1:
        raise ValueError("min_tasks_per_family must be >= 1")
    if mode == "baseline-intended-direction":
        return TaskCalibrationRule(
            require_baseline_intended_direction=True,
            min_abs_baseline_margin=None,
            min_tasks_per_family=min_tasks_per_family,
            allow_family_drop=allow_family_drop,
            reason=(
                "Preregistered baseline-only filter: keep only tasks where the "
                "unpatched model already favors the intended continuation."
            ),
        )
    if mode == "baseline-margin":
        if min_abs_baseline_margin is None:
            raise ValueError("baseline-margin calibration requires min_abs_baseline_margin")
        if min_abs_baseline_margin < 0:
            raise ValueError("min_abs_baseline_margin must be non-negative")
        return TaskCalibrationRule(
            require_baseline_intended_direction=True,
            min_abs_baseline_margin=min_abs_baseline_margin,
            min_tasks_per_family=min_tasks_per_family,
            allow_family_drop=allow_family_drop,
            reason=(
                "Preregistered baseline-only filter: keep only intended-direction "
                "tasks whose absolute baseline prompt margin exceeds the configured "
                "minimum."
            ),
        )
    raise ValueError(f"unknown task calibration mode: {mode}")


def _task_dict(task: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(task, dict):
        return task
    if hasattr(task, "model_dump"):
        return task.model_dump(mode="json")
    raise TypeError(f"task must be dict-like or Pydantic model, got {type(task).__name__}")


def _family_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(task.get("family")) for task in tasks if task.get("family"))
    families = set(TASK_FAMILY_ORDER) | set(counts)
    return {family: int(counts.get(family, 0)) for family in sorted(families)}


def _margin(row: dict[str, Any]) -> float | None:
    value = row.get("baseline_prompt_contrast")
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _passes_minimum(
    *,
    after_counts: dict[str, int],
    before_counts: dict[str, int],
    rule: TaskCalibrationRule,
) -> tuple[bool, list[str]]:
    required = list(TASK_FAMILY_ORDER)
    missing: list[str] = []
    for family in required:
        after = int(after_counts.get(family, 0))
        before = int(before_counts.get(family, 0))
        if rule.allow_family_drop and before == 0:
            continue
        if rule.allow_family_drop and after == 0:
            continue
        if after < rule.min_tasks_per_family:
            missing.append(family)
    return not missing, missing


def apply_task_calibration(
    *,
    tasks: list[dict[str, Any] | Any],
    baseline_rows: list[dict[str, Any]],
    rule: TaskCalibrationRule,
) -> TaskCalibrationResult:
    """Apply preregistered task calibration using baseline-only fields.

    This function deliberately does not accept or inspect intervention rows. It can
    only filter on task metadata and baseline calibration scores written before any
    decoded SAE intervention is run.
    """

    task_dicts = [_task_dict(task) for task in tasks]
    baseline_by_id = {str(row.get("task_id")): row for row in baseline_rows if row.get("task_id")}
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for task in task_dicts:
        task_id = task.get("id")
        family = task.get("family")
        if not task_id or not family:
            excluded.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "reason": "missing_task_metadata",
                    "details": "task id and family are required for calibration",
                }
            )
            continue
        if task.get("valid") is False or task.get("token_validation_valid") is False:
            excluded.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "reason": "token_validation_failed",
                    "details": "task was marked invalid before calibration",
                }
            )
            continue
        baseline = baseline_by_id.get(str(task_id))
        if baseline is None:
            excluded.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "reason": "missing_baseline_row",
                    "details": "no baseline row was found for this task id",
                }
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

    before_counts = _family_counts(task_dicts)
    after_counts = _family_counts(kept)
    passes, missing = _passes_minimum(
        after_counts=after_counts,
        before_counts=before_counts,
        rule=rule,
    )
    exclusion_counts = Counter(str(row["reason"]) for row in excluded)
    return TaskCalibrationResult(
        total_tasks_before=len(task_dicts),
        total_tasks_after=len(kept),
        kept_task_ids=[str(task["id"]) for task in kept],
        excluded=excluded,
        valid_by_family_before=before_counts,
        valid_by_family_after=after_counts,
        passes_minimum=passes,
        missing_required_families=missing,
        exclusions_by_reason={key: int(exclusion_counts[key]) for key in sorted(exclusion_counts)},
        rule=rule,
    )


def write_task_calibration_artifacts(
    *,
    out_dir: str | Path,
    tasks: list[dict[str, Any] | Any],
    result: TaskCalibrationResult,
) -> None:
    path = Path(out_dir)
    task_by_id = {str(_task_dict(task).get("id")): _task_dict(task) for task in tasks}
    kept = [task_by_id[task_id] for task_id in result.kept_task_ids if task_id in task_by_id]
    write_config(result.rule.model_dump(mode="json"), path / "task_calibration_rule.json")
    write_config(result.model_dump(mode="json"), path / "task_calibration_result.json")
    write_jsonl(kept, path / "calibrated_behavioral_tasks.jsonl")
    write_jsonl(result.excluded, path / "calibrated_excluded_behavioral_tasks.jsonl")
