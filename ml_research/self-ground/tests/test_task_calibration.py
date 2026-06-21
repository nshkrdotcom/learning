from __future__ import annotations

from self_ground.task_calibration import (
    TaskCalibrationRule,
    apply_task_calibration,
    task_calibration_rule_from_mode,
)


def _tasks() -> list[dict]:
    rows = []
    for family in ["sentiment_negation", "property_negation", "state_negation"]:
        for idx in range(3):
            rows.append({"id": f"{family}_{idx}", "family": family, "metadata": {}})
    return rows


def _baseline_rows(
    *,
    wrong_task: str | None = None,
    low_margin_task: str | None = None,
) -> list[dict]:
    rows = []
    for task in _tasks():
        task_id = task["id"]
        rows.append(
            {
                "task_id": task_id,
                "family": task["family"],
                "baseline_prompt_contrast": 0.05 if task_id == low_margin_task else 0.5,
                "intended_direction_pass": task_id != wrong_task,
            }
        )
    return rows


def test_calibration_excludes_wrong_direction_baseline_tasks() -> None:
    rule = TaskCalibrationRule(
        require_baseline_intended_direction=True,
        min_abs_baseline_margin=None,
        min_tasks_per_family=2,
        allow_family_drop=False,
        reason="test",
    )

    result = apply_task_calibration(
        tasks=_tasks(),
        baseline_rows=_baseline_rows(wrong_task="sentiment_negation_0"),
        rule=rule,
    )

    assert "sentiment_negation_0" not in result.kept_task_ids
    assert result.exclusions_by_reason == {"baseline_wrong_direction": 1}
    assert result.passes_minimum is True


def test_calibration_excludes_low_margin_tasks() -> None:
    rule = task_calibration_rule_from_mode(
        mode="baseline-margin",
        min_abs_baseline_margin=0.1,
        min_tasks_per_family=2,
        allow_family_drop=False,
    )
    assert rule is not None

    result = apply_task_calibration(
        tasks=_tasks(),
        baseline_rows=_baseline_rows(low_margin_task="property_negation_1"),
        rule=rule,
    )

    assert "property_negation_1" not in result.kept_task_ids
    assert result.exclusions_by_reason == {"baseline_margin_below_threshold": 1}
    assert result.passes_minimum is True


def test_calibration_fails_closed_if_required_family_underfilled() -> None:
    rule = TaskCalibrationRule(
        require_baseline_intended_direction=True,
        min_abs_baseline_margin=None,
        min_tasks_per_family=3,
        allow_family_drop=False,
        reason="test",
    )

    result = apply_task_calibration(
        tasks=_tasks(),
        baseline_rows=_baseline_rows(wrong_task="state_negation_0"),
        rule=rule,
    )

    assert result.valid_by_family_after["state_negation"] == 2
    assert result.passes_minimum is False
    assert result.missing_required_families == ["state_negation"]


def test_calibration_reports_missing_baseline_rows() -> None:
    rows = _baseline_rows()
    rows = [row for row in rows if row["task_id"] != "property_negation_2"]
    rule = TaskCalibrationRule(
        require_baseline_intended_direction=True,
        min_abs_baseline_margin=None,
        min_tasks_per_family=2,
        allow_family_drop=False,
        reason="test",
    )

    result = apply_task_calibration(tasks=_tasks(), baseline_rows=rows, rule=rule)

    assert result.exclusions_by_reason == {"missing_baseline_row": 1}
    assert "property_negation_2" not in result.kept_task_ids
