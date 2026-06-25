from __future__ import annotations

from mechledger.assessments.calibration import (
    TaskCalibrationRule,
    apply_task_calibration,
    evaluate_positive_control,
)
from mechledger.assessments.compatibility import (
    evaluate_compatibility_record,
    metadata_matches_requested_target,
    sae_shapes_are_compatible,
)
from mechledger.assessments.telemetry import evaluate_telemetry, telemetry_has_nonfinite
from mechledger.core.debt import DebtSeverity


def test_task_calibration_is_pure_dict_logic() -> None:
    result = apply_task_calibration(
        tasks=[
            {"id": "a", "family": "property_negation"},
            {"id": "b", "family": "property_negation"},
        ],
        baseline_rows=[
            {"task_id": "a", "baseline_prompt_contrast": 0.2, "intended_direction_pass": True},
            {"task_id": "b", "baseline_prompt_contrast": -0.1, "intended_direction_pass": False},
        ],
        rule=TaskCalibrationRule(
            require_baseline_intended_direction=True,
            min_abs_baseline_margin=0.1,
            min_tasks_per_family=1,
            required_families=["property_negation"],
            allow_family_drop=False,
            reason="test",
        ),
    )

    assert result.kept_task_ids == ["a"]
    assert result.exclusions_by_reason == {"baseline_wrong_direction": 1}
    assert result.passes_minimum


def test_positive_control_failure_is_blocking_condition() -> None:
    condition = evaluate_positive_control({"positive_control_pass_rate": 0.7})

    assert not condition.passed
    assert condition.default_consequence == "blocker"
    assert condition.debt_type == "failed_positive_control"


def test_compatibility_ports_shape_and_metadata_logic_without_framework_objects() -> None:
    assert sae_shapes_are_compatible([2, 10, 768], [2, 768])
    report = metadata_matches_requested_target(
        metadata={
            "declared_model": "EleutherAI/pythia-70m-deduped",
            "declared_hook_point": "blocks.2.hook_resid_post",
            "missing_fields": [],
        },
        requested_model_name="pythia-70m-deduped",
        requested_hook_point="blocks.2.hook_resid_post",
    )
    condition = evaluate_compatibility_record({"compatible": True, **report})

    assert report["metadata_compatible"]
    assert condition.passed


def test_telemetry_nonfinite_and_norm_drift_conditions() -> None:
    assert telemetry_has_nonfinite({"relative_norm_drift": float("inf")})
    report = evaluate_telemetry(
        {"relative_norm_drift": 0.7, "nonfinite_rate": 0.0, "skip_rate": 0.0}
    )

    assert report.conditions["relative_norm_drift"].severity == DebtSeverity.SERIOUS
    assert not report.conditions["relative_norm_drift"].passed
