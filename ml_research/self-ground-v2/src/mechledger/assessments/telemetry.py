from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

from mechledger.assessments.common import AssessmentConditionResult, AssessmentReport
from mechledger.core.debt import DebtSeverity


class InterventionTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_feature_activation_mean: float | None = None
    selected_feature_activation_abs_mean: float | None = None
    selected_feature_modified_mean: float | None = None
    selected_feature_delta_abs_mean: float | None = None
    decoded_delta_norm_mean: float | None = None
    activation_norm_mean: float | None = None
    patched_activation_norm_mean: float | None = None
    relative_norm_drift: float | None = None
    relative_norm_drift_mean: float | None = None
    decoded_delta_norm_ratio: float | None = None
    nonfinite_rate: float | None = None
    skip_rate: float | None = None


def telemetry_has_nonfinite(telemetry: InterventionTelemetry | dict[str, Any]) -> bool:
    values = _values(telemetry)
    return any(not math.isfinite(float(value)) for value in values.values() if value is not None)


def telemetry_warnings(
    telemetry: InterventionTelemetry | dict[str, Any],
    *,
    max_relative_norm_drift_warning: float = 0.5,
    max_decoded_delta_norm_ratio_warning: float = 0.5,
) -> dict[str, bool]:
    values = _values(telemetry)
    drift = values.get("relative_norm_drift", values.get("relative_norm_drift_mean", 0.0))
    return {
        "norm_drift_warning": float(drift or 0.0) > max_relative_norm_drift_warning,
        "decoded_delta_norm_ratio_warning": float(values.get("decoded_delta_norm_ratio") or 0.0)
        > max_decoded_delta_norm_ratio_warning,
    }


def mean_telemetry(
    left: InterventionTelemetry | dict[str, Any],
    right: InterventionTelemetry | dict[str, Any],
) -> dict[str, float]:
    left_values = _values(left)
    right_values = _values(right)
    means: dict[str, float] = {}
    for key in sorted(set(left_values) | set(right_values)):
        if key in left_values and key in right_values:
            means[key] = (float(left_values[key]) + float(right_values[key])) / 2.0
    return means


def evaluate_telemetry(metrics: dict[str, Any]) -> AssessmentReport:
    drift = float(
        metrics.get("relative_norm_drift", metrics.get("relative_norm_drift_mean", 0.0)) or 0.0
    )
    nonfinite_rate = float(metrics.get("nonfinite_rate") or 0.0)
    skip_rate = float(metrics.get("skip_rate") or 0.0)
    conditions = {
        "relative_norm_drift": AssessmentConditionResult(
            condition_id="relative_norm_drift",
            condition_type="metric_threshold",
            passed=drift <= 0.5,
            parameters={"relative_norm_drift": drift, "max": 0.5},
            failure_message="Relative norm drift exceeds the starter default 0.5 threshold.",
            default_consequence="scientific_debt",
            debt_type="high_norm_drift",
            severity=DebtSeverity.SERIOUS,
        ),
        "nonfinite_rate": AssessmentConditionResult(
            condition_id="nonfinite_rate",
            condition_type="nonfinite_rate_zero",
            passed=nonfinite_rate == 0.0,
            parameters={"nonfinite_rate": nonfinite_rate},
            failure_message="Non-finite telemetry or metric rows are present.",
            default_consequence="blocker",
            debt_type="nonfinite_rows",
            severity=DebtSeverity.BLOCKING,
        ),
        "skip_rate": AssessmentConditionResult(
            condition_id="skip_rate",
            condition_type="skip_rate_below",
            passed=skip_rate < 1.0,
            parameters={"skip_rate": skip_rate},
            failure_message="All rows were skipped.",
            default_consequence="blocker",
            debt_type="all_rows_skipped",
            severity=DebtSeverity.BLOCKING,
        ),
    }
    return AssessmentReport(assessment_id="telemetry", conditions=conditions)


def _values(telemetry: InterventionTelemetry | dict[str, Any]) -> dict[str, Any]:
    if isinstance(telemetry, InterventionTelemetry):
        return telemetry.model_dump(exclude_none=True)
    return telemetry
