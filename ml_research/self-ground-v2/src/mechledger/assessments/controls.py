from __future__ import annotations

from typing import Any

from mechledger.assessments.common import AssessmentConditionResult, AssessmentReport
from mechledger.core.debt import DebtSeverity


def evaluate_matched_controls(
    metrics: dict[str, Any],
    *,
    required: bool = True,
    max_top_control_ratio: float = 0.5,
) -> AssessmentReport:
    target_delta = _as_float(metrics.get("target_delta"))
    control_delta = _as_float(metrics.get("matched_control_delta", metrics.get("control_delta")))
    specificity_gap = _as_float(metrics.get("specificity_gap", metrics.get("specificity_gap_mean")))
    if specificity_gap is None and target_delta is not None and control_delta is not None:
        specificity_gap = target_delta - control_delta
    top_control_ratio = _as_float(metrics.get("top_control_ratio", metrics.get("collateral_ratio")))
    multi_control_min_gap = _as_float(metrics.get("multi_control_min_gap"))
    family_min_gap = _as_float(metrics.get("family_min_gap"))
    has_controls = control_delta is not None or top_control_ratio is not None
    conditions = {
        "matched_controls_present": AssessmentConditionResult(
            condition_id="matched_controls_present",
            condition_type="artifact_exists",
            passed=has_controls or not required,
            parameters={
                "matched_control_delta": control_delta,
                "top_control_ratio": top_control_ratio,
            },
            threshold_source=None,
            failure_message="Matched-control evidence is missing.",
            default_consequence="scientific_debt",
            debt_type="missing_matched_controls",
            severity=DebtSeverity.SERIOUS,
        ),
        "specificity_gap_positive": AssessmentConditionResult(
            condition_id="specificity_gap_positive",
            condition_type="metric_threshold",
            passed=specificity_gap is not None and specificity_gap > 0.0,
            parameters={
                "specificity_gap": specificity_gap,
                "min_specificity_gap": 0.0,
                "threshold": 0.0,
            },
            failure_message="Specificity gap is missing or non-positive.",
            default_consequence="scientific_debt",
            debt_type="specificity_gap_nonpositive",
            severity=DebtSeverity.SERIOUS,
        ),
    }
    if top_control_ratio is not None:
        conditions["top_control_ratio"] = AssessmentConditionResult(
            condition_id="top_control_ratio",
            condition_type="metric_threshold",
            passed=top_control_ratio <= max_top_control_ratio,
            parameters={
                "top_control_ratio": top_control_ratio,
                "max_top_control_ratio": max_top_control_ratio,
                "threshold": max_top_control_ratio,
            },
            failure_message="Collateral/control ratio is above the starter default threshold.",
            default_consequence="scientific_debt",
            debt_type="collateral_ratio_high",
            severity=DebtSeverity.SERIOUS,
        )
    if multi_control_min_gap is not None:
        conditions["multi_control_min_gap"] = AssessmentConditionResult(
            condition_id="multi_control_min_gap",
            condition_type="metric_threshold",
            passed=multi_control_min_gap >= 0.0,
            parameters={
                "multi_control_min_gap": multi_control_min_gap,
                "min_gap": 0.0,
                "threshold": 0.0,
            },
            failure_message="At least one multi-control gap is negative.",
            default_consequence="scientific_debt",
            debt_type="multi_control_gap_failed",
            severity=DebtSeverity.SERIOUS,
        )
    if family_min_gap is not None:
        conditions["family_min_gap"] = AssessmentConditionResult(
            condition_id="family_min_gap",
            condition_type="metric_threshold",
            passed=family_min_gap >= 0.0,
            parameters={"family_min_gap": family_min_gap, "min_gap": 0.0, "threshold": 0.0},
            failure_message="At least one family-specific control gap is negative.",
            default_consequence="scientific_debt",
            debt_type="family_gap_failed",
            severity=DebtSeverity.SERIOUS,
        )
    return AssessmentReport(assessment_id="matched_controls", conditions=conditions)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
