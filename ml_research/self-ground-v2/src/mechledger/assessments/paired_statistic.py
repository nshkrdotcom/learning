from __future__ import annotations

from typing import Any

from mechledger.assessments.common import AssessmentConditionResult, AssessmentReport
from mechledger.core.debt import DebtSeverity


def evaluate_paired_statistic(
    metrics: dict[str, Any],
    *,
    required: bool = True,
    min_pairs: int = 10,
    p_value_threshold: float = 0.05,
    min_sign_consistency: float | None = None,
) -> AssessmentReport:
    test_name = metrics.get("paired_test_name") or metrics.get("paired_test") or metrics.get("test")
    paired_by = metrics.get("paired_by")
    n_pairs = _as_int(metrics.get("paired_test_n_pairs", metrics.get("n_pairs")))
    p_value = _as_float(metrics.get("paired_test_p_value", metrics.get("p_value")))
    direction = metrics.get("effect_direction")
    sign_consistency = _as_float(metrics.get("sign_consistency"))
    present = bool(test_name and paired_by and n_pairs is not None)
    conditions = {
        "paired_statistic_present": AssessmentConditionResult(
            condition_id="paired_statistic_present",
            condition_type="artifact_exists",
            passed=present or not required,
            parameters={"paired_test_name": test_name, "paired_by": paired_by, "n_pairs": n_pairs},
            threshold_source=None,
            failure_message="Required paired-statistic registration is missing.",
            default_consequence="scientific_debt",
            debt_type="missing_paired_statistic",
            severity=DebtSeverity.SERIOUS,
        )
    }
    if present:
        conditions["paired_n_pairs"] = AssessmentConditionResult(
            condition_id="paired_n_pairs",
            condition_type="task_count_at_least",
            passed=n_pairs is not None and n_pairs >= min_pairs,
            parameters={"n_pairs": n_pairs, "min_pairs": min_pairs, "threshold": min_pairs},
            failure_message=f"Paired statistic has fewer than {min_pairs} paired rows.",
            default_consequence="scientific_debt",
            debt_type="insufficient_paired_pairs",
            severity=DebtSeverity.SERIOUS,
        )
        conditions["paired_p_value"] = AssessmentConditionResult(
            condition_id="paired_p_value",
            condition_type="metric_threshold",
            passed=p_value is not None and p_value <= p_value_threshold,
            parameters={
                "p_value": p_value,
                "max_p_value": p_value_threshold,
                "threshold": p_value_threshold,
            },
            failure_message="Registered paired statistic does not pass the p-value policy.",
            default_consequence="scientific_debt",
            debt_type="paired_statistic_failed",
            severity=DebtSeverity.SERIOUS,
        )
        conditions["paired_effect_direction"] = AssessmentConditionResult(
            condition_id="paired_effect_direction",
            condition_type="custom",
            passed=direction == "positive",
            parameters={"effect_direction": direction},
            threshold_source=None,
            failure_message="Registered paired statistic has missing or non-positive direction.",
            default_consequence="scientific_debt",
            debt_type="paired_effect_direction_failed",
            severity=DebtSeverity.SERIOUS,
        )
        if min_sign_consistency is not None or sign_consistency is not None:
            threshold = min_sign_consistency if min_sign_consistency is not None else 0.8
            conditions["paired_sign_consistency"] = AssessmentConditionResult(
                condition_id="paired_sign_consistency",
                condition_type="metric_threshold",
                passed=sign_consistency is not None and sign_consistency >= threshold,
                parameters={
                    "sign_consistency": sign_consistency,
                    "min_sign_consistency": threshold,
                    "threshold": threshold,
                },
                failure_message="Registered paired statistic has insufficient sign consistency.",
                default_consequence="scientific_debt",
                debt_type="paired_sign_consistency_low",
                severity=DebtSeverity.WARNING,
            )
    return AssessmentReport(assessment_id="paired_statistic", conditions=conditions)


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
