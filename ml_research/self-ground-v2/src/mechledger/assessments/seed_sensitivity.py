from __future__ import annotations

from typing import Any

from mechledger.assessments.common import AssessmentConditionResult, AssessmentReport
from mechledger.core.debt import DebtSeverity


def evaluate_seed_sensitivity(metrics: dict[str, Any]) -> AssessmentReport:
    seed_count = _as_int(metrics.get("distinct_seed_count", metrics.get("seed_count")))
    conditions: dict[str, AssessmentConditionResult] = {}
    if seed_count is not None:
        conditions["seed_sensitivity"] = AssessmentConditionResult(
            condition_id="seed_sensitivity",
            condition_type="seed_count_at_least",
            passed=seed_count >= 2,
            parameters={"seed_count": seed_count, "min_seed_count": 2, "threshold": 2},
            failure_message="Only one seed is represented; seed sensitivity is unknown.",
            default_consequence="warning",
            debt_type="singleton_seed",
            severity=DebtSeverity.WARNING,
        )
    return AssessmentReport(assessment_id="seed_sensitivity", conditions=conditions)


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
