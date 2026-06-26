from __future__ import annotations

from typing import Any

from mechledger.assessments.common import AssessmentConditionResult, AssessmentReport
from mechledger.core.debt import DebtSeverity


def evaluate_empirical_null(
    metrics: dict[str, Any],
    *,
    artifacts: list[dict[str, Any]] | None = None,
    min_seed_count: int = 30,
    percentile_threshold: float | None = None,
) -> AssessmentReport:
    artifacts = artifacts or []
    null_path = metrics.get("null_distribution_path")
    has_required_artifact = any(
        artifact.get("claim_relevance") == "required" for artifact in artifacts
    )
    present = bool(null_path) or has_required_artifact
    seed_count = _as_int(metrics.get("random_null_seed_count"))
    conditions = {
        "empirical_null_present": AssessmentConditionResult(
            condition_id="empirical_null_present",
            condition_type="artifact_exists",
            passed=present and seed_count is not None,
            parameters={
                "null_distribution_path": null_path,
                "has_required_artifact": has_required_artifact,
            },
            threshold_source=None,
            failure_message="Empirical-null distribution metadata is missing.",
            default_consequence="scientific_debt",
            debt_type="missing_empirical_null",
            severity=DebtSeverity.SERIOUS,
        ),
        "random_null_seed_count": AssessmentConditionResult(
            condition_id="random_null_seed_count",
            condition_type="seed_count_at_least",
            passed=seed_count is None or seed_count >= min_seed_count,
            parameters={
                "random_null_seed_count": seed_count,
                "min_seed_count": min_seed_count,
                "threshold": min_seed_count,
            },
            failure_message=(
                f"Empirical null has fewer than {min_seed_count} random-null seeds."
            ),
            default_consequence="scientific_debt",
            debt_type="insufficient_null_seeds"
            if seed_count is not None
            else "missing_empirical_null",
            severity=DebtSeverity.SERIOUS,
        ),
    }
    percentile_rank = _as_float(metrics.get("percentile_rank"))
    if percentile_threshold is not None and percentile_rank is not None:
        conditions["empirical_null_percentile_rank"] = AssessmentConditionResult(
            condition_id="empirical_null_percentile_rank",
            condition_type="metric_threshold",
            passed=percentile_rank >= percentile_threshold,
            parameters={
                "percentile_rank": percentile_rank,
                "min_percentile_rank": percentile_threshold,
                "threshold": percentile_threshold,
            },
            failure_message="Observed score does not clear the empirical-null percentile policy.",
            default_consequence="scientific_debt",
            debt_type="empirical_null_percentile_below_threshold",
            severity=DebtSeverity.SERIOUS,
        )
    return AssessmentReport(assessment_id="empirical_null", conditions=conditions)


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
