from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mechledger.assessments.calibration import (
    evaluate_baseline_calibration,
    evaluate_positive_control,
)
from mechledger.assessments.common import AssessmentConditionResult, AssessmentReport
from mechledger.assessments.compatibility import evaluate_compatibility_record
from mechledger.assessments.controls import evaluate_matched_controls
from mechledger.assessments.empirical_null import evaluate_empirical_null
from mechledger.assessments.evidence_report import EvidenceAssessmentReport
from mechledger.assessments.paired_statistic import evaluate_paired_statistic
from mechledger.assessments.seed_sensitivity import evaluate_seed_sensitivity
from mechledger.assessments.telemetry import evaluate_telemetry
from mechledger.core.debt import DebtSeverity, DebtStatus, DebtType, ScientificDebtRecord
from mechledger.core.decision_log import parse_decision_log
from mechledger.project import now_utc

ALLOWED_CLEAN_RUN_CLASSES = {
    "serious_evidence_run",
    "paper_candidate",
    "replication",
    "published_result",
}

NON_CANDIDATE_RUN_CLASS_DEBT = {
    "diagnostic": DebtType.DIAGNOSTIC_RUN_ONLY,
    "smoke_test": DebtType.SMOKE_TEST_ONLY,
    "notebook_exploration": DebtType.UNREVIEWED_NOTEBOOK_RUN,
    "scratch": DebtType.CUSTOM,
}

DEBT_ID_BY_CONDITION = {
    "run_class_allowed": "DPT001",
    "positive_control_pass_rate": "DPT002",
    "baseline_calibration_recorded": "DPT003",
    "nonfinite_rate": "DPT004",
    "relative_norm_drift": "DPT005",
    "empirical_null_present": "DPT006",
    "random_null_seed_count": "DPT007",
    "empirical_null_percentile_rank": "DPT008",
    "paired_statistic_present": "DPT009",
    "paired_n_pairs": "DPT010",
    "paired_p_value": "DPT011",
    "paired_effect_direction": "DPT012",
    "matched_controls_present": "DPT013",
    "specificity_gap_positive": "DPT014",
    "top_control_ratio": "DPT015",
    "multi_control_min_gap": "DPT016",
    "family_min_gap": "DPT017",
    "metadata_compatibility": "DPT018",
    "skip_rate": "DPT019",
    "paired_sign_consistency": "DPT020",
    "seed_sensitivity": "DPT021",
}

THRESHOLD_CONDITION_TYPES = {
    "metric_threshold",
    "seed_count_at_least",
    "task_count_at_least",
    "family_count_at_least",
    "positive_control_passed",
    "baseline_calibration_recorded",
    "skip_rate_below",
    "nonfinite_rate_zero",
}

NEGATIVE_EVIDENCE_DEBT_TYPES = {
    "specificity_gap_nonpositive",
    "collateral_ratio_high",
    "multi_control_gap_failed",
    "family_gap_failed",
    "failed_positive_control",
    "nonfinite_rows",
    "all_rows_skipped",
}


def assess_run_evidence(run_dir: Path, *, project_root: Path) -> EvidenceAssessmentReport:
    run_data = _read_json(run_dir / "run.json")
    run_id = str(run_data["run_id"])
    run_class = str(run_data.get("run_class") or "scratch")
    run_status = str(run_data.get("status") or "unknown")
    metrics = _load_metrics(run_dir / "metrics.jsonl")
    artifacts = _load_artifacts(run_dir)
    reports = _assessment_reports(run_class, run_status, metrics, artifacts)
    conditions = _flatten_conditions(reports)
    prior_waivers = _load_valid_waivers(
        run_dir / "scientific_debt_report.json",
        project_root / "research/logs/decision_log.md",
    )
    debts = _debts_from_conditions(
        run_id=run_id,
        experiment_id=run_data.get("experiment_id"),
        reports=reports,
        prior_waivers=prior_waivers,
    )
    for debt_id, prior in prior_waivers.items():
        if not any(debt.debt_id == debt_id for debt in debts):
            debts.append(prior)
    unwaived_failed_required = _unwaived_failed_required(conditions, debts)
    open_serious_or_blocking = [
        debt
        for debt in debts
        if debt.status == DebtStatus.OPEN
        and debt.severity in {DebtSeverity.SERIOUS, DebtSeverity.BLOCKING}
    ]
    clean = (
        run_class in ALLOWED_CLEAN_RUN_CLASSES
        and run_status == "completed"
        and not unwaived_failed_required
        and not open_serious_or_blocking
    )
    recommended = _recommended_claim_status(run_class, run_status, conditions, clean)
    open_debt_count = len([debt for debt in debts if debt.status == DebtStatus.OPEN])
    summary = (
        f"recommended_claim_status: {recommended}; "
        f"clean_candidate_support={clean}; open_debt={open_debt_count}"
    )
    return EvidenceAssessmentReport(
        run_id=run_id,
        generated_at=now_utc(),
        run_class=run_class,
        clean_candidate_support=clean,
        recommended_claim_status=recommended,
        conditions=conditions,
        debts=sorted(debts, key=lambda debt: debt.debt_id),
        waivers_applied=sorted(debt.debt_id for debt in debts if debt.status == DebtStatus.WAIVED),
        summary=summary,
    )


def _assessment_reports(
    run_class: str,
    run_status: str,
    metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> list[AssessmentReport]:
    reports = [
        AssessmentReport(
            assessment_id="run_status_completed",
            conditions={"run_status_completed": _run_status_condition(run_status)},
        ),
        AssessmentReport(
            assessment_id="run_class_allowed",
            conditions={"run_class_allowed": _run_class_condition(run_class)},
        ),
        AssessmentReport(
            assessment_id="baseline_calibration",
            conditions={
                "baseline_calibration_recorded": evaluate_baseline_calibration(metrics)
            },
        ),
        AssessmentReport(
            assessment_id="positive_control",
            conditions={"positive_control_pass_rate": evaluate_positive_control(metrics)},
        ),
        evaluate_empirical_null(metrics, artifacts=artifacts, percentile_threshold=0.95),
        evaluate_paired_statistic(metrics, required=True),
        evaluate_matched_controls(metrics, required=True),
        evaluate_telemetry(metrics),
        evaluate_seed_sensitivity(metrics),
    ]
    if "metadata_compatible" in metrics or "compatible" in metrics:
        reports.append(
            AssessmentReport(
                assessment_id="metadata_compatibility",
                conditions={
                    "metadata_compatibility": evaluate_compatibility_record(
                        {
                            "compatible": bool(
                                metrics.get("metadata_compatible", metrics.get("compatible"))
                            ),
                            "diagnostic_only": bool(metrics.get("diagnostic_only")),
                            "error": metrics.get("metadata_error"),
                        }
                    )
                },
            )
        )
    return reports


def _run_class_condition(run_class: str) -> AssessmentConditionResult:
    allowed = run_class in ALLOWED_CLEAN_RUN_CLASSES
    debt_type = NON_CANDIDATE_RUN_CLASS_DEBT.get(run_class, DebtType.CUSTOM)
    return AssessmentConditionResult(
        condition_id="run_class_allowed",
        condition_type="run_class_allowed",
        passed=allowed,
        parameters={"run_class": run_class, "allowed": sorted(ALLOWED_CLEAN_RUN_CLASSES)},
        threshold_source=None,
        failure_message=f"Run class `{run_class}` cannot provide clean candidate support.",
        default_consequence="scientific_debt",
        debt_type=debt_type.value if hasattr(debt_type, "value") else str(debt_type),
        severity=DebtSeverity.SERIOUS,
    )


def _run_status_condition(run_status: str) -> AssessmentConditionResult:
    return AssessmentConditionResult(
        condition_id="run_status_completed",
        condition_type="status_at_least",
        passed=run_status == "completed",
        parameters={"status": run_status, "required_status": "completed"},
        threshold_source=None,
        failure_message=(
            f"Run status `{run_status}` is not completed and cannot provide "
            "clean candidate support."
        ),
        default_consequence="blocker",
        debt_type=DebtType.CUSTOM.value,
        severity=DebtSeverity.BLOCKING,
    )


def _flatten_conditions(
    reports: list[AssessmentReport],
) -> dict[str, AssessmentConditionResult]:
    conditions: dict[str, AssessmentConditionResult] = {}
    for report in reports:
        for condition_id, condition in report.conditions.items():
            conditions[condition_id] = condition
    return dict(sorted(conditions.items()))


def _debts_from_conditions(
    *,
    run_id: str,
    experiment_id: str | None,
    reports: list[AssessmentReport],
    prior_waivers: dict[str, ScientificDebtRecord],
) -> list[ScientificDebtRecord]:
    debts: list[ScientificDebtRecord] = []
    for report in reports:
        for condition in report.conditions.values():
            if not condition.passed and condition.debt_type and condition.severity:
                debt_id = DEBT_ID_BY_CONDITION.get(
                    condition.condition_id, f"DPT-{condition.condition_id}"
                )
                debt = _condition_debt(
                    debt_id=debt_id,
                    report_id=report.assessment_id,
                    condition=condition,
                    run_id=run_id,
                    experiment_id=experiment_id,
                )
                if debt_id in prior_waivers:
                    debt = debt.model_copy(
                        update={
                            "status": DebtStatus.WAIVED,
                            "waiver_decision_id": prior_waivers[debt_id].waiver_decision_id,
                            "resolved_at": prior_waivers[debt_id].resolved_at,
                        }
                    )
                debts.append(debt)
            if _uses_unreviewed_tool_default(condition):
                debts.append(
                    ScientificDebtRecord(
                        debt_id=f"DTD-{condition.condition_id}",
                        debt_type=DebtType.UNJUSTIFIED_THRESHOLD_DEFAULT,
                        severity=DebtSeverity.INFO,
                        claim_id=None,
                        run_id=run_id,
                        experiment_id=experiment_id,
                        evidence_paths=[],
                        message=(
                            f"Condition {condition.condition_id} uses an unreviewed "
                            "tool-default threshold."
                        ),
                        required_resolution="Review threshold policy or accept the tool default.",
                        status=DebtStatus.OPEN,
                        waiver_decision_id=None,
                        created_at=now_utc(),
                        assessment_id=report.assessment_id,
                    )
                )
    return _dedupe_debts(debts)


def _condition_debt(
    *,
    debt_id: str,
    report_id: str,
    condition: AssessmentConditionResult,
    run_id: str,
    experiment_id: str | None,
) -> ScientificDebtRecord:
    return ScientificDebtRecord(
        debt_id=debt_id,
        debt_type=str(condition.debt_type),
        severity=condition.severity or DebtSeverity.WARNING,
        claim_id=None,
        run_id=run_id,
        experiment_id=experiment_id,
        evidence_paths=[],
        message=condition.failure_message,
        required_resolution=_resolution_for(condition),
        status=DebtStatus.OPEN,
        waiver_decision_id=None,
        created_at=now_utc(),
        assessment_id=report_id,
    )


def _resolution_for(condition: AssessmentConditionResult) -> str:
    if condition.debt_type == "missing_empirical_null":
        return (
            "Register an empirical-null distribution with at least 30 seeds "
            "or waive by decision."
        )
    if condition.debt_type == "missing_paired_statistic":
        return "Register paired-statistic metadata or waive by decision."
    if condition.debt_type == "missing_matched_controls":
        return "Register matched-control metrics or waive by decision."
    if condition.debt_type == "failed_positive_control":
        return "Fix the intervention/scoring harness and rerun the positive control."
    return "Register stronger evidence, resolve the metric failure, or waive by accepted decision."


def _uses_unreviewed_tool_default(condition: AssessmentConditionResult) -> bool:
    return (
        condition.condition_type in THRESHOLD_CONDITION_TYPES
        and condition.threshold_source is not None
        and condition.threshold_source.value == "tool_default"
        and condition.threshold_decision_id is None
    )


def _dedupe_debts(debts: list[ScientificDebtRecord]) -> list[ScientificDebtRecord]:
    deduped: dict[str, ScientificDebtRecord] = {}
    for debt in debts:
        deduped[debt.debt_id] = debt
    return list(deduped.values())


def _load_valid_waivers(
    report_path: Path,
    decision_log_path: Path,
) -> dict[str, ScientificDebtRecord]:
    if not report_path.exists():
        return {}
    payload = _read_json(report_path)
    decisions = parse_decision_log(decision_log_path).decisions
    waivers: dict[str, ScientificDebtRecord] = {}
    for raw_debt in payload.get("debts", []):
        if raw_debt.get("status") != DebtStatus.WAIVED:
            continue
        decision_id = raw_debt.get("waiver_decision_id")
        decision = decisions.get(str(decision_id or ""))
        if decision is None or decision.status != "accepted":
            raise ValueError(
                f"Debt {raw_debt.get('debt_id')} has invalid waiver decision {decision_id}."
            )
        debt = ScientificDebtRecord.model_validate(raw_debt)
        waivers[debt.debt_id] = debt
    return waivers


def _unwaived_failed_required(
    conditions: dict[str, AssessmentConditionResult],
    debts: list[ScientificDebtRecord],
) -> list[AssessmentConditionResult]:
    waived = {debt.debt_id for debt in debts if debt.status == DebtStatus.WAIVED}
    failed = []
    for condition in conditions.values():
        debt_id = DEBT_ID_BY_CONDITION.get(condition.condition_id, f"DPT-{condition.condition_id}")
        if condition.passed or debt_id in waived:
            continue
        if condition.severity in {DebtSeverity.SERIOUS, DebtSeverity.BLOCKING}:
            failed.append(condition)
    return failed


def _recommended_claim_status(
    run_class: str,
    run_status: str,
    conditions: dict[str, AssessmentConditionResult],
    clean: bool,
) -> str:
    if clean:
        return "candidate_claim"
    if run_status == "failed":
        return "failed_or_weakened"
    if run_status in {"cancelled", "interrupted", "running", "created"}:
        return "unsupported"
    failed_debt_types = {
        str(condition.debt_type)
        for condition in conditions.values()
        if not condition.passed and condition.debt_type
    }
    if failed_debt_types & NEGATIVE_EVIDENCE_DEBT_TYPES:
        return "failed_or_weakened"
    if run_class in ALLOWED_CLEAN_RUN_CLASSES:
        return "single_run_evidence"
    if any(condition.passed for condition in conditions.values()):
        return "exploratory_signal"
    return "unsupported"


def _load_metrics(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metric_name = row.get("metric_name")
        if metric_name:
            metrics[str(metric_name)] = row.get("value")
    return metrics


def _load_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    manifest = _read_json(run_dir / "artifact_manifest.json")
    artifacts = list(manifest.get("artifacts", []))
    artifacts_jsonl = run_dir / "artifacts.jsonl"
    if artifacts_jsonl.exists():
        for line in artifacts_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                artifact = json.loads(line)
                if isinstance(artifact, dict):
                    artifacts.append(artifact)
    return artifacts


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
