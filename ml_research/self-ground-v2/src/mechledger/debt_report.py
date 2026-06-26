from __future__ import annotations

import json
from pathlib import Path

from mechledger.assessments.candidate import assess_run_evidence
from mechledger.assessments.evidence_report import write_evidence_assessment
from mechledger.core.debt import (
    ScientificDebtReport,
)
from mechledger.project import Project, now_utc


def generate_scientific_debt_report(project: Project, run_id: str) -> ScientificDebtReport:
    run_dir = project.runs_dir / run_id
    assessment = assess_run_evidence(run_dir, project_root=project.root)
    write_evidence_assessment(run_dir, assessment)
    report = ScientificDebtReport(
        report_id=f"SDR-{run_id}",
        run_id=run_id,
        experiment_id=_run_experiment_id(run_dir),
        generated_at=now_utc(),
        evaluated_assessments=sorted(
            {
                debt.assessment_id
                for debt in assessment.debts
                if debt.assessment_id is not None
            }
            | {
                "run_class_allowed",
                "baseline_calibration",
                "positive_control",
                "empirical_null",
                "paired_statistic",
                "matched_controls",
                "telemetry",
            }
        ),
        debts=assessment.debts,
        threshold_sources=[
            {
                "assessment_id": _assessment_id_for_condition(condition.condition_id),
                "condition_id": condition.condition_id,
                "metric_name": None,
                "threshold_source": condition.threshold_source.value,
                "threshold_justification": condition.threshold_justification,
                "threshold_decision_id": condition.threshold_decision_id,
            }
            for condition in assessment.conditions.values()
            if condition.threshold_source is not None
        ],
        clean_candidate_support=assessment.clean_candidate_support,
        summary=assessment.summary,
    )
    write_scientific_debt_report(run_dir, report)
    return report


def write_scientific_debt_report(run_dir: Path, report: ScientificDebtReport) -> None:
    payload = report.model_dump(mode="json")
    payload["blockers"] = [item.model_dump(mode="json") for item in report.blockers]
    payload["warnings"] = [item.model_dump(mode="json") for item in report.warnings]
    (run_dir / "scientific_debt_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Scientific Debt Report for {report.run_id}",
        "",
        report.summary,
        "",
        "## Debts",
    ]
    for debt in report.debts:
        waiver = f" waiver={debt.waiver_decision_id}" if debt.waiver_decision_id else ""
        lines.append(
            f"- {debt.debt_id} [{debt.severity.value}/{debt.status}]"
            f"{waiver} {debt.debt_type}: {debt.message}"
        )
    (run_dir / "scientific_debt_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_experiment_id(run_dir: Path) -> str | None:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return None
    return json.loads(run_json.read_text(encoding="utf-8")).get("experiment_id")


def _assessment_id_for_condition(condition_id: str) -> str:
    if condition_id in {"baseline_calibration_recorded"}:
        return "baseline_calibration"
    if condition_id in {"positive_control_pass_rate"}:
        return "positive_control"
    if condition_id.startswith("empirical_null") or condition_id == "random_null_seed_count":
        return "empirical_null"
    if condition_id.startswith("paired_"):
        return "paired_statistic"
    if condition_id in {
        "matched_controls_present",
        "specificity_gap_positive",
        "top_control_ratio",
        "multi_control_min_gap",
        "family_min_gap",
    }:
        return "matched_controls"
    if condition_id in {"relative_norm_drift", "nonfinite_rate", "skip_rate"}:
        return "telemetry"
    if condition_id == "seed_sensitivity":
        return "seed_sensitivity"
    return "candidate_claim"
