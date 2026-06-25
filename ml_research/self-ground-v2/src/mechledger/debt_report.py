from __future__ import annotations

import json
from pathlib import Path

from mechledger.core.debt import (
    DebtSeverity,
    DebtStatus,
    DebtType,
    ScientificDebtRecord,
    ScientificDebtReport,
)
from mechledger.project import Project, now_utc


def generate_scientific_debt_report(project: Project, run_id: str) -> ScientificDebtReport:
    run_dir = project.runs_dir / run_id
    run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    debts: list[ScientificDebtRecord] = []
    run_class = run_data.get("run_class")
    if run_class in {"diagnostic", "notebook_exploration", "smoke_test", "scratch"}:
        debt_type = {
            "diagnostic": DebtType.DIAGNOSTIC_RUN_ONLY,
            "notebook_exploration": DebtType.UNREVIEWED_NOTEBOOK_RUN,
            "smoke_test": DebtType.SMOKE_TEST_ONLY,
            "scratch": DebtType.CUSTOM,
        }[run_class]
        debts.append(
            _debt(
                run_id=run_id,
                experiment_id=run_data.get("experiment_id"),
                debt_id="DPT001",
                debt_type=debt_type,
                severity=DebtSeverity.SERIOUS,
                message=f"Run class `{run_class}` cannot provide clean candidate support.",
                required_resolution="Reclassify by human review or run a serious evidence run.",
            )
        )
    metrics = _load_metrics(run_dir / "metrics.jsonl")
    if not any(item.get("metric_name") == "positive_control_pass_rate" for item in metrics):
        debts.append(
            _debt(
                run_id=run_id,
                experiment_id=run_data.get("experiment_id"),
                debt_id="DPT002",
                debt_type=DebtType.MISSING_POSITIVE_CONTROL,
                severity=DebtSeverity.WARNING,
                message="No positive-control pass-rate metric was recorded.",
                required_resolution=(
                    "Record positive_control_pass_rate or waive by accepted decision."
                ),
            )
        )
    if not any(item.get("metric_name", "").startswith("baseline_") for item in metrics):
        debts.append(
            _debt(
                run_id=run_id,
                experiment_id=run_data.get("experiment_id"),
                debt_id="DPT003",
                debt_type=DebtType.MISSING_BASELINE_CALIBRATION,
                severity=DebtSeverity.WARNING,
                message="No baseline calibration metric was recorded.",
                required_resolution="Record baseline calibration metrics.",
            )
        )
    for metric in metrics:
        if metric.get("metric_name") == "nonfinite_rate" and (metric.get("value") or 0) > 0:
            debts.append(
                _debt(
                    run_id=run_id,
                    experiment_id=run_data.get("experiment_id"),
                    debt_id="DPT004",
                    debt_type=DebtType.NONFINITE_ROWS,
                    severity=DebtSeverity.BLOCKING,
                    message="Nonfinite rows were recorded.",
                    required_resolution="Fix nonfinite metrics and rerun.",
                )
            )
        if metric.get("metric_name") == "relative_norm_drift" and (metric.get("value") or 0) > 0.5:
            debts.append(
                _debt(
                    run_id=run_id,
                    experiment_id=run_data.get("experiment_id"),
                    debt_id="DPT005",
                    debt_type=DebtType.HIGH_NORM_DRIFT,
                    severity=DebtSeverity.SERIOUS,
                    message="Relative norm drift exceeded the tool default threshold.",
                    required_resolution="Lower norm drift or justify threshold by decision.",
                )
            )
    clean = run_class in {
        "serious_evidence_run",
        "paper_candidate",
        "replication",
        "published_result",
    } and not any(debt.severity in {DebtSeverity.BLOCKING, DebtSeverity.SERIOUS} for debt in debts)
    report = ScientificDebtReport(
        report_id=f"SDR-{run_id}",
        run_id=run_id,
        experiment_id=run_data.get("experiment_id"),
        generated_at=now_utc(),
        evaluated_assessments=[
            "run_class_allowed",
            "baseline_calibration",
            "positive_control",
            "telemetry",
        ],
        debts=debts,
        threshold_sources=[],
        clean_candidate_support=clean,
        summary=f"{len(debts)} debt records; clean_candidate_support={clean}",
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


def _debt(
    *,
    run_id: str,
    experiment_id: str | None,
    debt_id: str,
    debt_type: DebtType,
    severity: DebtSeverity,
    message: str,
    required_resolution: str,
) -> ScientificDebtRecord:
    return ScientificDebtRecord(
        debt_id=debt_id,
        debt_type=debt_type,
        severity=severity,
        claim_id=None,
        run_id=run_id,
        experiment_id=experiment_id,
        evidence_paths=[],
        message=message,
        required_resolution=required_resolution,
        status=DebtStatus.OPEN,
        waiver_decision_id=None,
        created_at=now_utc(),
    )


def _load_metrics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
