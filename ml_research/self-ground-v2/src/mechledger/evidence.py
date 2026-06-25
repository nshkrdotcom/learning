from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from mechledger.alias import resolve_run_alias
from mechledger.io import read_json, read_jsonl, utc_now, write_json
from mechledger.models import ScientificDebtRecord, ScientificDebtReport
from mechledger.paths import alias_cache_path, runs_dir

ALLOWED_CLEAN_RUN_CLASSES = {
    "serious_evidence_run",
    "paper_candidate",
    "replication",
    "published_result",
}


def gate_check(project_root: str | Path, run_id_or_alias: str) -> ScientificDebtReport:
    project_root = Path(project_root)
    run_id = _resolve_run_id(project_root, run_id_or_alias)
    run_dir = runs_dir(project_root) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))
    run = read_json(run_dir / "run.json")
    metrics = _metric_values(run_dir / "metrics.jsonl")
    manifest = (
        read_json(run_dir / "artifact_manifest.json")
        if (run_dir / "artifact_manifest.json").exists()
        else {"artifacts": []}
    )
    debts: list[ScientificDebtRecord] = []
    evaluated = [
        "run_class_allowed",
        "artifact_completeness",
        "baseline_calibration",
        "positive_control",
        "empirical_null",
        "paired_statistic",
        "telemetry",
        "claim_promotion_assessment",
    ]

    run_class = run.get("run_class")
    if run_class not in ALLOWED_CLEAN_RUN_CLASSES:
        debt_type = {
            "diagnostic": "diagnostic_run_only",
            "smoke_test": "smoke_test_only",
            "notebook_exploration": "unreviewed_notebook_run",
        }.get(str(run_class), "custom")
        debts.append(
            _debt(
                debt_type,
                "blocking",
                run,
                f"Run class {run_class!r} cannot provide clean candidate support by default.",
                "Reclassify with human rationale or run a serious evidence run.",
            )
        )

    if "baseline_contrast" not in metrics and "intended_direction_pass_rate" not in metrics:
        debts.append(
            _debt(
                "missing_baseline_calibration",
                "serious",
                run,
                "Baseline calibration metrics are not recorded.",
                "Register baseline calibration metrics or waive with an accepted decision.",
            )
        )
    if float(metrics.get("positive_control_pass_rate", 1.0) or 0.0) < 0.9:
        debts.append(
            _debt(
                "failed_positive_control",
                "blocking",
                run,
                "Positive-control pass rate is below the default 0.9 threshold.",
                "Debug scoring/control setup or justify a different threshold with a decision.",
            )
        )
    null_seed_count = metrics.get("random_null_seed_count")
    if null_seed_count is None:
        debts.append(
            _debt(
                "missing_empirical_null",
                "serious",
                run,
                "No empirical-null seed count was registered.",
                "Register an empirical null distribution.",
            )
        )
    elif float(null_seed_count) < 30:
        debts.append(
            _debt(
                "insufficient_null_seeds",
                "serious",
                run,
                "Empirical-null seed count is below the starter default of 30.",
                "Run at least 30 null seeds or waive with an accepted decision.",
            )
        )
    if float(metrics.get("nonfinite_rate", 0.0) or 0.0) > 0:
        debts.append(
            _debt(
                "nonfinite_rows",
                "blocking",
                run,
                "Non-finite rows are present in required metrics.",
                "Fix non-finite rows before candidate support.",
            )
        )
    if float(metrics.get("skip_rate", 0.0) or 0.0) >= 1.0:
        debts.append(
            _debt(
                "all_rows_skipped",
                "blocking",
                run,
                "All rows were skipped.",
                "Fix skipped-row causes and rerun.",
            )
        )
    if float(metrics.get("relative_norm_drift", 0.0) or 0.0) > 0.5:
        debts.append(
            _debt(
                "high_norm_drift",
                "serious",
                run,
                "Relative norm drift exceeds the default 0.5 threshold.",
                "Tune intervention magnitude or justify threshold.",
            )
        )
    if not any(row.get("claim_relevance") == "supporting" for row in manifest.get("artifacts", [])):
        debts.append(
            _debt(
                "missing_matched_controls",
                "warning",
                run,
                "No annotated supporting artifact is registered.",
                "Annotate reviewed supporting artifacts.",
            )
        )

    blockers = [debt for debt in debts if debt.severity == "blocking" and debt.status == "open"]
    warnings = [debt for debt in debts if debt.severity in {"info", "warning"}]
    clean_candidate_support = (
        not blockers
        and not any(debt.severity == "serious" and debt.status == "open" for debt in debts)
        and run_class in ALLOWED_CLEAN_RUN_CLASSES
        and any(row.get("claim_relevance") == "supporting" for row in manifest.get("artifacts", []))
    )
    report = ScientificDebtReport(
        report_id=f"SDR-{secrets.token_hex(4)}",
        run_id=run_id,
        experiment_id=run.get("experiment_id"),
        generated_at=utc_now(),
        evaluated_assessments=evaluated,
        debts=debts,
        blockers=blockers,
        warnings=warnings,
        threshold_sources=_threshold_sources(metrics),
        clean_candidate_support=clean_candidate_support,
        summary="clean candidate support"
        if clean_candidate_support
        else f"{len(blockers)} blockers, {len(debts)} total debt records",
    )
    write_json(run_dir / "scientific_debt_report.json", report.to_dict())
    (run_dir / "scientific_debt_report.md").write_text(_report_markdown(report), encoding="utf-8")
    return report


def calibration_check(project_root: str | Path, run_id_or_alias: str) -> ScientificDebtReport:
    return gate_check(project_root, run_id_or_alias)


def telemetry_check(project_root: str | Path, run_id_or_alias: str) -> ScientificDebtReport:
    return gate_check(project_root, run_id_or_alias)


def _resolve_run_id(project_root: Path, run_id_or_alias: str) -> str:
    if (runs_dir(project_root) / run_id_or_alias).exists():
        return run_id_or_alias
    return resolve_run_alias(alias_cache_path(project_root), run_id_or_alias).run_id


def _metric_values(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for row in read_jsonl(path):
        if "metric_name" in row:
            values[str(row["metric_name"])] = row.get("value")
    return values


def _debt(
    debt_type: str,
    severity: str,
    run: dict[str, Any],
    message: str,
    required_resolution: str | None,
) -> ScientificDebtRecord:
    return ScientificDebtRecord(
        debt_id=f"DEBT-{secrets.token_hex(4)}",
        debt_type=debt_type,
        severity=severity,
        claim_id=None,
        run_id=run.get("run_id"),
        experiment_id=run.get("experiment_id"),
        evidence_paths=[],
        message=message,
        required_resolution=required_resolution,
        status="open",
        waiver_decision_id=None,
        created_at=utc_now(),
    )


def _threshold_sources(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for metric_name in [
        "positive_control_pass_rate",
        "random_null_seed_count",
        "nonfinite_rate",
        "skip_rate",
        "relative_norm_drift",
    ]:
        if metric_name in metrics:
            sources.append(
                {
                    "assessment_id": "gate_check",
                    "condition_id": metric_name,
                    "metric_name": metric_name,
                    "threshold_source": "tool_default",
                    "threshold_justification": None,
                    "threshold_decision_id": None,
                }
            )
    return sources


def _report_markdown(report: ScientificDebtReport) -> str:
    rows = [f"# Scientific Debt Report: {report.run_id}", "", f"Summary: {report.summary}", ""]
    for debt in report.debts:
        rows.extend(
            [
                f"## {debt.debt_id} - {debt.debt_type}",
                "",
                f"Severity: {debt.severity}",
                "",
                debt.message,
                "",
            ]
        )
    return "\n".join(rows)
