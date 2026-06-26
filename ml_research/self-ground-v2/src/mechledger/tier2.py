from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mechledger.assessments.calibration import (
    evaluate_baseline_calibration,
    evaluate_positive_control,
)
from mechledger.assessments.candidate import (
    _debts_from_conditions,
    _flatten_conditions,
    _load_artifacts,
    _load_metrics,
    _load_valid_waivers,
)
from mechledger.assessments.common import AssessmentConditionResult, AssessmentReport
from mechledger.assessments.empirical_null import evaluate_empirical_null
from mechledger.assessments.paired_statistic import evaluate_paired_statistic
from mechledger.assessments.telemetry import evaluate_telemetry
from mechledger.core.debt import DebtSeverity, DebtStatus, ScientificDebtRecord
from mechledger.project import Project, now_utc

FilterMode = Literal["calibration", "telemetry", "empirical_null", "paired_statistic"]


class Tier2CheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    generated_at: str
    assessment_ids: list[str]
    conditions: dict[str, AssessmentConditionResult]
    debts: list[ScientificDebtRecord] = Field(default_factory=list)
    summary: str
    clean: bool

    @property
    def open_debts(self) -> list[ScientificDebtRecord]:
        return [debt for debt in self.debts if debt.status == DebtStatus.OPEN]

    @property
    def blocking_debts(self) -> list[ScientificDebtRecord]:
        return [
            debt
            for debt in self.open_debts
            if debt.severity == DebtSeverity.BLOCKING
        ]

    @property
    def serious_or_blocking_debts(self) -> list[ScientificDebtRecord]:
        return [
            debt
            for debt in self.open_debts
            if debt.severity in {DebtSeverity.SERIOUS, DebtSeverity.BLOCKING}
        ]


class PairedTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    paired_by: str
    metric: str
    test: Literal["sign", "wilcoxon", "permutation", "custom_registered"]
    n_pairs: int
    p_value: float | None
    effect_direction: Literal["positive", "negative", "mixed", "unknown"]
    sign_consistency: float | None = None
    threshold_source: Literal["tool_default", "project_default", "decision_justified"] | None
    threshold_justification: str | None = None
    threshold_decision_id: str | None = None
    input_artifact_path: str
    output_artifact_path: str

    @field_validator("n_pairs")
    @classmethod
    def _positive_pairs(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_pairs must be > 0")
        return value

    @field_validator("p_value", "sign_consistency")
    @classmethod
    def _probability(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("probability fields must be between 0 and 1")
        return value


def evaluate_filtered_report(
    project: Project,
    run_id: str,
    mode: FilterMode,
) -> Tier2CheckReport:
    run_dir = project.runs_dir / run_id
    run_data = _read_required_json(run_dir / "run.json")
    metrics = _load_metrics(run_dir / "metrics.jsonl")
    artifacts = _load_artifacts(run_dir)
    reports = _reports_for_mode(mode, metrics, artifacts)
    conditions = _flatten_conditions(reports)
    prior_waivers = _load_valid_waivers(
        run_dir / "scientific_debt_report.json",
        project.root / "research/logs/decision_log.md",
    )
    debts = _debts_from_conditions(
        run_id=run_id,
        experiment_id=run_data.get("experiment_id"),
        reports=reports,
        prior_waivers=prior_waivers,
    )
    clean = not [
        debt
        for debt in debts
        if debt.status == DebtStatus.OPEN
        and debt.severity in {DebtSeverity.SERIOUS, DebtSeverity.BLOCKING}
    ]
    summary = _summary(mode, conditions, debts)
    return Tier2CheckReport(
        run_id=run_id,
        generated_at=now_utc(),
        assessment_ids=[report.assessment_id for report in reports],
        conditions=conditions,
        debts=sorted(debts, key=lambda debt: debt.debt_id),
        summary=summary,
        clean=clean,
    )


def write_tier2_check_report(run_dir: Path, name: str, report: Tier2CheckReport) -> None:
    payload = report.model_dump(mode="json")
    (run_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / f"{name}.md").write_text(
        tier2_check_markdown(name, report), encoding="utf-8"
    )


def tier2_check_markdown(name: str, report: Tier2CheckReport) -> str:
    title = name.replace("_", " ").title()
    lines = [
        f"# {title} for {report.run_id}",
        "",
        report.summary,
        "",
        "## Conditions",
        "",
        "| condition | passed | severity | threshold_source | debt_type |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in report.conditions.values():
        severity = condition.severity.value if condition.severity else ""
        source = condition.threshold_source.value if condition.threshold_source else ""
        lines.append(
            f"| {condition.condition_id} | {condition.passed} | {severity} | "
            f"{source} | {condition.debt_type or ''} |"
        )
    lines.extend(["", "## Debt And Resolution", ""])
    if report.debts:
        for debt in report.debts:
            status = str(debt.status)
            lines.append(
                f"- {debt.debt_id} [{debt.severity.value}/{status}] "
                f"{debt.debt_type}: {debt.message}"
            )
            if debt.required_resolution:
                lines.append(f"  resolution: {debt.required_resolution}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "MechLedger assessed registered metadata/artifacts and reported metrics only. "
            "It did not run model code or compute external statistics.",
            "",
        ]
    )
    return "\n".join(lines)


def append_metric(run_dir: Path, metric_name: str, value: object) -> None:
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        payload = {"metric_name": metric_name, "value": value}
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_paired_test_result(path: Path, *, run_id: str) -> PairedTestResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = PairedTestResult.model_validate(payload)
    if result.run_id is not None and result.run_id != run_id:
        raise ValueError(
            f"Registered paired-test run_id `{result.run_id}` does not match `{run_id}`."
        )
    if result.run_id is None:
        result = result.model_copy(update={"run_id": run_id})
    return result


def paired_test_markdown(result: PairedTestResult, report: Tier2CheckReport) -> str:
    payload = result.model_dump(mode="json")
    lines = [
        f"# Paired Test for {result.run_id}",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key in [
        "paired_by",
        "metric",
        "test",
        "n_pairs",
        "p_value",
        "effect_direction",
        "sign_consistency",
        "threshold_source",
        "threshold_justification",
        "threshold_decision_id",
        "input_artifact_path",
        "output_artifact_path",
    ]:
        lines.append(f"| {key} | {payload.get(key)} |")
    lines.extend(["", "## Policy Assessment", ""])
    lines.append(report.summary)
    lines.extend(["", "## Conditions", ""])
    for condition in report.conditions.values():
        source = condition.threshold_source.value if condition.threshold_source else ""
        lines.append(
            f"- {condition.condition_id}: passed={condition.passed} "
            f"threshold_source={source}"
        )
    lines.extend(["", "## Debt And Resolution", ""])
    if report.debts:
        for debt in report.debts:
            lines.append(
                f"- {debt.debt_id} [{debt.severity.value}/{debt.status}] "
                f"{debt.debt_type}: {debt.message}"
            )
            if debt.required_resolution:
                lines.append(f"  resolution: {debt.required_resolution}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def has_registered_artifact_path(project: Project, run_id: str, path: Path) -> bool:
    run_dir = project.runs_dir / run_id
    manifest = _read_required_json(run_dir / "artifact_manifest.json")
    resolved = path.resolve()
    for artifact in manifest.get("artifacts", []):
        raw = artifact.get("resolved_path") or artifact.get("original_path")
        if raw and Path(str(raw)).resolve() == resolved:
            return True
    return False


def _reports_for_mode(
    mode: FilterMode,
    metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> list[AssessmentReport]:
    if mode == "calibration":
        return [
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
        ]
    if mode == "telemetry":
        return [evaluate_telemetry(metrics)]
    if mode == "empirical_null":
        return [evaluate_empirical_null(metrics, artifacts=artifacts, percentile_threshold=0.95)]
    if mode == "paired_statistic":
        return [evaluate_paired_statistic(metrics, required=True)]
    raise ValueError(f"Unknown Tier 2 check mode: {mode}")


def _summary(
    mode: FilterMode,
    conditions: dict[str, AssessmentConditionResult],
    debts: list[ScientificDebtRecord],
) -> str:
    failed = [condition.condition_id for condition in conditions.values() if not condition.passed]
    blockers = [
        debt
        for debt in debts
        if debt.status == DebtStatus.OPEN and debt.severity == DebtSeverity.BLOCKING
    ]
    serious = [
        debt
        for debt in debts
        if debt.status == DebtStatus.OPEN and debt.severity == DebtSeverity.SERIOUS
    ]
    return (
        f"assessment: {mode}; failed_conditions={len(failed)}; "
        f"blocking_debt={len(blockers)}; serious_debt={len(serious)}"
    )


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required run file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload
