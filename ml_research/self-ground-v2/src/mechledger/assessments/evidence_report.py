from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mechledger.assessments.common import AssessmentConditionResult
from mechledger.core.debt import DebtStatus, ScientificDebtRecord
from mechledger.project import SCHEMA_VERSION


class EvidenceAssessmentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    generated_at: str
    run_class: str
    clean_candidate_support: bool
    recommended_claim_status: str
    conditions: dict[str, AssessmentConditionResult]
    debts: list[ScientificDebtRecord] = Field(default_factory=list)
    waivers_applied: list[str] = Field(default_factory=list)
    summary: str

    @property
    def open_debts(self) -> list[ScientificDebtRecord]:
        return [debt for debt in self.debts if debt.status == DebtStatus.OPEN]

    @property
    def waived_debts(self) -> list[ScientificDebtRecord]:
        return [debt for debt in self.debts if debt.status == DebtStatus.WAIVED]


def write_evidence_assessment(run_dir: Path, report: EvidenceAssessmentReport) -> None:
    payload = report.model_dump(mode="json")
    (run_dir / "evidence_assessment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "evidence_assessment.md").write_text(
        evidence_assessment_markdown(report), encoding="utf-8"
    )


def evidence_assessment_markdown(report: EvidenceAssessmentReport) -> str:
    lines = [
        f"# Evidence Assessment for {report.run_id}",
        "",
        f"- Run class: {report.run_class}",
        f"- Clean candidate support: {'yes' if report.clean_candidate_support else 'no'}",
        f"- Recommended claim status: {report.recommended_claim_status}",
        "",
        "## Conditions",
        "",
        "| Condition | Passed | Debt type | Severity | Threshold source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in report.conditions.values():
        severity = condition.severity.value if condition.severity else ""
        threshold_source = condition.threshold_source.value if condition.threshold_source else ""
        lines.append(
            f"| {condition.condition_id} | {condition.passed} | "
            f"{condition.debt_type or ''} | {severity} | {threshold_source} |"
        )
    lines.extend(["", "## Open Scientific Debt", ""])
    open_debts = report.open_debts
    if open_debts:
        for debt in open_debts:
            lines.append(
                f"- {debt.debt_id} [{debt.severity.value}] "
                f"{debt.debt_type}: {debt.message}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Waived Debt", ""])
    waived_debts = report.waived_debts
    if waived_debts:
        for debt in waived_debts:
            lines.append(
                f"- {debt.debt_id} [{debt.severity.value}/waived] "
                f"waiver={debt.waiver_decision_id} {debt.debt_type}: {debt.message}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Suggested Next Evidence",
            "",
            *_suggestions(report),
            "",
            "MechLedger assessed registered metadata/artifacts and reported metrics only. "
            "It did not execute model code or compute statistical tests.",
            "",
        ]
    )
    return "\n".join(lines)


def _suggestions(report: EvidenceAssessmentReport) -> list[str]:
    if report.clean_candidate_support:
        return ["- Review artifacts and claim prose before proposing promotion."]
    suggestions = []
    for debt in report.open_debts:
        if debt.required_resolution:
            suggestions.append(f"- {debt.required_resolution}")
    return suggestions or ["- Register missing evidence or document an accepted waiver decision."]
