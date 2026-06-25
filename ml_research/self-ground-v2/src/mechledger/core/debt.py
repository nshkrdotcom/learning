from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DebtSeverity(StrEnum):
    # Debt severity is about evidence limitations; it includes `serious`.
    INFO = "info"
    WARNING = "warning"
    SERIOUS = "serious"
    BLOCKING = "blocking"


class DebtStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    WAIVED = "waived"
    SUPERSEDED = "superseded"


class DebtType(StrEnum):
    MISSING_POSITIVE_CONTROL = "missing_positive_control"
    FAILED_POSITIVE_CONTROL = "failed_positive_control"
    MISSING_BASELINE_CALIBRATION = "missing_baseline_calibration"
    MISSING_EMPIRICAL_NULL = "missing_empirical_null"
    INSUFFICIENT_NULL_SEEDS = "insufficient_null_seeds"
    MISSING_PAIRED_STATISTIC = "missing_paired_statistic"
    MISSING_MATCHED_CONTROLS = "missing_matched_controls"
    HIGH_NORM_DRIFT = "high_norm_drift"
    NONFINITE_ROWS = "nonfinite_rows"
    ALL_ROWS_SKIPPED = "all_rows_skipped"
    METADATA_MISMATCH_OVERRIDE = "metadata_mismatch_override"
    SINGLETON_SEED = "singleton_seed"
    DIAGNOSTIC_RUN_ONLY = "diagnostic_run_only"
    SMOKE_TEST_ONLY = "smoke_test_only"
    UNREVIEWED_NOTEBOOK_RUN = "unreviewed_notebook_run"
    REDACTED_SUPPORTING_EVIDENCE = "redacted_supporting_evidence"
    UNJUSTIFIED_THRESHOLD_DEFAULT = "unjustified_threshold_default"
    CUSTOM = "custom"


class ScientificDebtRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    debt_id: str
    debt_type: DebtType | str
    severity: DebtSeverity
    claim_id: str | None
    run_id: str | None
    experiment_id: str | None
    evidence_paths: list[str]
    message: str
    required_resolution: str | None
    status: DebtStatus | str
    waiver_decision_id: str | None
    created_at: str
    resolved_at: str | None = None
    assessment_id: str | None = None


class ScientificDebtReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    run_id: str
    experiment_id: str | None
    generated_at: str
    evaluated_assessments: list[str]
    debts: list[ScientificDebtRecord]
    threshold_sources: list[dict[str, Any]]
    clean_candidate_support: bool
    summary: str = Field(default="")

    @property
    def blockers(self) -> list[ScientificDebtRecord]:
        return [
            debt
            for debt in self.debts
            if debt.severity == DebtSeverity.BLOCKING and debt.status == DebtStatus.OPEN
        ]

    @property
    def warnings(self) -> list[ScientificDebtRecord]:
        return [
            debt
            for debt in self.debts
            if debt.severity in {DebtSeverity.INFO, DebtSeverity.WARNING, DebtSeverity.SERIOUS}
            and debt.status == DebtStatus.OPEN
        ]

    def tool_default_rollup(self) -> dict[str, Any]:
        default_debts = [
            debt
            for debt in self.debts
            if debt.debt_type == DebtType.UNJUSTIFIED_THRESHOLD_DEFAULT
            and debt.status == DebtStatus.OPEN
        ]
        return {
            "count": len(default_debts),
            "assessment_ids": sorted(
                key for key in Counter(debt.assessment_id for debt in default_debts) if key
            ),
        }
