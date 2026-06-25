from __future__ import annotations

from mechledger.core.debt import DebtSeverity, ScientificDebtRecord, ScientificDebtReport


def _debt(debt_id: str, severity: DebtSeverity, status: str = "open") -> ScientificDebtRecord:
    return ScientificDebtRecord(
        debt_id=debt_id,
        debt_type="missing_empirical_null",
        severity=severity,
        claim_id="C001",
        run_id="runs/example",
        experiment_id="E001",
        evidence_paths=[],
        message="example",
        required_resolution=None,
        status=status,
        waiver_decision_id=None,
        created_at="2026-06-25T00:00:00Z",
    )


def test_report_blockers_and_warnings_are_filtered_views() -> None:
    blocking = _debt("D1", DebtSeverity.BLOCKING)
    serious = _debt("D2", DebtSeverity.SERIOUS)
    waived = _debt("D3", DebtSeverity.BLOCKING, status="waived")
    report = ScientificDebtReport(
        report_id="R1",
        run_id="runs/example",
        experiment_id="E001",
        generated_at="2026-06-25T00:00:00Z",
        evaluated_assessments=["telemetry"],
        debts=[blocking, serious, waived],
        threshold_sources=[],
        clean_candidate_support=False,
        summary="",
    )

    assert report.blockers == [blocking]
    assert report.warnings == [serious]
    assert report.debts[0] is blocking


def test_tool_default_debt_summary_rolls_up_noise() -> None:
    debts = [
        ScientificDebtRecord(
            debt_id=f"D{idx}",
            debt_type="unjustified_threshold_default",
            severity=DebtSeverity.INFO,
            claim_id=None,
            run_id="runs/example",
            experiment_id="E001",
            evidence_paths=[],
            message="tool default",
            required_resolution=None,
            status="open",
            waiver_decision_id=None,
            created_at="2026-06-25T00:00:00Z",
            assessment_id=assessment,
        )
        for idx, assessment in enumerate(["telemetry", "telemetry", "positive_control"], start=1)
    ]
    report = ScientificDebtReport(
        report_id="R1",
        run_id="runs/example",
        experiment_id="E001",
        generated_at="2026-06-25T00:00:00Z",
        evaluated_assessments=["telemetry", "positive_control"],
        debts=debts,
        threshold_sources=[],
        clean_candidate_support=False,
        summary="",
    )

    assert report.tool_default_rollup() == {
        "count": 3,
        "assessment_ids": ["positive_control", "telemetry"],
    }
