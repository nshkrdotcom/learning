from __future__ import annotations

from pathlib import Path


def test_matrix_csv_and_report_are_generated() -> None:
    matrix_path = Path("artifacts/scoring/construct_mismatch_matrix.csv")
    report_path = Path("reports/construct_mismatch_report.md")
    assert matrix_path.exists() and matrix_path.stat().st_size > 0
    assert report_path.exists() and report_path.stat().st_size > 0
