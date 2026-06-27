from __future__ import annotations

from pathlib import Path

from local_mi_lab.attention import ATTENTION_LIMITATION
from local_mi_lab.report import MANDATORY_LANGUAGE, generate_run_summary


def test_summary_generation_from_fixture_artifacts() -> None:
    fixture = Path(__file__).parent / "fixtures" / "small_run"
    summary = generate_run_summary(fixture)
    assert "# Run Summary" in summary
    assert "Examples: 2" in summary
    assert "Selected activation cache present" in summary
    assert "## Attention patterns" in summary
    assert "Top induction-like attention pattern candidates" in summary
    assert "Patching results present" in summary


def test_missing_artifact_handling(tmp_path: Path) -> None:
    summary = generate_run_summary(tmp_path)
    assert "Missing: `baseline_metrics.json` was not found." in summary
    assert "Missing: activation manifest was not found." in summary
    assert "Missing: `attention_summary.json` was not found." in summary
    assert "Missing: `patching_results.csv` was not found." in summary


def test_summary_has_no_mechanism_overclaiming() -> None:
    fixture = Path(__file__).parent / "fixtures" / "small_run"
    summary = generate_run_summary(fixture)
    for sentence in MANDATORY_LANGUAGE:
        assert sentence in summary
    assert ATTENTION_LIMITATION in summary
    assert "proves a mechanism" not in summary.lower()
