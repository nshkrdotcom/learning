from __future__ import annotations

from pathlib import Path

from helpers_project import populate_project, runner, write_claim_ledger

from mechledger.cli import app


def test_appendix_export_includes_claim_debt_decision_run_and_is_deterministic(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    out = tmp_path / "research/paper/mechledger_appendix.md"

    first = runner.invoke(
        app,
        ["export", "appendix", "--out", str(out), "--include-debt", "--include-decisions"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert first.exit_code == 0, first.output
    first_text = out.read_text(encoding="utf-8")

    second = runner.invoke(
        app,
        ["export", "appendix", "--out", str(out), "--include-debt", "--include-decisions"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert second.exit_code == 0, second.output
    assert first_text == out.read_text(encoding="utf-8")
    assert "C001" in first_text
    assert "RUN_E001" in first_text
    assert "D001" in first_text
    assert "missing_empirical_null" in first_text
    assert "does not prove scientific truth" in first_text


def test_appendix_export_accepts_project_relative_out_path(tmp_path: Path) -> None:
    populate_project(tmp_path)
    out = tmp_path / "research/paper/relative_appendix.md"

    result = runner.invoke(
        app,
        ["export", "appendix", "--out", "research/paper/relative_appendix.md"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "appendix: research/paper/relative_appendix.md" in result.output
    assert out.exists()


def test_appendix_filters_and_preserves_negative_claim_status(tmp_path: Path) -> None:
    populate_project(tmp_path)
    write_claim_ledger(tmp_path, status="failed_or_weakened")
    out = tmp_path / "research/paper/filtered_appendix.md"

    result = runner.invoke(
        app,
        ["export", "appendix", "--out", str(out), "--claim", "C001", "--run", "RUN_E001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "failed_or_weakened" in text
    assert "not phrased as support" in text
