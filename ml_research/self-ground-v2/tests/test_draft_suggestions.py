from __future__ import annotations

from pathlib import Path

from helpers_project import populate_project, runner

from mechledger.cli import app


def test_draft_suggest_report_surfaces_forbidden_caveat_debt_and_unknown_claim(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    draft = tmp_path / "research/paper/draft.md"
    draft.write_text(
        "This proves causally that the feature works. [CLAIM:C001]\n\n"
        "Unknown claim mention. [CLAIM:C999]\n",
        encoding="utf-8",
    )
    out = tmp_path / "research/paper/draft_suggestions.md"

    result = runner.invoke(
        app,
        ["draft", "suggest", str(draft), "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "forbidden_language" in text
    assert "single-run evidence" in text
    assert "missing_empirical_null" in text
    assert "Unknown claim ID C999" in text
    assert "Safe Language Checklist" in text


def test_claim_language_report_single_and_all_are_deterministic(tmp_path: Path) -> None:
    populate_project(tmp_path)
    out = tmp_path / "research/paper/claim_language_report.md"

    single = runner.invoke(
        app,
        ["claim", "language-report", "--claim", "C001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    first_all = runner.invoke(
        app,
        ["claim", "language-report", "--all", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    first_text = out.read_text(encoding="utf-8")
    second_all = runner.invoke(
        app,
        ["claim", "language-report", "--all", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert single.exit_code == 0 and "C001" in single.output
    assert first_all.exit_code == 0, first_all.output
    assert second_all.exit_code == 0, second_all.output
    assert first_text == out.read_text(encoding="utf-8")
    assert "allowed phrases" in first_text
    assert "forbidden phrases" in first_text
