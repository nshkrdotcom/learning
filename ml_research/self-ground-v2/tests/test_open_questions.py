from __future__ import annotations

from pathlib import Path

from helpers_project import populate_project, runner, write_decision_log

from mechledger.cli import app


def test_questions_list_add_show_and_next_surface_open_questions(tmp_path: Path) -> None:
    populate_project(tmp_path)

    listed = runner.invoke(
        app,
        ["questions", "list"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert listed.exit_code == 0, listed.output
    assert "Does the feature survive density-matched controls?" in listed.output

    added = runner.invoke(
        app,
        [
            "questions",
            "add",
            "--text",
            "Should C001 require another control family?",
            "--claim",
            "C001",
            "--experiment",
            "E001",
            "--priority",
            "high",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert added.exit_code == 0, added.output
    question_id = next(
        line.split(":", 1)[1].strip()
        for line in added.output.splitlines()
        if line.startswith("question_id:")
    )

    shown = runner.invoke(
        app,
        ["questions", "show", question_id],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert shown.exit_code == 0
    assert "Should C001 require" in shown.output

    next_result = runner.invoke(app, ["next"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert next_result.exit_code in {0, 1}
    assert "OPEN QUESTIONS" in next_result.output
    assert question_id in next_result.output


def test_questions_resolve_requires_accepted_decision(tmp_path: Path) -> None:
    populate_project(tmp_path)
    write_decision_log(tmp_path, status="proposed")
    added = runner.invoke(
        app,
        ["questions", "add", "--text", "Resolve with decision?", "--claim", "C001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    question_id = next(
        line.split(":", 1)[1].strip()
        for line in added.output.splitlines()
        if line.startswith("question_id:")
    )

    bad = runner.invoke(
        app,
        [
            "questions",
            "resolve",
            question_id,
            "--decision",
            "D001",
            "--resolution",
            "accepted by review",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert bad.exit_code == 2
    assert "accepted decision" in bad.output

    write_decision_log(tmp_path, status="accepted")
    good = runner.invoke(
        app,
        [
            "questions",
            "resolve",
            question_id,
            "--decision",
            "D001",
            "--resolution",
            "accepted by review",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert good.exit_code == 0, good.output
    listed = runner.invoke(
        app,
        ["questions", "list"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert "resolved" in listed.output
