from __future__ import annotations

import json
from pathlib import Path

from helpers_project import init_project, runner, write_decision_log

from mechledger.cli import app


def _session_id(output: str) -> str:
    return next(
        line.split(":", 1)[1].strip()
        for line in output.splitlines()
        if line.startswith("session_id:")
    )


def test_session_start_note_attach_close_list_show_and_canonical_files_unchanged(
    tmp_path: Path,
) -> None:
    init_project(tmp_path)
    before = (tmp_path / "research/logs/research_log.md").read_text(encoding="utf-8")
    start = runner.invoke(
        app,
        ["session", "start", "--title", "Interpret feature labels"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert start.exit_code == 0, start.output
    session_id = _session_id(start.output)

    note1 = runner.invoke(
        app,
        ["session", "note", "--session", session_id, "--text", "first note"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    note2 = runner.invoke(
        app,
        ["session", "note", "--session", session_id, "--text", "second note"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert note1.exit_code == 0 and note2.exit_code == 0

    attachment = tmp_path / "notes.txt"
    attachment.write_text("session attachment\n", encoding="utf-8")
    attached = runner.invoke(
        app,
        ["session", "attach", "--session", session_id, str(attachment)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert attached.exit_code == 0, attached.output

    closed = runner.invoke(
        app,
        ["session", "close", "--session", session_id],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert closed.exit_code == 0, closed.output
    session_path = tmp_path / ".mechledger/copilot" / session_id / "session.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    assert [note["text"] for note in payload["notes"]] == ["first note", "second note"]
    assert payload["attached_paths"][0]["sha256"]
    assert (session_path.parent / "summary.md").exists()
    assert (tmp_path / "research/logs/research_log.md").read_text(encoding="utf-8") == before

    listed = runner.invoke(
        app,
        ["session", "list"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    shown = runner.invoke(
        app,
        ["session", "show", session_id],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert listed.exit_code == 0 and session_id in listed.output
    assert shown.exit_code == 0 and "Interpret feature labels" in shown.output


def test_session_review_requires_accepted_decision_and_can_reject(tmp_path: Path) -> None:
    init_project(tmp_path)
    write_decision_log(tmp_path, status="proposed")
    start = runner.invoke(
        app,
        ["session", "start", "--title", "Review draft"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    session_id = _session_id(start.output)
    runner.invoke(
        app,
        ["session", "close", "--session", session_id],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    rejected_decision = runner.invoke(
        app,
        ["session", "review", "--session", session_id, "--accept", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert rejected_decision.exit_code == 2
    assert "accepted decision" in rejected_decision.output

    write_decision_log(tmp_path, status="accepted")
    accepted = runner.invoke(
        app,
        ["session", "review", "--session", session_id, "--accept", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert accepted.exit_code == 0, accepted.output
    payload = json.loads(
        (tmp_path / ".mechledger/copilot" / session_id / "session.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "accepted"
    assert payload["review_decision_id"] == "D001"

    start2 = runner.invoke(
        app,
        ["session", "start", "--title", "Rejectable"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    session_id2 = _session_id(start2.output)
    rejected = runner.invoke(
        app,
        ["session", "review", "--session", session_id2, "--reject", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert rejected.exit_code == 0, rejected.output
