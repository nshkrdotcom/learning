import json
import sqlite3
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_session_dir(root: Path) -> Path:
    sessions = sorted((root / ".mechanism" / "sessions").glob("sess_*"))
    assert sessions
    return sessions[-1]


def test_ipython_execute_captures_bounded_stdout_and_stderr(
    tmp_path: Path, monkeypatch
) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ipython",
            "--execute",
            "import sys\nprint('captured stdout')\nprint('captured stderr', file=sys.stderr)",
        ],
    )

    assert result.exit_code == 0, result.output
    session_dir = latest_session_dir(tmp_path)
    cell = jsonl(session_dir / "cells.jsonl")[0]
    assert cell["stdout_ref"]
    assert cell["stderr_ref"]
    assert "captured stdout" in (session_dir / "stdout" / "cell_000001.txt").read_text()
    assert "captured stderr" in (session_dir / "stderr" / "cell_000001.txt").read_text()


def test_repair_index_alias_rebuilds_sqlite_from_files(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    run = runner.invoke(app, ["ipython", "--execute", "note = ctx.note('repair')"])
    repaired = tmp_path / "repair.sqlite"

    result = runner.invoke(app, ["repair-index", "--output", str(repaired)])

    assert run.exit_code == 0, run.output
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["counts"]["objects"] == 1
    assert repaired.exists()


def test_ipython_capture_indexes_cell_and_parent_lineage_edges(
    tmp_path: Path, monkeypatch
) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ipython",
            "--execute",
            (
                "parent = ctx.objects.create('Parent')\n"
                "child = ctx.objects.create('Child', parents=[parent.wb_ref])"
            ),
        ],
    )

    assert result.exit_code == 0, result.output
    with sqlite3.connect(project.sqlite_path) as conn:
        rows = [
            json.loads(row[0])
            for row in conn.execute("select payload_json from lineage_edges").fetchall()
        ]
    relations = {(row["src_ref"], row["dst_ref"], row["relation"]) for row in rows}
    parent_ref = next(row["src_ref"] for row in rows if row["relation"] == "parent")
    child_ref = next(row["dst_ref"] for row in rows if row["relation"] == "parent")

    assert (parent_ref, child_ref, "parent") in relations
    assert ("cell_000001", parent_ref, "created_in_cell") in relations
    assert ("cell_000001", child_ref, "created_in_cell") in relations
