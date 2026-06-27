import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mwb.cli import app
from mwb.git_state import capture_git_state
from mwb.project import ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_project_init_creates_layout_and_is_idempotent(tmp_path: Path) -> None:
    init_git_repo(tmp_path)

    project = ProjectManager.init(tmp_path, name="self-ground")
    second = ProjectManager.init(tmp_path, name="self-ground")

    assert project.root == second.root == tmp_path
    mechanism = tmp_path / ".mechanism"
    for relative in [
        "project.toml",
        "workbench.sqlite",
        "events.jsonl",
        "sessions",
        "runs",
        "artifacts",
        "cards",
        "claims",
        "hypotheses",
        "exports",
        "cache",
        "adapters",
        "logs",
        "redactions",
    ]:
        assert (mechanism / relative).exists(), relative

    project_toml = (mechanism / "project.toml").read_text()
    assert 'name = "self-ground"' in project_toml
    assert 'schema_version = 1' in project_toml


def test_project_init_appends_project_created_event_once(tmp_path: Path) -> None:
    init_git_repo(tmp_path)

    ProjectManager.init(tmp_path, name="self-ground")
    ProjectManager.init(tmp_path, name="self-ground")

    events = [
        json.loads(line)
        for line in (tmp_path / ".mechanism" / "events.jsonl").read_text().splitlines()
    ]
    assert [event["event_type"] for event in events] == ["project_created"]
    assert events[0]["payload"]["project_name"] == "self-ground"


def test_sqlite_schema_is_initialized(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")

    sqlite_path = tmp_path / ".mechanism" / "workbench.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type='table' order by name"
            )
        }

    assert "projects" in tables
    assert "events" in tables
    assert "artifacts" in tables


def test_git_state_captures_dirty_status_without_commit(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    (tmp_path / "dirty.txt").write_text("changed\n")

    state = capture_git_state(tmp_path)

    assert state["branch"] == "main"
    assert state["commit"]
    assert state["dirty"] is True
    assert state["untracked"]
    assert state["diff_hash"].startswith("sha256:")


def test_cli_init_and_doctor_validate_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    init_result = runner.invoke(app, ["init", "--name", "self-ground"])
    doctor_result = runner.invoke(app, ["doctor"])

    assert init_result.exit_code == 0, init_result.output
    assert doctor_result.exit_code == 0, doctor_result.output
    assert "project: self-ground" in doctor_result.output
    assert "status: ok" in doctor_result.output
