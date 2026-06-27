import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_session_dir(root: Path) -> Path:
    sessions = sorted((root / ".mechanism" / "sessions").glob("sess_*"))
    assert sessions
    return sessions[-1]


def test_ipython_execute_injects_ctx_and_captures_ctx_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            "obj = ctx.objects.create('analysis_note', metadata={'score': 1})",
        ],
    )

    assert result.exit_code == 0, result.output
    session_dir = latest_session_dir(tmp_path)
    session = json.loads((session_dir / "session.json").read_text())
    cells = jsonl(session_dir / "cells.jsonl")
    objects = jsonl(session_dir / "namespace_objects.jsonl")

    assert session["surface"] == "ipython"
    assert session["ended_at"] is not None
    assert cells[0]["status"] == "ok"
    assert cells[0]["created_object_refs"]
    assert objects[0]["event"] == "object_registered"
    assert objects[0]["variable_name"] == "obj"
    assert objects[0]["object_type"] == "analysis_note"


def test_ipython_execute_records_failed_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["ipython", "--execute", "raise RuntimeError('boom')"])

    assert result.exit_code == 1
    session_dir = latest_session_dir(tmp_path)
    cells = jsonl(session_dir / "cells.jsonl")
    exceptions = list((session_dir / "exceptions").glob("*.txt"))
    assert cells[0]["status"] == "error"
    assert cells[0]["exception_ref"]
    assert exceptions and "RuntimeError" in exceptions[0].read_text()


def test_ipython_execute_captures_direct_protocol_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            (
                "from mwb.domain.objects import WorkbenchObject\n"
                "direct = WorkbenchObject(wb_ref='obj_direct', wb_type='DirectObject')"
            ),
        ],
    )

    assert result.exit_code == 0, result.output
    session_dir = latest_session_dir(tmp_path)
    objects = jsonl(session_dir / "namespace_objects.jsonl")
    assert objects[0]["object_ref"] == "obj_direct"
    assert objects[0]["object_type"] == "DirectObject"


def test_ipython_execute_records_alias_and_deleted_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            "features = ctx.objects.create('FeatureRanking')",
            "--execute",
            "top_features = features",
            "--execute",
            "del features",
        ],
    )

    assert result.exit_code == 0, result.output
    session_dir = latest_session_dir(tmp_path)
    object_events = [record["event"] for record in jsonl(session_dir / "namespace_objects.jsonl")]
    assert object_events == ["object_registered", "alias_bound", "alias_deleted"]


def test_inspect_session_latest_reports_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    run_result = runner.invoke(app, ["ipython", "--execute", "obj = ctx.objects.create('Note')"])
    inspect_result = runner.invoke(app, ["inspect", "session", "latest"])

    assert run_result.exit_code == 0, run_result.output
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "session:" in inspect_result.output
    assert "surface: ipython" in inspect_result.output

