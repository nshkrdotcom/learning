from __future__ import annotations

import json
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from helpers_project import create_run, populate_project, runner

from mechledger.cli import app


def _set_started(tmp_path: Path, run_id: str, started_at: str) -> None:
    run_json = tmp_path / ".mechledger/runs" / run_id / "run.json"
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    payload["started_at"] = started_at
    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_alias(tmp_path: Path, run_id: str, timestamp: str) -> None:
    with (tmp_path / ".mechledger/alias_cache.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{run_id}\t{timestamp}\tE001\t{run_id}\n")


def _add_run(tmp_path: Path, run_id: str, started_at: str) -> None:
    create_run(tmp_path, run_id=run_id)
    _set_started(tmp_path, run_id, started_at)
    _append_alias(tmp_path, run_id, started_at)


def test_pin_is_idempotent_and_appends_single_event(tmp_path: Path) -> None:
    populate_project(tmp_path)

    first = runner.invoke(app, ["pin", "latest"], env={"PWD": str(tmp_path)})
    second = runner.invoke(app, ["pin", "RUN_E001"], env={"PWD": str(tmp_path)})

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    run_json = json.loads(
        (tmp_path / ".mechledger/runs/RUN_E001/run.json").read_text(encoding="utf-8")
    )
    assert run_json["pinned"] is True
    events = [
        json.loads(line)
        for line in (tmp_path / ".mechledger/runs/RUN_E001/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [event["event_type"] for event in events].count("run_pinned") == 1


def test_gc_dry_run_preserves_runs_and_research_logs(tmp_path: Path) -> None:
    populate_project(tmp_path)
    _add_run(tmp_path, "RUN_OLD", "2026-06-24T00:00:00Z")
    before_log = (tmp_path / "research/logs/run_ledger.csv").read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        ["gc", "--keep-last", "1", "--keep-pinned"],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".mechledger/runs/RUN_OLD").exists()
    assert (tmp_path / "research/logs/run_ledger.csv").read_text(encoding="utf-8") == before_log
    manifest = json.loads((tmp_path / ".mechledger/gc_manifest.json").read_text())
    assert manifest["dry_run"] is True
    assert manifest["removed_run_ids"] == []
    assert manifest["planned_remove_run_ids"] == ["RUN_OLD"]


def test_gc_yes_removes_only_unpinned_eligible_runs(tmp_path: Path) -> None:
    populate_project(tmp_path)
    _add_run(tmp_path, "RUN_OLD", "2026-06-24T00:00:00Z")
    _add_run(tmp_path, "RUN_PINNED", "2026-06-23T00:00:00Z")
    runner.invoke(app, ["pin", "RUN_PINNED"], env={"PWD": str(tmp_path)})

    result = runner.invoke(
        app,
        ["gc", "--keep-last", "1", "--keep-pinned", "--yes"],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".mechledger/runs/RUN_OLD").exists()
    assert (tmp_path / ".mechledger/runs/RUN_PINNED").exists()
    assert (tmp_path / ".mechledger/runs/RUN_E001").exists()


def test_gc_archives_before_delete_and_failed_archive_preserves_run(tmp_path: Path) -> None:
    populate_project(tmp_path)
    _add_run(tmp_path, "RUN_OLD", "2026-06-24T00:00:00Z")
    archive_dir = tmp_path / "bundles"

    result = runner.invoke(
        app,
        ["gc", "--archive", str(archive_dir), "--keep-last", "1", "--yes"],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".mechledger/runs/RUN_OLD").exists()
    assert (archive_dir / "RUN_OLD.tar.gz").exists()
    manifest = json.loads((tmp_path / ".mechledger/gc_manifest.json").read_text())
    assert manifest["archived_run_ids"] == ["RUN_OLD"]

    _add_run(tmp_path, "RUN_FAIL", "2026-06-24T00:00:00Z")
    archive_file = tmp_path / "archive-file"
    archive_file.write_text("not a directory\n", encoding="utf-8")
    failed = runner.invoke(
        app,
        ["gc", "--archive", str(archive_file), "--keep-last", "1", "--yes"],
        env={"PWD": str(tmp_path)},
    )
    assert failed.exit_code == 2
    assert (tmp_path / ".mechledger/runs/RUN_FAIL").exists()


def test_per_run_bundle_manifest_hashes_every_included_file(tmp_path: Path) -> None:
    populate_project(tmp_path)
    out = tmp_path / "bundles/RUN_E001.tar.gz"

    result = runner.invoke(
        app,
        ["bundle", "RUN_E001", "--out", str(out)],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    with tarfile.open(out, "r:gz") as archive:
        names = set(archive.getnames())
        manifest_file = archive.extractfile("manifest.json")
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read().decode("utf-8"))
    assert ".mechledger/runs/RUN_E001/run.json" in names
    assert "artifacts/result.json" in names
    assert manifest["run_id"] == "RUN_E001"
    assert {entry["path"] for entry in manifest["files"]} == names - {"manifest.json"}
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_run_repair_marks_stale_running_run_without_promoting_evidence(tmp_path: Path) -> None:
    populate_project(tmp_path)
    run_dir = create_run(tmp_path, run_id="RUN_STALE")
    run_json = run_dir / "run.json"
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    payload.update({"status": "running", "finished_at": None, "exit_code": None})
    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "heartbeat.json").write_text(
        json.dumps({"last_heartbeat_at": "2026-06-25T00:00:00Z", "pid": 999}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["run", "repair", "RUN_STALE", "--status", "interrupted"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    repaired = json.loads(run_json.read_text(encoding="utf-8"))
    assert repaired["status"] == "interrupted"
    assert repaired["exit_code"] == 130
    assert not (run_dir / "heartbeat.json").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["event_type"] == "run_repaired" for event in events)
    proposal = json.loads((run_dir / "claim_update_proposal.json").read_text())
    assert proposal["proposed_status"] == "unsupported"


def test_run_resume_creates_child_run_with_parent_and_complete_contract(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "resume",
            "RUN_E001",
            "--class",
            "notebook_exploration",
            "--purpose",
            "continue notebook",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    child_id = next(
        line.split(":", 1)[1].strip()
        for line in result.output.splitlines()
        if line.startswith("Created child run:")
    )
    child_dir = tmp_path / ".mechledger/runs" / child_id
    run_json = json.loads((child_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["parent_run_id"] == "RUN_E001"
    assert run_json["status"] == "running"
    assert run_json["run_class"] == "notebook_exploration"
    expected_files = {
        "run.json",
        "heartbeat.json",
        "events.jsonl",
        "metrics.jsonl",
        "artifacts.jsonl",
        "artifact_manifest.json",
        "resource_usage.json",
        "stdout.txt",
        "stderr.txt",
        "command.txt",
        "environment.json",
        "git.json",
        "summary.json",
        "run_ledger_row.csv",
        "claim_update_proposal.md",
        "claim_update_proposal.json",
        "scientific_debt_report.md",
        "scientific_debt_report.json",
        "run_class_transition.json",
        "artifacts",
    }
    assert expected_files <= {path.name for path in child_dir.iterdir()}
    parent = json.loads((tmp_path / ".mechledger/runs/RUN_E001/run.json").read_text())
    assert parent.get("parent_run_id") is None


def test_index_marks_only_stale_running_heartbeat_as_interrupted_indexed(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    fresh_dir = create_run(tmp_path, run_id="RUN_FRESH")
    stale_dir = create_run(tmp_path, run_id="RUN_STALE")
    for run_dir, timestamp in [
        (fresh_dir, datetime.now(UTC).isoformat().replace("+00:00", "Z")),
        (stale_dir, "2026-06-25T00:00:00Z"),
    ]:
        run_json = run_dir / "run.json"
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        payload.update({"status": "running", "finished_at": None, "exit_code": None})
        run_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (run_dir / "heartbeat.json").write_text(
            json.dumps({"last_heartbeat_at": timestamp, "pid": 999}) + "\n",
            encoding="utf-8",
        )

    result = runner.invoke(app, ["index"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert result.exit_code == 0, result.output
    with sqlite3.connect(tmp_path / ".mechledger/index.sqlite") as conn:
        rows = dict(conn.execute("SELECT run_id, indexed_status FROM local_runs"))

    assert rows["RUN_FRESH"] == "running"
    assert rows["RUN_STALE"] == "interrupted_indexed"
    stale_payload = json.loads((stale_dir / "run.json").read_text(encoding="utf-8"))
    assert stale_payload["status"] == "running"


def test_gc_refuses_keep_last_zero_without_explicit_override(tmp_path: Path) -> None:
    populate_project(tmp_path)

    result = runner.invoke(
        app,
        ["gc", "--keep-last", "0", "--yes"],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "allow-remove-all-unpinned" in result.output
