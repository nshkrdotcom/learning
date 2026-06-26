from __future__ import annotations

import json
from pathlib import Path

from helpers_project import create_run, populate_project, runner, write_claim_ledger

from mechledger.cli import app


def test_sync_status_clean_project_exits_zero(tmp_path: Path) -> None:
    populate_project(tmp_path)

    result = runner.invoke(app, ["sync", "status"], env={"PWD": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "blocking_findings: 0" in result.output
    assert "local_run_missing_from_ledger: 0" in result.output


def test_sync_diff_reports_run_and_claim_drift(tmp_path: Path) -> None:
    populate_project(tmp_path)
    create_run(tmp_path, run_id="RUN_LOCAL_ONLY")
    (tmp_path / ".mechledger/runs/RUN_LEDGER_ONLY").mkdir(parents=True)
    ledger = tmp_path / "research/logs/run_ledger.csv"
    ledger.write_text(
        ledger.read_text(encoding="utf-8")
        + "2026-06-25,RUN_LEDGER_ONLY,abc,phase,purpose,hypothesis,cmd,pythia,hook,,,,,"
        + ",,,baseline,ablate,completed,,specificity_gap=0.3,,artifact_manifest.json,D001\n",
        encoding="utf-8",
    )
    write_claim_ledger(tmp_path)
    claim_ledger = tmp_path / "research/logs/claim_ledger.md"
    claim_ledger.write_text(
        claim_ledger.read_text(encoding="utf-8").replace(
            "linked_runs: [RUN_E001]", "linked_runs: [RUN_MISSING]"
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["sync", "diff"], env={"PWD": str(tmp_path)})

    assert result.exit_code == 1, result.output
    assert "local_run_missing_from_ledger" in result.output
    assert "RUN_LOCAL_ONLY" in result.output
    assert "ledger_run_missing_locally" in result.output
    assert "RUN_LEDGER_ONLY" in result.output
    assert "claim_run_missing_from_ledger" in result.output
    assert "RUN_MISSING" in result.output


def test_sync_detects_alias_cache_conflicts_and_malformed_lines(tmp_path: Path) -> None:
    populate_project(tmp_path)
    alias_cache = tmp_path / ".mechledger/alias_cache.txt"
    alias_cache.write_text(
        "RUN_E001\t2026-06-25T00:00:00Z\tE001\tslug\n"
        "RUN_E001\t2026-06-25T00:00:01Z\tE001\tduplicate\n"
        "RUN_ABSENT\t2026-06-25T00:00:02Z\tE001\tmissing\n"
        "malformed partial line\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["sync", "diff"], env={"PWD": str(tmp_path)})

    assert result.exit_code == 1, result.output
    assert "duplicate_run_id" in result.output
    assert "alias_points_to_absent_run" in result.output
    assert "malformed_alias_cache_line" in result.output


def test_sync_rejects_alias_references_in_claim_ledger(tmp_path: Path) -> None:
    populate_project(tmp_path)
    claim_ledger = tmp_path / "research/logs/claim_ledger.md"
    claim_ledger.write_text(
        claim_ledger.read_text(encoding="utf-8").replace(
            "linked_runs: [RUN_E001]", "linked_runs: [latest, '#1']"
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["sync", "diff"], env={"PWD": str(tmp_path)})

    assert result.exit_code == 1, result.output
    assert "claim_uses_run_alias" in result.output
    assert "latest" in result.output
    assert "#1" in result.output


def test_sync_detects_duplicate_local_run_id_hash_mismatch(tmp_path: Path) -> None:
    populate_project(tmp_path)
    duplicate_dir = tmp_path / ".mechledger/runs/RUN_E001_COPY"
    duplicate_dir.mkdir(parents=True)
    duplicate_payload = {
        "run_id": "RUN_E001",
        "experiment_id": "E001",
        "status": "completed",
        "started_at": "2026-06-25T00:00:02Z",
    }
    (duplicate_dir / "run.json").write_text(
        json.dumps(duplicate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["sync", "diff"], env={"PWD": str(tmp_path)})

    assert result.exit_code == 1, result.output
    assert "duplicate_run_id" in result.output
    assert "run_json_hash_mismatch" in result.output
    assert "RUN_E001_COPY/run.json" in result.output
