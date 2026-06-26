from __future__ import annotations

import json
from pathlib import Path

from helpers_project import populate_project, runner

from mechledger.cli import app


def test_redact_run_writes_run_local_redaction_record(tmp_path: Path) -> None:
    populate_project(tmp_path)

    result = runner.invoke(
        app,
        ["redact", "RUN_E001", "--reason", "contains private subject notes"],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    record_path = tmp_path / ".mechledger/runs/RUN_E001/redaction_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["target_type"] == "run"
    assert record["target_path"] == ".mechledger/runs/RUN_E001/"
    assert record["reason"] == "contains private subject notes"
    assert record["original_hash"]
    assert record["placeholder_path"] is None


def test_redact_registered_supporting_artifact_updates_manifest_stub_and_debt(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    artifact = tmp_path / "artifacts/result.json"

    result = runner.invoke(
        app,
        [
            "redact",
            "artifact",
            "artifacts/result.json",
            "--run",
            "RUN_E001",
            "--reason",
            "contains private labels",
        ],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert not artifact.exists()
    stub = tmp_path / "artifacts/result.json.redacted"
    assert stub.exists()
    stub_text = stub.read_text(encoding="utf-8")
    assert "redaction_id:" in stub_text
    assert "redaction_record:" in stub_text
    record = json.loads(
        (tmp_path / ".mechledger/runs/RUN_E001/redaction_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["target_type"] == "artifact"
    assert record["original_hash"]
    assert record["redacted_hash"]
    assert record["placeholder_path"] == "artifacts/result.json.redacted"
    manifest = json.loads(
        (tmp_path / ".mechledger/runs/RUN_E001/artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    artifact_record = manifest["artifacts"][0]
    assert artifact_record["redaction_status"] == "redacted"
    assert artifact_record["review_status"] == "redacted"
    assert artifact_record["placeholder_path"] == "artifacts/result.json.redacted"
    debt_report = json.loads(
        (tmp_path / ".mechledger/runs/RUN_E001/scientific_debt_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(
        debt["debt_type"] == "redacted_supporting_evidence"
        for debt in debt_report["debts"]
    )


def test_redact_unregistered_artifact_is_safe_and_explicit(tmp_path: Path) -> None:
    populate_project(tmp_path)
    path = tmp_path / "artifacts/unregistered.json"
    path.write_text('{"private": true}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "redact",
            "artifact",
            "artifacts/unregistered.json",
            "--reason",
            "not registered",
        ],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "not registered" in result.output
    assert path.exists()
    assert not (tmp_path / "artifacts/unregistered.json.redacted").exists()


def test_redact_out_of_project_path_fails_without_mutation(tmp_path: Path) -> None:
    populate_project(tmp_path)
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "redact",
            "artifact",
            str(outside),
            "--run",
            "RUN_E001",
            "--reason",
            "outside",
        ],
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "outside project" in result.output
    assert outside.exists()


def test_redact_artifact_repeated_is_idempotent_when_record_valid(tmp_path: Path) -> None:
    populate_project(tmp_path)
    command = [
        "redact",
        "artifact",
        "artifacts/result.json",
        "--run",
        "RUN_E001",
        "--reason",
        "contains private labels",
    ]
    first = runner.invoke(app, command, env={"PWD": str(tmp_path)})
    second = runner.invoke(app, command, env={"PWD": str(tmp_path)})

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already_redacted" in second.output
