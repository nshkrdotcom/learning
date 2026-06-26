from __future__ import annotations

import json
from pathlib import Path

REQUIRED_COMMANDS = [
    "uv sync",
    "uv run ruff check .",
    "uv run pytest tests/test_prd_coverage.py",
    "uv run pytest tests/test_prd_completion_ledger.py",
    "uv run pytest tests/test_prd_evidence_review.py",
    "uv run pytest tests/test_qc_proof.py",
    "uv run pytest tests/test_records.py",
    "uv run pytest tests/test_diagnostics_contract.py",
    "uv run pytest tests/test_architecture_boundaries.py",
    "uv run pytest",
]


def _proof() -> dict:
    return json.loads(Path("docs/qc_proof_0430_0432.json").read_text(encoding="utf-8"))


def test_qc_proof_artifacts_exist_and_parse() -> None:
    assert Path("docs/qc_proof_0430_0432.json").exists()
    assert Path("docs/QC_PROOF_0430_0432.md").exists()
    proof = _proof()
    assert proof["commit_base_before_pass"] == "4b7001cf96fd5da107a5e26afbceba033c0a74ad"
    assert proof["verification_started_at_local"]


def test_qc_proof_records_required_successful_commands() -> None:
    commands = {row["command"]: row for row in _proof()["commands"]}

    for command in REQUIRED_COMMANDS:
        assert command in commands
    for row in commands.values():
        assert row["command"]
        assert row["exit_code"] == 0
        assert row["result"] == "passed"
        assert "notes" in row


def test_qc_proof_records_push_status() -> None:
    push = _proof()["push"]

    assert set(push) >= {"attempted", "result", "remote", "branch", "notes"}
    assert push["attempted"] is True
    assert push["result"] in {"passed", "failed"}
    assert push["remote"]
    assert push["branch"]


def test_qc_proof_markdown_mentions_commands_and_push_status() -> None:
    markdown = Path("docs/QC_PROOF_0430_0432.md").read_text(encoding="utf-8")

    for command in REQUIRED_COMMANDS:
        assert command in markdown
    assert "Push" in markdown
