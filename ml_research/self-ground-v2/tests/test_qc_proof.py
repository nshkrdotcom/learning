from __future__ import annotations

import json
from pathlib import Path

BASE_COMMIT = "221052c729b55828b82ab13b7116ffcf6feac9b8"

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
    "uv run pytest tests/test_dashboard_query.py",
    "uv run pytest tests/test_open_questions.py",
    "uv run pytest tests/test_parsers_and_run_auditor.py",
    "uv run pytest tests/test_run_lifecycle.py",
    "uv run pytest tests/test_spec_conformance_hardening.py",
    "uv run pytest",
]


def _proof() -> dict:
    return json.loads(Path("docs/qc_proof_0430_0432.json").read_text(encoding="utf-8"))


def test_qc_proof_artifacts_exist_parse_and_name_current_pass() -> None:
    json_path = Path("docs/qc_proof_0430_0432.json")
    markdown_path = Path("docs/QC_PROOF_0430_0432.md")

    assert json_path.exists()
    assert markdown_path.exists()
    proof = _proof()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.strip()
    assert set(proof) >= {
        "commit_base_before_pass",
        "verification_started_at_local",
        "commands",
        "push",
    }
    assert proof["commit_base_before_pass"] == BASE_COMMIT
    assert proof["verification_started_at_local"]
    assert BASE_COMMIT in markdown
    assert "current pass" in markdown.lower()
    assert "not a historical prior pass" in markdown.lower()


def test_qc_proof_records_required_successful_commands_once() -> None:
    proof = _proof()
    commands = [row["command"] for row in proof["commands"]]

    for command in REQUIRED_COMMANDS:
        assert commands.count(command) == 1, command
    for row in proof["commands"]:
        assert set(row) >= {"command", "exit_code", "result", "notes"}
        assert row["exit_code"] == 0, row["command"]
        assert row["result"] == "passed", row["command"]
        assert row["notes"].strip(), row["command"]


def test_qc_proof_records_push_status_honestly() -> None:
    push = _proof()["push"]

    assert set(push) >= {"attempted", "result", "remote", "branch", "notes"}
    assert push["attempted"] is True
    assert push["result"] in {"passed", "failed"}
    assert push["remote"]
    assert push["branch"]
    assert push["notes"].strip()
    if push["result"] == "failed":
        assert any(
            word in push["notes"].lower()
            for word in ("permission", "network", "upstream", "rejected", "failed")
        )


def test_qc_proof_markdown_mentions_commands_commit_and_push_status() -> None:
    markdown = Path("docs/QC_PROOF_0430_0432.md").read_text(encoding="utf-8")

    for command in REQUIRED_COMMANDS:
        assert command in markdown
    assert "Push" in markdown
    assert BASE_COMMIT in markdown
    assert "current pass" in markdown.lower()
