from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from helpers_project import init_project, runner

from mechledger.cli import app
from mechledger.redaction_policy import redact_environment
from mechledger.sdk import ActiveRun


def test_redact_environment_preserves_allowlist_and_redacts_secret_values() -> None:
    payload = redact_environment(
        {
            "PYTHONPATH": "src",
            "CUDA_VISIBLE_DEVICES": "0",
            "OPENAI_API_KEY": "sk-secret",
            "HF_TOKEN": "hf-secret",
            "BENIGN_UNLISTED": "drop-me",
        }
    )

    assert payload["PYTHONPATH"] == "src"
    assert payload["CUDA_VISIBLE_DEVICES"] == "0"
    assert payload["OPENAI_API_KEY"] == "[REDACTED]"
    assert payload["HF_TOKEN"] == "[REDACTED]"
    assert "BENIGN_UNLISTED" not in payload


def test_run_capture_writes_redacted_environment(tmp_path: Path) -> None:
    init_project(tmp_path)

    result = runner.invoke(
        app,
        ["run", "--run-id", "RUN_ENV", "--", "python", "-c", "print('ok')"],
        env={
            "PWD": str(tmp_path),
            "PYTHONPATH": "src",
            "OPENAI_API_KEY": "sk-secret",
            "HF_TOKEN": "hf-secret",
        },
    )

    assert result.exit_code == 0, result.output
    env = json.loads(
        (tmp_path / ".mechledger/runs/RUN_ENV/environment.json").read_text(
            encoding="utf-8"
        )
    )
    assert env["PYTHONPATH"]
    assert env["OPENAI_API_KEY"] == "[REDACTED]"
    assert env["HF_TOKEN"] == "[REDACTED]"
    assert "sk-secret" not in json.dumps(env)


def test_external_call_event_requires_explicit_scope(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    active = ActiveRun(run_dir, "RUN_EXT")

    active.log_event(
        "external_call",
        "queried external source",
        metadata={
            "external": True,
            "service": "neuronpedia",
            "reproducibility_scope": "metadata-only",
        },
    )

    row = json.loads((run_dir / "events.jsonl").read_text(encoding="utf-8"))
    assert row["metadata"]["external"] is True
    assert row["metadata"]["service"] == "neuronpedia"

    with pytest.raises(ValueError, match="external_call events require"):
        active.log_event("external_call", "missing scope", metadata={"external": True})


def test_core_modules_do_not_import_network_clients_or_access_networks() -> None:
    banned_import_roots = {"requests", "httpx", "urllib", "socket", "aiohttp"}
    for path in Path("src/mechledger").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                assert not names & banned_import_roots, (
                    f"network client import in {path}: {names & banned_import_roots}"
                )
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned_import_roots, (
                    f"network client import in {path}: {node.module}"
                )
