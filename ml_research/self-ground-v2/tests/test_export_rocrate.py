from __future__ import annotations

import json
from pathlib import Path

from helpers_project import populate_project, runner

from mechledger.cli import app


def test_rocrate_export_entities_relationships_and_determinism(tmp_path: Path) -> None:
    populate_project(tmp_path)
    out = tmp_path / "bundles/ro-crate"

    first = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert first.exit_code == 0, first.output
    first_bytes = (out / "ro-crate-metadata.json").read_bytes()

    second = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert second.exit_code == 0, second.output
    assert first_bytes == (out / "ro-crate-metadata.json").read_bytes()

    payload = json.loads(first_bytes)
    graph = {entity["@id"]: entity for entity in payload["@graph"]}
    assert "./" in graph
    assert "research/logs/claim_ledger.md#C001" in graph
    assert "research/logs/decision_log.md#D001" in graph
    assert ".mechledger/runs/RUN_E001/" in graph
    assert ".mechledger/runs/RUN_E001/artifact_manifest.json#A001" in graph
    claim = graph["research/logs/claim_ledger.md#C001"]
    assert claim["linkedRuns"] == [".mechledger/runs/RUN_E001/"]
    assert claim["linkedDecisions"] == ["research/logs/decision_log.md#D001"]
    run = graph[".mechledger/runs/RUN_E001/"]
    assert run["experiment"] == "research/experiments/E001_test.md#E001"
    assert run["artifacts"] == [".mechledger/runs/RUN_E001/artifact_manifest.json#A001"]


def test_rocrate_missing_optional_files_warns_and_malformed_claim_fails(tmp_path: Path) -> None:
    populate_project(tmp_path)
    (tmp_path / "research/logs/research_log.md").unlink()
    out = tmp_path / "bundles/ro-crate"

    missing_optional = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert missing_optional.exit_code == 0, missing_optional.output
    payload = json.loads((out / "ro-crate-metadata.json").read_text(encoding="utf-8"))
    assert any("research_log.md" in warning for warning in payload["mechledger:warnings"])

    (tmp_path / "research/logs/claim_ledger.md").write_text(
        "### C001 - Broken\n\nnot yaml\n", encoding="utf-8"
    )
    malformed = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert malformed.exit_code == 2
    assert "claim_ledger.md" in malformed.output
