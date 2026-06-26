from __future__ import annotations

import json
from pathlib import Path

from helpers_project import populate_project, runner

from mechledger.cli import app


def test_dashboard_data_generates_deterministic_project_summary(tmp_path: Path) -> None:
    populate_project(tmp_path)
    out = tmp_path / ".mechledger/dashboard/data.json"

    first = runner.invoke(
        app,
        ["dashboard", "data", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert first.exit_code == 0, first.output
    first_bytes = out.read_bytes()
    second = runner.invoke(
        app,
        ["dashboard", "data", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert second.exit_code == 0, second.output
    assert first_bytes == out.read_bytes()
    payload = json.loads(first_bytes)
    assert payload["claims_by_status"]["candidate_claim"] == 1
    assert payload["runs_by_status"]["completed"] == 1
    assert payload["debt_by_type"]["missing_empirical_null"] == 1
    assert payload["artifacts_by_review_status"]["annotated"] == 1
    assert payload["open_questions"]


def test_dashboard_data_accepts_project_relative_output_path(tmp_path: Path) -> None:
    populate_project(tmp_path)

    result = runner.invoke(
        app,
        ["dashboard", "data", "--out", ".mechledger/dashboard/data.json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "dashboard_data: .mechledger/dashboard/data.json" in result.output
    assert (tmp_path / ".mechledger/dashboard/data.json").exists()


def test_query_commands_return_filtered_json_rows(tmp_path: Path) -> None:
    populate_project(tmp_path)
    claims = runner.invoke(
        app,
        ["query", "claims", "--json", "--status", "candidate_claim"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    runs = runner.invoke(
        app,
        ["query", "runs", "--json", "--experiment", "E001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    debt = runner.invoke(
        app,
        ["query", "debt", "--json", "--severity", "serious"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    artifacts = runner.invoke(
        app,
        ["query", "artifacts", "--json", "--run", "RUN_E001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    decisions = runner.invoke(
        app,
        ["query", "decisions", "--json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    experiments = runner.invoke(
        app,
        ["query", "experiments", "--json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert json.loads(claims.output)[0]["claim_id"] == "C001"
    assert json.loads(runs.output)[0]["run_id"] == "RUN_E001"
    assert json.loads(debt.output)[0]["debt_type"] == "missing_empirical_null"
    assert json.loads(artifacts.output)[0]["artifact_id"] == "A001"
    assert json.loads(decisions.output)[0]["decision_id"] == "D001"
    assert json.loads(experiments.output)[0]["experiment_id"] == "E001"
