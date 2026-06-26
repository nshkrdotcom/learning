from __future__ import annotations

import json
from pathlib import Path

from helpers_project import populate_project, runner
from test_external_labels import label_record
from test_records import activation_record

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
    assert payload["claims"][0]["claim_id"] == "C001"
    assert payload["runs"][0]["run_id"] == "RUN_E001"
    assert payload["debt"][0]["debt_id"] == "DPT006"
    assert payload["artifacts"][0]["artifact_id"] == "A001"
    assert payload["blockers"] == []
    assert payload["draft_findings"] == []
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
    labels_source = tmp_path / "labels.jsonl"
    labels_source.write_text(json.dumps(label_record()) + "\n", encoding="utf-8")
    imported = runner.invoke(
        app,
        ["labels", "import", str(labels_source)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert imported.exit_code == 0, imported.output
    record_path = tmp_path / "research/records/activation.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(activation_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    questions = runner.invoke(
        app,
        ["query", "questions", "--json", "--claim", "C001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    labels = runner.invoke(
        app,
        ["query", "labels", "--json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    records = runner.invoke(
        app,
        ["query", "records", "--json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert json.loads(claims.output)[0]["claim_id"] == "C001"
    assert json.loads(runs.output)[0]["run_id"] == "RUN_E001"
    assert json.loads(debt.output)[0]["debt_type"] == "missing_empirical_null"
    assert json.loads(artifacts.output)[0]["artifact_id"] == "A001"
    assert json.loads(decisions.output)[0]["decision_id"] == "D001"
    assert json.loads(experiments.output)[0]["experiment_id"] == "E001"
    assert json.loads(questions.output)[0]["question_id"].endswith("-Q1")
    assert json.loads(labels.output)[0]["label_id"] == "L001"
    assert json.loads(records.output)[0]["record_id"] == "REC-ACT-001"
