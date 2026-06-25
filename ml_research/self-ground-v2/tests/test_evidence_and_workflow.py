from __future__ import annotations

import json
import textwrap

from typer.testing import CliRunner

from mechledger.cli import app
from mechledger.evidence import gate_check
from mechledger.indexer import check_project, index_project
from mechledger.next_actions import classify_experiments
from mechledger.scaffold import init_project

runner = CliRunner()


def test_scaffold_init_creates_canonical_files_and_gitignore(tmp_path):
    init_project(tmp_path, project_name="demo")

    assert (tmp_path / "research" / "logs" / "claim_ledger.md").exists()
    assert (tmp_path / "research" / "experiments" / "TEMPLATE_experiment.md").exists()
    assert (tmp_path / ".mechledger" / "project.json").exists()
    assert ".mechledger/runs/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_index_check_validates_project_and_rebuilds_sqlite(tmp_path):
    init_project(tmp_path, project_name="demo")
    index = index_project(tmp_path)
    result = check_project(tmp_path)

    assert index.claim_count_by_status["unsupported"] == 1
    assert result.ok
    assert (tmp_path / ".mechledger" / "index.sqlite").exists()


def test_gate_check_generates_scientific_debt_report_for_diagnostic_run(tmp_path):
    init_project(tmp_path, project_name="demo")
    run_id = "20260625T120301Z_e001_diagnostic_abc123"
    run_dir = tmp_path / ".mechledger" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "parent_run_id": None,
                "experiment_id": "E001",
                "run_class": "diagnostic",
                "status": "completed",
                "purpose": "diagnostic",
                "hypothesis": None,
                "command": None,
                "started_at": "2026-06-25T12:03:01Z",
                "finished_at": "2026-06-25T12:04:01Z",
                "exit_code": 0,
                "git_commit": None,
                "git_diff_hash": None,
                "cwd": str(tmp_path),
                "model": None,
                "tokenizer": None,
                "hook_point": None,
                "sae_release": None,
                "sae_id": None,
                "seed": None,
                "blocker": None,
                "pinned": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                '{"metric_name": "positive_control_pass_rate", "value": 0.7}',
                '{"metric_name": "random_null_seed_count", "value": 3}',
                '{"metric_name": "nonfinite_rate", "value": 0.2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "artifact_manifest.json").write_text('{"artifacts": []}\n', encoding="utf-8")

    report = gate_check(tmp_path, run_id)

    assert not report.clean_candidate_support
    assert [debt.debt_type for debt in report.blockers] == [
        "diagnostic_run_only",
        "failed_positive_control",
        "nonfinite_rows",
    ]
    assert (run_dir / "scientific_debt_report.json").exists()


def test_next_actions_classifies_ready_and_blocked_experiments(tmp_path):
    init_project(tmp_path, project_name="demo")
    (tmp_path / "research" / "logs" / "decision_log.md").write_text(
        textwrap.dedent(
            """
            ## D001 - Accepted method

            ```yaml
            decision_id: D001
            status: accepted
            ```
            """
        ).lstrip(),
        encoding="utf-8",
    )
    exp_dir = tmp_path / "research" / "experiments"
    (exp_dir / "E001_ready.md").write_text(
        textwrap.dedent(
            """
            # E001: Ready

            ```yaml
            experiment_id: E001
            status: planned
            claim_targets: []
            prerequisites:
              - type: decision_accepted
                id: D001
            expected_artifacts: []
            ```

            ## Status
            ## Research question
            ## Hypothesis
            ## Metrics
            ## Controls
            ## Success criterion
            ## Failure criterion
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (exp_dir / "E002_blocked.md").write_text(
        textwrap.dedent(
            """
            # E002: Blocked

            ```yaml
            experiment_id: E002
            status: planned
            claim_targets: []
            prerequisites:
              - type: decision_accepted
                id: D999
            expected_artifacts: []
            ```

            ## Status
            ## Research question
            ## Hypothesis
            ## Metrics
            ## Controls
            ## Success criterion
            ## Failure criterion
            """
        ).lstrip(),
        encoding="utf-8",
    )

    groups = classify_experiments(tmp_path)

    assert [item.experiment_id for item in groups.ready] == ["E001"]
    assert groups.blocked[0].unmet == ["decision D999 is not accepted"]


def test_cli_init_draft_check_and_gate_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--project-name", "demo"], catch_exceptions=False, env={})
    assert result.exit_code == 0

    project = tmp_path
    draft = project / "research" / "paper" / "draft.md"
    draft.write_text(
        "This single run observed a preliminary result. [CLAIM:C001]\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["draft", "check", str(draft)], catch_exceptions=False)

    assert result.exit_code == 0
    assert "0 blocking" in result.output
