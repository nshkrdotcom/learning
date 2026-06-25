from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mechledger.cli import app
from mechledger.core.decision_log import parse_decision_log
from mechledger.core.experiment_spec import parse_experiment_spec
from mechledger.core.research_log import parse_research_log
from mechledger.core.run_ledger import parse_run_ledger

runner = CliRunner()


def test_flat_file_parsers_success_and_failures(tmp_path: Path) -> None:
    decision = tmp_path / "decision_log.md"
    decision.write_text(
        """# Decision Log

## D001 - Accepted threshold

```yaml
decision_id: D001
status: accepted
affected_experiments: [E001]
affected_claims: [C001]
copilot_session_id: null
```
""",
        encoding="utf-8",
    )
    assert parse_decision_log(decision).decisions["D001"].status == "accepted"

    duplicate = tmp_path / "bad_decision.md"
    duplicate.write_text(
        decision.read_text(encoding="utf-8") + decision.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    try:
        parse_decision_log(duplicate)
    except ValueError as exc:
        assert "decision.id.duplicate" in str(exc)
        assert str(duplicate) in str(exc)
    else:
        raise AssertionError("duplicate decision should fail")

    experiment = tmp_path / "E001_test.md"
    experiment.write_text(
        """# E001: Test

```yaml
experiment_id: E001
status: planned
claim_targets: [C001]
source_runs: []
prerequisites:
  - type: decision_accepted
    id: D001
config_files: []
expected_artifacts: []
```

## Status
planned
## Research question
TODO
## Hypothesis
TODO
## Model / SAE / Hook
TODO
## Task
TODO
## Mechanism objects
TODO
## Claim format
TODO
## Intervention
TODO
## Metrics
metric
## Baselines
baseline
## Controls
control
## Success criterion
success
## Failure criterion
failure
## Prerequisites
decision D001
## Expected artifacts
none
## Notes
none
""",
        encoding="utf-8",
    )
    assert parse_experiment_spec(experiment).experiment_id == "E001"

    research = tmp_path / "research_log.md"
    research.write_text(
        """# Research Log

## 2026-06-25

```yaml
entry_id: R2026-06-25-001
linked_runs: []
linked_claims: []
linked_decisions: []
open_questions: []
copilot_session_id: null
```

### Question
x
### Context
x
### Hypothesis
x
### Work done
x
### Result
x
### Interpretation
x
### Decision
x
### Open questions
x
""",
        encoding="utf-8",
    )
    assert parse_research_log(research).entries[0].entry_id == "R2026-06-25-001"

    ledger = tmp_path / "run_ledger.csv"
    ledger.write_text(
        "date,run_id,git_commit,phase,purpose,hypothesis,command,model,hook_point,sae_release,sae_id,ranking_dir,out_dir,seed,per_family,top_k_features,baseline_mode,operations,status,blocker,key_metric_1,key_metric_2,artifact_paths,decision\n"
        "2026-06-25,R001,abc,phase,purpose,hyp,cmd,,,,,,,,,,,,completed,,,,,\n",
        encoding="utf-8",
    )
    assert parse_run_ledger(ledger).rows[0]["run_id"] == "R001"


def test_run_capture_alias_artifacts_append_status_and_sdk(tmp_path: Path) -> None:
    runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    result = runner.invoke(
        app,
        [
            "run",
            "--experiment",
            "E001",
            "--class",
            "diagnostic",
            "--purpose",
            "hello run",
            "--",
            "python",
            "-c",
            (
                "import os; from pathlib import Path; import mechledger as ml; "
                "print('hello'); ml.log_metric('specificity_gap', 0.1); "
                "Path(os.environ['MECHLEDGER_RUN_DIR'], 'artifacts', 'out.txt').write_text('x')"
            ),
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert result.exit_code == 0, result.output
    run_id = next(
        line.split(":", 1)[1].strip()
        for line in result.output.splitlines()
        if line.startswith("Created run:")
    )
    run_dir = tmp_path / ".mechledger/runs" / run_id
    assert (run_dir / "run.json").exists()
    assert "hello" in (run_dir / "stdout.txt").read_text(encoding="utf-8")
    assert not (run_dir / "heartbeat.json").exists()
    assert json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))["artifacts"]
    assert (run_dir / "scientific_debt_report.json").exists()

    attach_target = tmp_path / "extra.json"
    attach_target.write_text("{}", encoding="utf-8")
    attached = runner.invoke(
        app,
        ["attach", "latest", str(attach_target), "--claim-relevance", "supporting"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert attached.exit_code == 0, attached.output
    annotated = runner.invoke(
        app,
        ["artifact", "annotate", "latest", "A001", "--claim-relevance", "supporting"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert annotated.exit_code == 0, annotated.output

    appended = runner.invoke(
        app,
        ["run-ledger", "append", "latest", "--yes"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert appended.exit_code == 0, appended.output
    assert run_id in (tmp_path / "research/logs/run_ledger.csv").read_text(encoding="utf-8")

    status = runner.invoke(app, ["status"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert status.exit_code == 0, status.output
    assert "Scientific Debt" in status.output


def test_session_experiment_claim_decision_and_debt_workflows(tmp_path: Path) -> None:
    runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    run = runner.invoke(
        app,
        [
            "run",
            "--class",
            "diagnostic",
            "--purpose",
            "workflow",
            "--",
            "python",
            "-c",
            "print('workflow')",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert run.exit_code == 0, run.output
    run_id = next(
        line.split(":", 1)[1].strip()
        for line in run.output.splitlines()
        if line.startswith("Created run:")
    )

    staged_draft = runner.invoke(
        app,
        ["draft", "check", "--staged"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert staged_draft.exit_code == 0
    assert "no staged draft or claim files changed" in staged_draft.output

    session = runner.invoke(
        app,
        ["session", "close"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert session.exit_code == 0, session.output
    assert list((tmp_path / ".mechledger/session_drafts").glob("*.md"))

    crystallize = runner.invoke(
        app,
        [
            "experiment",
            "crystallize",
            "--runs",
            "latest",
            "--id",
            "E099",
            "--title",
            "Workflow Run",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert crystallize.exit_code == 0, crystallize.output
    experiment_path = tmp_path / "research/experiments/E099_workflow_run.md"
    assert run_id in experiment_path.read_text(encoding="utf-8")

    decision = runner.invoke(
        app,
        ["decision", "new", "--from-diff"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert decision.exit_code == 0, decision.output
    decision_log = tmp_path / "research/logs/decision_log.md"
    assert "Review declared-surface diff" in decision_log.read_text(encoding="utf-8")

    with decision_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## D900 - Waive generated diagnostic debt\n\n"
            "```yaml\n"
            "decision_id: D900\n"
            "status: accepted\n"
            "affected_experiments: []\n"
            "affected_claims: []\n"
            "decision_type: methodology\n"
            "copilot_session_id: null\n"
            "```\n"
        )
    waive = runner.invoke(
        app,
        ["debt", "waive", "DPT002", "--decision", "D900"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert waive.exit_code == 0, waive.output
    report = tmp_path / ".mechledger/runs" / run_id / "scientific_debt_report.json"
    debts = json.loads(report.read_text(encoding="utf-8"))["debts"]
    assert any(item["debt_id"] == "DPT002" and item["status"] == "waived" for item in debts)
