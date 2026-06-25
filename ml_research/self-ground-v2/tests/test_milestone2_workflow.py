from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mechledger.cli import app

runner = CliRunner()


def init_project(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert result.exit_code == 0, result.output
    write_claim_ledger(tmp_path)
    write_decision_log(tmp_path)
    write_run_ledger(tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/e000.json").write_text("{}", encoding="utf-8")


def write_claim_ledger(tmp_path: Path, c001_status: str = "single_run_evidence") -> None:
    (tmp_path / "research/logs/claim_ledger.md").write_text(
        f"""# Claim Ledger

### C001 - Dogfood baseline claim

```yaml
claim_id: C001
status: {c001_status}
allowed:
  - single-run
forbidden:
  - proves that
required_caveats:
  - single run
debt_flags: []
linked_experiments:
  - E000
linked_runs:
  - legacy_e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density
linked_decisions:
  - D001
```

Evidence:
- Dogfood-style E003 fixed baseline coverage but failed feature specificity.

Notes:
Freeform prose here must not make claim proposals stale.
""",
        encoding="utf-8",
    )


def write_decision_log(tmp_path: Path, d001_status: str = "accepted") -> None:
    (tmp_path / "research/logs/decision_log.md").write_text(
        f"""# Decision Log

## D001 - Accept dogfood prerequisite

```yaml
decision_id: D001
status: {d001_status}
affected_experiments:
  - E001
affected_claims:
  - C001
decision_type: methodology
copilot_session_id: null
```
""",
        encoding="utf-8",
    )


def write_run_ledger(tmp_path: Path) -> None:
    (tmp_path / "research/logs/run_ledger.csv").write_text(
        "date,run_id,git_commit,phase,purpose,hypothesis,command,model,hook_point,"
        "sae_release,sae_id,ranking_dir,out_dir,seed,per_family,top_k_features,"
        "baseline_mode,operations,status,blocker,key_metric_1,key_metric_2,"
        "artifact_paths,decision\n"
        "2026-06-25,legacy_e000_completed,abc,E000,purpose,hypothesis,cmd,,,,,,,,,,,,"
        "completed,,,,,D001\n",
        encoding="utf-8",
    )


def write_experiment(tmp_path: Path, experiment_id: str, prerequisites: str) -> Path:
    path = tmp_path / f"research/experiments/{experiment_id.lower()}_test.md"
    path.write_text(
        f"""# {experiment_id}: Milestone 2 prerequisite test

```yaml
experiment_id: {experiment_id}
status: planned
claim_targets:
  - C001
source_runs: []
prerequisites:
{prerequisites}
config_files:
  - configs/e001.yaml
expected_artifacts:
  - artifacts/e000.json
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
machine-readable prerequisites above

## Expected artifacts
artifacts/e000.json

## Notes
Dogfood-style fixture derived from the C001/E003 ledgers.
""",
        encoding="utf-8",
    )
    return path


def run_hello(tmp_path: Path) -> str:
    result = runner.invoke(
        app,
        ["run", "--class", "diagnostic", "--purpose", "m2", "--", "python", "-c", "print('hi')"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert result.exit_code == 0, result.output
    return next(
        line.split(":", 1)[1].strip()
        for line in result.output.splitlines()
        if line.startswith("Created run:")
    )


def test_experiment_validate_passes_with_satisfied_blocking_prerequisites(tmp_path: Path) -> None:
    init_project(tmp_path)
    path = write_experiment(
        tmp_path,
        "E001",
        """  - type: decision_accepted
    id: D001
  - type: experiment_completed
    id: E000
  - type: experiment_completed_and_reviewed
    id: E000
  - type: claim_status_at_least
    id: C001
    status: engineering_verified
  - type: artifact_exists
    path: artifacts/e000.json
""",
    )

    result = runner.invoke(
        app,
        ["experiment", "validate", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "valid E001" in result.output


def test_experiment_validate_uses_local_run_and_artifact_state(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_result = runner.invoke(
        app,
        [
            "run",
            "--experiment",
            "E200",
            "--class",
            "serious_evidence_run",
            "--purpose",
            "local state",
            "--",
            "python",
            "-c",
            (
                "import os; from pathlib import Path; "
                "Path(os.environ['MECHLEDGER_RUN_DIR'], 'artifacts', 'result.json')"
                ".write_text('{}')"
            ),
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert run_result.exit_code == 0, run_result.output
    path = write_experiment(
        tmp_path,
        "E201",
        """  - type: experiment_completed
    id: E200
  - type: artifact_exists
    artifact_id: A001
""",
    )

    result = runner.invoke(
        app,
        ["experiment", "validate", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "valid E201" in result.output


def test_experiment_validate_blocks_missing_decision_and_reports_context(tmp_path: Path) -> None:
    init_project(tmp_path)
    path = write_experiment(
        tmp_path,
        "E002",
        """  - type: decision_accepted
    id: D404
""",
    )

    result = runner.invoke(
        app,
        ["experiment", "validate", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 1
    assert str(path) in result.output
    assert "E002" in result.output
    assert "decision_accepted" in result.output
    assert "experiment.prerequisite.decision_accepted" in result.output
    assert "Suggested fix" in result.output


def test_experiment_validate_scientific_debt_and_incomparable_status(tmp_path: Path) -> None:
    init_project(tmp_path)
    debt_path = write_experiment(
        tmp_path,
        "E003",
        """  - type: artifact_exists
    path: artifacts/missing.json
    consequence: scientific_debt
""",
    )

    debt = runner.invoke(
        app,
        ["experiment", "validate", str(debt_path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert debt.exit_code == 1
    assert "SCIENTIFIC_DEBT" in debt.output
    assert "artifacts/missing.json" in debt.output

    incomparable_path = write_experiment(
        tmp_path,
        "E004",
        """  - type: claim_status_at_least
    id: C001
    status: correlational_support
""",
    )
    incomparable = runner.invoke(
        app,
        ["experiment", "validate", str(incomparable_path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert incomparable.exit_code == 2
    assert "incomparable" in incomparable.output.lower()


def test_next_uses_prerequisite_engine_and_splits_ready_blocked_gated(
    tmp_path: Path,
) -> None:
    init_project(tmp_path)
    write_experiment(
        tmp_path,
        "E010",
        """  - type: decision_accepted
    id: D001
""",
    )
    write_experiment(
        tmp_path,
        "E011",
        """  - type: decision_accepted
    id: D404
""",
    )
    write_experiment(
        tmp_path,
        "E012",
        """  - type: artifact_exists
    path: artifacts/missing.json
    consequence: scientific_debt
""",
    )
    write_experiment(
        tmp_path,
        "E013",
        """  - type: artifact_exists
    path: artifacts/missing-warning.json
    consequence: warning
""",
    )

    result = runner.invoke(app, ["next"], catch_exceptions=False, env={"PWD": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "READY" in result.output and "E010" in result.output
    assert "BLOCKED" in result.output and "E011" in result.output
    assert "DEBT/WARNING GATED" in result.output
    assert "E012" in result.output and "E013" in result.output


def test_run_reclassify_requires_accepted_decision_and_updates_run_files(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_id = run_hello(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "reclassify",
            "latest",
            "--to",
            "serious_evidence_run",
            "--decision",
            "D001",
            "--reason",
            "human-reviewed dogfood transition",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    run_dir = tmp_path / ".mechledger/runs" / run_id
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["run_class"] == "serious_evidence_run"
    transitions = json.loads((run_dir / "run_class_transition.json").read_text(encoding="utf-8"))
    assert transitions[0]["from_run_class"] == "diagnostic"
    assert transitions[0]["to_run_class"] == "serious_evidence_run"
    assert "run_reclassified" in (run_dir / "events.jsonl").read_text(encoding="utf-8")
    debts = json.loads((run_dir / "scientific_debt_report.json").read_text(encoding="utf-8"))[
        "debts"
    ]
    assert not any(item["debt_type"] == "diagnostic_run_only" for item in debts)


def test_run_reclassify_rejects_invalid_inputs(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_hello(tmp_path)

    cases = [
        [
            "run",
            "reclassify",
            "missing",
            "--to",
            "serious_evidence_run",
            "--decision",
            "D001",
            "--reason",
            "x",
        ],
        ["run", "reclassify", "latest", "--to", "bogus", "--decision", "D001", "--reason", "x"],
        [
            "run",
            "reclassify",
            "latest",
            "--to",
            "serious_evidence_run",
            "--decision",
            "D404",
            "--reason",
            "x",
        ],
        [
            "run",
            "reclassify",
            "latest",
            "--to",
            "serious_evidence_run",
            "--decision",
            "D001",
            "--reason",
            "",
        ],
    ]
    write_decision_log(tmp_path, d001_status="proposed")
    cases.append(
        [
            "run",
            "reclassify",
            "latest",
            "--to",
            "serious_evidence_run",
            "--decision",
            "D001",
            "--reason",
            "x",
        ]
    )

    for command in cases:
        result = runner.invoke(app, command, catch_exceptions=False, env={"PWD": str(tmp_path)})
        assert result.exit_code == 2, result.output


def test_run_reclassify_help_does_not_require_project(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "reclassify", "--help"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert "Usage: mechledger run reclassify" in result.output


def test_decision_new_from_declared_surfaces_documents_scope(tmp_path: Path) -> None:
    init_project(tmp_path)
    config = tmp_path / "configs/e001.yaml"
    config.parent.mkdir()
    config.write_text("threshold: 1\n", encoding="utf-8")
    write_experiment(
        tmp_path,
        "E020",
        """  - type: decision_accepted
    id: D001
""",
    )

    result = runner.invoke(
        app,
        ["decision", "new", "--from-declared-surfaces"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    text = (tmp_path / "research/logs/decision_log.md").read_text(encoding="utf-8")
    assert "status: proposed" in text
    assert "configs/e001.yaml" in text
    assert "Refused implicit surfaces" in text
    assert "Python source constants" in text


def test_claim_review_staleness_uses_canonical_yaml_hash_only(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_id = run_hello(tmp_path)
    proposal = runner.invoke(
        app,
        ["claim", "propose", "--run", run_id, "--regenerate"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert proposal.exit_code == 0, proposal.output

    ledger_path = tmp_path / "research/logs/claim_ledger.md"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write("\nAdditional freeform prose.\n")
    current = runner.invoke(
        app,
        ["claim", "review", run_id],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert current.exit_code == 0
    assert "current" in current.output

    write_claim_ledger(tmp_path, c001_status="candidate_claim")
    stale = runner.invoke(
        app,
        ["claim", "review", run_id],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert stale.exit_code == 0
    assert "stale" in stale.output
    refused = runner.invoke(
        app,
        ["claim", "review", run_id, "--apply", "--yes"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert refused.exit_code == 2
    assert "stale" in refused.output.lower()
    forced = runner.invoke(
        app,
        ["claim", "review", run_id, "--apply", "--yes", "--force-stale"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert forced.exit_code == 0, forced.output
    assert "force-stale" in forced.output


def test_debt_waive_requires_accepted_decision_and_keeps_debt_visible(tmp_path: Path) -> None:
    init_project(tmp_path)
    run_id = run_hello(tmp_path)

    write_decision_log(tmp_path, d001_status="proposed")
    proposed = runner.invoke(
        app,
        ["debt", "waive", "DPT002", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert proposed.exit_code == 2
    assert "accepted" in proposed.output

    write_decision_log(tmp_path, d001_status="accepted")
    missing = runner.invoke(
        app,
        ["debt", "waive", "NOPE", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert missing.exit_code == 2
    assert "NOPE" in missing.output

    waived = runner.invoke(
        app,
        ["debt", "waive", "DPT002", "--decision", "D001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert waived.exit_code == 0, waived.output
    report_json = tmp_path / ".mechledger/runs" / run_id / "scientific_debt_report.json"
    debts = json.loads(report_json.read_text(encoding="utf-8"))["debts"]
    assert any(
        item["debt_id"] == "DPT002"
        and item["status"] == "waived"
        and item["waiver_decision_id"] == "D001"
        for item in debts
    )
    report_md = report_json.with_suffix(".md")
    assert "DPT002" in report_md.read_text(encoding="utf-8")
    assert "waived" in report_md.read_text(encoding="utf-8")
