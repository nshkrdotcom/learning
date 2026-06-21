from __future__ import annotations

import json

from typer.testing import CliRunner

from mechanismlab.cli import app

runner = CliRunner()


def test_cli_help_works() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "backends" in result.output


def test_cli_report_command_writes_generic_report(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifact.json").write_text("{}")
    (run_dir / "evidence_payload.json").write_text(
        json.dumps(
            {
                "compatibility": {"compatible": True},
                "effect_abs": 1.0,
                "n_tasks": 1,
                "n_controls": 1,
                "controls_passed": True,
            }
        )
    )
    claim = tmp_path / "claim.json"
    claim.write_text(
        json.dumps(
            {
                "claim_id": "claim.cli",
                "claim_type": "unit",
                "title": "CLI claim",
                "claim_text": "A CLI claim.",
            }
        )
    )
    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps(
            {
                "experiment_id": "experiment.cli",
                "claim_id": "claim.cli",
                "hypothesis": "CLI hypothesis",
                "required_artifacts": ["artifact.json"],
            }
        )
    )

    result = runner.invoke(
        app,
        ["report", str(run_dir), "--claim", str(claim), "--experiment", str(experiment)],
    )

    assert result.exit_code == 0
    assert (run_dir / "claim_report.json").exists()
    assert (run_dir / "claim_report.md").exists()


def test_cli_backends_works() -> None:
    result = runner.invoke(app, ["backends"])

    assert result.exit_code == 0
    assert "transformer_lens" in result.output
