from __future__ import annotations

from typer.testing import CliRunner

from self_ground.cli import app


def test_phase2_cli_surface_contains_sae_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "check-sae-compatibility" in result.output
    assert "run-sae-intervention" in result.output


def test_phase2_cli_help_has_no_fake_adapter_options() -> None:
    runner = CliRunner()
    for command in [
        ["--help"],
        ["check-sae-compatibility", "--help"],
        ["run-sae-intervention", "--help"],
    ]:
        result = runner.invoke(app, command)
        assert result.exit_code == 0
        assert "fake" not in result.output.lower()
        assert "--adapter" not in result.output


def test_check_sae_compatibility_help_mentions_semantic_metadata() -> None:
    result = CliRunner().invoke(app, ["check-sae-compatibility", "--help"])

    assert result.exit_code == 0
    output = result.output.lower()
    assert "metadata" in output
    assert "shape-only diagnostic" in output
    assert "not production" in output


def test_run_sae_intervention_help_exposes_no_metadata_bypass() -> None:
    result = CliRunner().invoke(app, ["run-sae-intervention", "--help"])

    assert result.exit_code == 0
    output = result.output.lower()
    assert "metadata" not in output or "bypass" not in output
    assert "shape-only" not in output


def test_run_sae_intervention_requires_release_and_id() -> None:
    missing_release = CliRunner().invoke(
        app,
        ["run-sae-intervention", "--out", "runs/x"],
    )
    missing_id = CliRunner().invoke(
        app,
        [
            "run-sae-intervention",
            "--out",
            "runs/x",
            "--sae-release",
            "release",
        ],
    )

    assert missing_release.exit_code != 0
    assert "sae-release" in missing_release.output
    assert missing_id.exit_code != 0
    assert "sae-id" in missing_id.output


def test_run_sae_intervention_rejects_noop_amplification() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run-sae-intervention",
            "--out",
            "runs/x",
            "--sae-release",
            "release",
            "--sae-id",
            "id",
            "--operation",
            "amplify",
            "--factor",
            "1.0",
        ],
    )

    assert result.exit_code != 0
    assert "requires --factor not equal to 1.0" in result.output
