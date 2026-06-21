from __future__ import annotations

from typer.testing import CliRunner

from self_ground.cli import app


def test_cli_help_does_not_expose_fake_mode() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "fake" not in result.output.lower()
    assert "--adapter" not in result.output


def test_activation_ranking_help_does_not_expose_fake_mode() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run-activation-ranking", "--help"])

    assert result.exit_code == 0
    assert "fake" not in result.output.lower()
    assert "--adapter" not in result.output
