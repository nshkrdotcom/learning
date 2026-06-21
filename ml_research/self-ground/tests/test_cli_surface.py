from __future__ import annotations

from typer.testing import CliRunner

from self_ground.cli import app


def test_phase1_cli_surface_contains_real_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "check-real-model" in result.output
    assert "run-activation-ranking" in result.output
    assert "run-residual-intervention" in result.output
    assert "run-negation" not in result.output


def test_cli_surface_has_no_fake_adapter_options() -> None:
    runner = CliRunner()
    for command in [
        ["--help"],
        ["run-activation-ranking", "--help"],
        ["run-residual-intervention", "--help"],
    ]:
        result = runner.invoke(app, command)
        assert result.exit_code == 0
        assert "fake" not in result.output.lower()
        assert "--adapter" not in result.output


def test_proxy_language_is_not_presented_as_causal_intervention() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "feature-space proxy" not in result.output.lower()
