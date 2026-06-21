from __future__ import annotations

from typer.testing import CliRunner

from self_ground.cli import app


def test_phase3_command_surface() -> None:
    runner = CliRunner()

    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    assert "run-phase3-behavioral-evaluation" in root.output

    result = runner.invoke(app, ["run-phase3-behavioral-evaluation", "--help"])
    assert result.exit_code == 0
    output = result.output.lower()
    assert "ranking-dir" in output
    assert "sae-release" in output
    assert "sae-id" in output
    assert "baseline-mode" in output
    assert "random-seeds" in output
    assert "operations" in output
    assert "amplify-facto" in output
    assert "allow-metadat" in output
    assert "max-relative-" in output
    assert "max-decoded-d" in output
    assert "fake" not in output
