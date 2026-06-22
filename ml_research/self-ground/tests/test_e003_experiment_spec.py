from __future__ import annotations

from pathlib import Path


def test_e003_experiment_spec_contains_serious_calibrated_command() -> None:
    text = Path("experiments/E003_calibrated_negation_sae_run.md").read_text()

    assert "scripts/run_e003_calibrated_negation_sae.py" in text
    assert "--device cuda" in text
    assert "--min-calibrated-per-family 10" in text
    assert "--min-baseline-margin 0.1" in text
    assert "top-vs-random-density-and-bottom-active" in text
    assert "baseline-calibrated task bank" in text
