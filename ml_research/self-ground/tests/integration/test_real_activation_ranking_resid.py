from __future__ import annotations

import csv
import json
import subprocess

import pytest


@pytest.mark.integration
def test_real_activation_ranking_residual_dimensions(tmp_path) -> None:
    out_dir = tmp_path / "real_ranking"

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_real_activation_ranking.py",
            "--device",
            "cpu",
            "--per-family",
            "1",
            "--top-k-features",
            "5",
            "--out",
            str(out_dir),
        ],
        check=True,
    )

    metadata = json.loads((out_dir / "activation_metadata.json").read_text())
    assert metadata["feature_source"] == "residual_dimensions"
    assert metadata["n_pairs"] == 4
    assert metadata["n_features"] == 512

    with (out_dir / "feature_rankings.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["feature_id"].startswith("resid_")
