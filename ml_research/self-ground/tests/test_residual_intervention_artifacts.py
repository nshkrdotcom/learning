from __future__ import annotations

import csv
import json

from self_ground.real_ranking import run_activation_ranking
from self_ground.real_residual_intervention import run_real_residual_intervention


def test_residual_intervention_artifacts_with_test_local_model(
    tmp_path,
    tiny_model_adapter,
) -> None:
    ranking_dir = tmp_path / "ranking"
    out_dir = tmp_path / "intervention"
    run_activation_ranking(
        out_dir=ranking_dir,
        per_family=1,
        model_name="test-local",
        hook_point="test.layer",
        feature_source="residual_dimensions",
        top_k_features=2,
        model_adapter=tiny_model_adapter,
    )

    result = run_real_residual_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        model_name="test-local",
        hook_point="test.layer",
        top_k_features=2,
        operation="zero",
        model_adapter=tiny_model_adapter,
    )

    assert result.n_pairs == 4
    assert result.n_features == 2
    expected = {
        "config.json",
        "selected_features.json",
        "intervention_results.jsonl",
        "summary.csv",
        "README.md",
    }
    assert expected == {path.name for path in out_dir.iterdir()}

    selected = json.loads((out_dir / "selected_features.json").read_text())
    assert selected["source"] == "ranking_dir"
    assert all(feature_id.startswith("resid_") for feature_id in selected["feature_ids"])

    rows = [
        json.loads(line)
        for line in (out_dir / "intervention_results.jsonl").read_text().splitlines()
    ]
    assert rows
    first = rows[0]
    assert "baseline" in first
    assert "patched" in first
    assert "delta" in first
    assert "specificity_score" in first
    assert set(first["delta"]) == {"x_pos", "x_neg", "x_para", "x_decoy"}

    with (out_dir / "summary.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert header == [
        "feature_set",
        "operation",
        "n_pairs",
        "negation_specific_delta_mean",
        "control_delta_mean",
        "specificity_score_mean",
    ]

    readme = (out_dir / "README.md").read_text().lower()
    assert "real transformerlens residual intervention" in readme
    assert "not an sae decoded intervention" in readme
    assert "sae feature intervention" not in readme
