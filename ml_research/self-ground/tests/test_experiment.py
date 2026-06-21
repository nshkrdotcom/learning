from __future__ import annotations

import csv
import json

from self_ground.experiment import run_negation_experiment
from self_ground.io import read_jsonl, write_jsonl
from self_ground.negation import generate_negation_pairs


def test_experiment_end_to_end_with_test_local_adapters_writes_meaningful_artifacts(
    tmp_path,
    tiny_model_adapter,
    tiny_sae_adapter,
) -> None:
    pairs = generate_negation_pairs(per_family=3, seed=17)
    pairs_path = tmp_path / "pairs.jsonl"
    out_dir = tmp_path / "run"
    write_jsonl(pairs, pairs_path)

    result = run_negation_experiment(
        pairs_path=pairs_path,
        out_dir=out_dir,
        model_name="test-local",
        layer="test.layer",
        top_k_features=4,
        model_adapter=tiny_model_adapter,
        sae_adapter=tiny_sae_adapter,
    )

    assert result.out_dir == out_dir
    expected = {
        "config.json",
        "pairs.jsonl",
        "feature_rankings.csv",
        "intervention_results.jsonl",
        "summary.csv",
        "README.md",
    }
    assert expected == {path.name for path in out_dir.iterdir()}

    with (out_dir / "feature_rankings.csv").open(newline="") as handle:
        rankings = list(csv.DictReader(handle))
    assert rankings[0]["feature_id"] == "negation"
    assert float(rankings[0]["score"]) > 0.0

    with (out_dir / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    top = summary[0]
    assert top["feature_id"] == "negation"
    assert float(top["specificity_mean"]) > 1.0
    assert float(top["cleanliness_mean"]) > 1.0

    interventions = read_jsonl(out_dir / "intervention_results.jsonl")
    assert len(interventions) == len(pairs) * 4

    config = json.loads((out_dir / "config.json").read_text())
    assert config["model"] == "test-local"
    assert config["top_k_features"] == 4
    assert "adapter" not in config
