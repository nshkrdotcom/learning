from __future__ import annotations

import csv
import json

import pytest

from self_ground.real_ranking import run_activation_ranking


def test_sae_ranking_writes_sae_features_and_metadata(
    tmp_path,
    tiny_model_adapter,
    tiny_sae_adapter,
) -> None:
    out_dir = tmp_path / "sae_ranking"

    run_activation_ranking(
        out_dir=out_dir,
        per_family=1,
        model_name="test-local",
        hook_point="test.layer",
        feature_source="sae",
        sae_release="test-release",
        sae_id="test-sae",
        top_k_features=2,
        model_adapter=tiny_model_adapter,
        sae_adapter=tiny_sae_adapter,
    )

    metadata = json.loads((out_dir / "activation_metadata.json").read_text())
    assert metadata["feature_source"] == "sae"
    assert metadata["sae_release"] == "test-release"
    assert metadata["sae_id"] == "test-sae"

    with (out_dir / "feature_rankings.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["feature_id"].startswith("sae_")

    top = json.loads((out_dir / "top_examples.jsonl").read_text().splitlines()[0])
    assert top["top_pos_examples"][0]["condition"] == "x_pos"
    assert "text" in top["top_pos_examples"][0]
    assert "activation" in top["top_pos_examples"][0]


def test_sae_ranking_requires_release_and_id(
    tmp_path,
    tiny_model_adapter,
    tiny_sae_adapter,
) -> None:
    with pytest.raises(ValueError, match="sae-release"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_sae",
            feature_source="sae",
            model_adapter=tiny_model_adapter,
            sae_adapter=tiny_sae_adapter,
        )
