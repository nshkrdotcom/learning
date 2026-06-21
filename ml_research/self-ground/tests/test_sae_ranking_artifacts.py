from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from self_ground.activations import FeatureActivations
from self_ground.real_ranking import run_activation_ranking


class TinySemanticRankingSAE:
    def __init__(
        self,
        *,
        model_name: str = "test-local",
        hook_name: str = "blocks.2.hook_resid_post",
    ) -> None:
        metadata = SimpleNamespace(model_name=model_name, hook_name=hook_name)
        self.sae = SimpleNamespace(
            cfg=SimpleNamespace(
                metadata=metadata,
                d_in=4,
                d_sae=4,
                architecture=lambda: "standard",
            )
        )

    def encode(self, activation) -> FeatureActivations:
        values = activation.detach().cpu().numpy() if hasattr(activation, "detach") else activation
        values = np.asarray(values, dtype=float)
        return FeatureActivations(
            values=values,
            feature_ids=[f"sae_{idx}" for idx in range(values.shape[-1])],
        )

    def decode(self, feature_activations: FeatureActivations):
        return np.asarray(feature_activations.values, dtype=float)


def test_sae_ranking_writes_sae_features_and_metadata(
    tmp_path,
    tiny_model_adapter,
) -> None:
    out_dir = tmp_path / "sae_ranking"

    run_activation_ranking(
        out_dir=out_dir,
        per_family=1,
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        feature_source="sae",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticRankingSAE(),
    )

    metadata = json.loads((out_dir / "activation_metadata.json").read_text())
    assert metadata["feature_source"] == "sae"
    assert metadata["sae_release"] == "test-release"
    assert metadata["sae_id"] == "blocks.2.hook_resid_post"
    assert metadata["metadata_compatible"] is True
    assert metadata["declared_model"] == "test-local"
    assert metadata["declared_hook_point"] == "blocks.2.hook_resid_post"
    assert metadata["metadata_report"]["metadata_compatible"] is True

    readme = (out_dir / "README.md").read_text()
    assert "declared SAE model" in readme
    assert "requested model" in readme

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
) -> None:
    with pytest.raises(ValueError, match="sae-release"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_sae",
            feature_source="sae",
            model_adapter=tiny_model_adapter,
            sae_adapter=TinySemanticRankingSAE(),
        )


def test_sae_ranking_rejects_semantic_metadata_mismatch(
    tmp_path,
    tiny_model_adapter,
) -> None:
    with pytest.raises(ValueError, match="different checkpoints"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_semantic_sae",
            per_family=1,
            model_name="EleutherAI/pythia-70m",
            hook_point="blocks.2.hook_resid_post",
            feature_source="sae",
            sae_release="test-release",
            sae_id="blocks.2.hook_resid_post",
            model_adapter=tiny_model_adapter,
            sae_adapter=TinySemanticRankingSAE(model_name="pythia-70m-deduped"),
        )
