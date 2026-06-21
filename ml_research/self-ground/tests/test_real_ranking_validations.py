from __future__ import annotations

import numpy as np
import pytest

from self_ground.real_ranking import run_activation_ranking


def test_ranking_validates_top_k_and_per_family(tmp_path, tiny_model_adapter) -> None:
    with pytest.raises(ValueError, match="top_k_features"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_top",
            top_k_features=0,
            model_adapter=tiny_model_adapter,
        )
    with pytest.raises(ValueError, match="per_family"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_family",
            per_family=0,
            model_adapter=tiny_model_adapter,
        )


def test_ranking_validates_feature_source_and_pooling(tmp_path, tiny_model_adapter) -> None:
    with pytest.raises(ValueError, match="feature_source"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_source",
            feature_source="mlp",
            model_adapter=tiny_model_adapter,
        )
    with pytest.raises(ValueError, match="pooling"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_pooling",
            pooling="last",
            model_adapter=tiny_model_adapter,
        )


def test_sae_ranking_requires_release_and_id(tmp_path, tiny_model_adapter) -> None:
    with pytest.raises(ValueError, match="sae-release"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_sae",
            feature_source="sae",
            model_adapter=tiny_model_adapter,
        )


class BadBatchModel:
    def get_activations(self, texts: list[str], hook_point: str) -> np.ndarray:
        return np.zeros((len(texts) - 1, 1, 2), dtype=float)


def test_ranking_validates_activation_batch_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="activation batch count mismatch"):
        run_activation_ranking(
            out_dir=tmp_path / "bad_batch",
            per_family=1,
            model_adapter=BadBatchModel(),
        )
