from __future__ import annotations

import numpy as np
import pytest

from self_ground.activations import residual_activations_to_features


def test_final_token_pooling_converts_3d_residuals_to_features() -> None:
    activations = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    features = residual_activations_to_features(activations, pooling="final_token")

    assert features.values.tolist() == [[3.0, 4.0], [7.0, 8.0]]
    assert features.feature_ids == ["resid_0", "resid_1"]


def test_mean_pooling_converts_3d_residuals_to_features() -> None:
    activations = np.asarray(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )

    features = residual_activations_to_features(activations, pooling="mean")

    assert features.values.tolist() == [[3.0, 5.0], [4.0, 6.0]]
    assert features.feature_ids == ["resid_0", "resid_1"]


def test_2d_residuals_keep_stable_feature_ids() -> None:
    activations = np.asarray([[1.0, 2.0, 3.0]])

    features = residual_activations_to_features(activations, pooling="final_token")

    assert features.values.tolist() == [[1.0, 2.0, 3.0]]
    assert features.feature_ids == ["resid_0", "resid_1", "resid_2"]


def test_invalid_pooling_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="invalid pooling"):
        residual_activations_to_features(np.zeros((1, 2, 3)), pooling="last")
