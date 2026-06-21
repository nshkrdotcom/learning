from __future__ import annotations

import pytest
import torch

from self_ground.activations import FeatureActivations
from self_ground.sae_interventions import (
    decoded_sae_patch_from_activation,
    modify_sae_features,
    parse_sae_feature_id,
)


class TinyDecodedSAE:
    def __init__(self, decoded_shape: str = "same") -> None:
        self.decoded_shape = decoded_shape

    def encode(self, activation: torch.Tensor) -> FeatureActivations:
        return FeatureActivations(
            values=activation.detach().cpu().numpy(),
            feature_ids=[f"sae_{idx}" for idx in range(activation.shape[-1])],
        )

    def decode(self, feature_activations: FeatureActivations):
        values = torch.as_tensor(feature_activations.values, dtype=torch.float32)
        if self.decoded_shape == "bad_width":
            return values[..., :-1].numpy()
        if self.decoded_shape == "2d":
            return values[:, -1, :].numpy()
        return values.numpy()


def test_parse_sae_feature_id() -> None:
    assert parse_sae_feature_id("sae_0") == 0
    assert parse_sae_feature_id("sae_42") == 42
    with pytest.raises(ValueError, match="SAE feature id"):
        parse_sae_feature_id("resid_1")
    with pytest.raises(ValueError, match="non-negative"):
        parse_sae_feature_id("sae_x")


def test_ablate_on_2d_encoded_tensor() -> None:
    encoded = torch.tensor([[1.0, 2.0, 3.0]])

    modified = modify_sae_features(encoded, [1], operation="ablate")

    assert modified.tolist() == [[1.0, 0.0, 3.0]]


def test_amplify_on_2d_encoded_tensor() -> None:
    encoded = torch.tensor([[1.0, 2.0, 3.0]])

    modified = modify_sae_features(encoded, [2], operation="amplify", factor=4.0)

    assert modified.tolist() == [[1.0, 2.0, 12.0]]


def test_ablate_on_3d_encoded_tensor_final_token() -> None:
    encoded = torch.ones((1, 2, 3))

    modified = modify_sae_features(encoded, [1], operation="ablate", token_position=-1)

    assert modified[0, 0, 1].item() == 1.0
    assert modified[0, 1, 1].item() == 0.0


def test_ablate_on_3d_encoded_tensor_all_positions() -> None:
    encoded = torch.ones((1, 2, 3))

    modified = modify_sae_features(encoded, [1], operation="ablate", token_position=None)

    assert modified[:, :, 1].tolist() == [[0.0, 0.0]]


def test_modify_sae_features_does_not_mutate_input() -> None:
    encoded = torch.ones((1, 2, 3))

    modify_sae_features(encoded, [1], operation="ablate", token_position=None)

    assert torch.all(encoded == 1.0)


def test_modify_sae_features_rejects_out_of_range_feature() -> None:
    with pytest.raises(ValueError, match="out of range"):
        modify_sae_features(torch.ones((1, 3)), [3], operation="ablate")


def test_decoded_patch_rejects_incompatible_decoded_shape() -> None:
    activation = torch.ones((1, 2, 3))

    with pytest.raises(ValueError, match="not compatible"):
        decoded_sae_patch_from_activation(
            activation=activation,
            sae_adapter=TinyDecodedSAE(decoded_shape="bad_width"),
            feature_ids=["sae_1"],
            operation="ablate",
        )


def test_decoded_patch_delta_mode_math() -> None:
    activation = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    patched = decoded_sae_patch_from_activation(
        activation=activation,
        sae_adapter=TinyDecodedSAE(),
        feature_ids=["sae_1"],
        operation="ablate",
        token_position=-1,
        patch_mode="delta",
    )

    assert patched.tolist() == [[[1.0, 2.0, 3.0], [4.0, 0.0, 6.0]]]


def test_decoded_patch_replace_mode_math_with_2d_decode() -> None:
    activation = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    patched = decoded_sae_patch_from_activation(
        activation=activation,
        sae_adapter=TinyDecodedSAE(decoded_shape="2d"),
        feature_ids=["sae_2"],
        operation="ablate",
        token_position=-1,
        patch_mode="replace",
    )

    assert patched.tolist() == [[[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]]
