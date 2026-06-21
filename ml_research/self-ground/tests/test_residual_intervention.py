from __future__ import annotations

import pytest
import torch

from self_ground.residual_intervention import (
    parse_residual_feature_id,
    patch_residual_dimensions,
)


def test_parse_residual_feature_id_accepts_valid_ids() -> None:
    assert parse_residual_feature_id("resid_0") == 0
    assert parse_residual_feature_id("resid_123") == 123


@pytest.mark.parametrize("feature_id", ["0", "sae_1", "resid_", "resid_-1", "resid_one"])
def test_parse_residual_feature_id_rejects_invalid_ids(feature_id: str) -> None:
    with pytest.raises(ValueError, match="residual feature id"):
        parse_residual_feature_id(feature_id)


def test_zero_patch_modifies_only_selected_dimensions() -> None:
    activation = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    patched = patch_residual_dimensions(
        activation,
        [1, 3],
        operation="zero",
        factor=1.0,
        token_position=-1,
    )

    assert patched[:, -1, 1].tolist() == [0.0, 0.0]
    assert patched[:, -1, 3].tolist() == [0.0, 0.0]
    assert torch.equal(patched[:, :-1, :], activation[:, :-1, :])
    assert torch.equal(patched[:, -1, [0, 2]], activation[:, -1, [0, 2]])


def test_amplify_patch_modifies_only_selected_dimensions() -> None:
    activation = torch.ones((1, 2, 4), dtype=torch.float32)

    patched = patch_residual_dimensions(
        activation,
        [2],
        operation="amplify",
        factor=3.0,
        token_position=0,
    )

    assert patched[0, 0, 2].item() == 3.0
    assert patched[0, 1, 2].item() == 1.0
    assert patched[0, 0, 0].item() == 1.0


def test_patch_does_not_mutate_original_activation() -> None:
    activation = torch.ones((1, 2, 3), dtype=torch.float32)

    patch_residual_dimensions(activation, [0], operation="zero", factor=1.0)

    assert torch.all(activation == 1.0)


def test_out_of_range_residual_dimension_raises_clear_error() -> None:
    activation = torch.zeros((1, 2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="out of range"):
        patch_residual_dimensions(activation, [3], operation="zero", factor=1.0)
