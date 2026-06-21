from __future__ import annotations

import json

from self_ground.sae_compat import (
    SAECompatibilityResult,
    sae_encoded_shape_is_compatible,
    sae_shapes_are_compatible,
)


def test_compatible_full_sequence_decode_shape() -> None:
    assert sae_shapes_are_compatible([2, 4, 8], [2, 4, 8])


def test_compatible_2d_decode_final_token_shape() -> None:
    assert sae_shapes_are_compatible([2, 4, 8], [2, 8])


def test_incompatible_decode_width_rejected() -> None:
    assert not sae_shapes_are_compatible([2, 4, 8], [2, 7])


def test_incompatible_batch_rejected() -> None:
    assert not sae_shapes_are_compatible([2, 4, 8], [3, 4, 8])


def test_encoded_shape_must_preserve_batch_and_position_when_3d() -> None:
    assert sae_encoded_shape_is_compatible([2, 4, 8], [2, 4, 16])
    assert sae_encoded_shape_is_compatible([2, 4, 8], [2, 16])
    assert not sae_encoded_shape_is_compatible([2, 4, 8], [3, 4, 16])
    assert not sae_encoded_shape_is_compatible([2, 4, 8], [2, 5, 16])


def test_compatibility_result_json_serializes_cleanly() -> None:
    result = SAECompatibilityResult(
        model_name="model",
        hook_point="hook",
        sae_release="release",
        sae_id="id",
        activation_shape=[2, 4, 8],
        encoded_shape=[2, 4, 16],
        decoded_shape=[2, 4, 8],
        d_model=8,
        d_sae=16,
        compatible=True,
        status="ok",
    )

    encoded = json.dumps(result.model_dump(), sort_keys=True)

    assert json.loads(encoded)["compatible"] is True
