from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_sae_compatibility_optional(tmp_path) -> None:
    release = os.getenv("SELF_GROUND_SAE_RELEASE")
    sae_id = os.getenv("SELF_GROUND_SAE_ID")
    if not release or not sae_id:
        pytest.skip("set SELF_GROUND_SAE_RELEASE and SELF_GROUND_SAE_ID")

    from self_ground.sae_compat import verify_sae_compatibility

    out = tmp_path / "compatibility.json"
    result = verify_sae_compatibility(
        sae_release=release,
        sae_id=sae_id,
        out=out,
        device="cpu",
    )

    assert out.exists()
    assert result.compatible is True
    assert result.activation_shape
    assert result.encoded_shape
    assert result.decoded_shape
