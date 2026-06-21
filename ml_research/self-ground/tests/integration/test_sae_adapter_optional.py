from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.mark.integration
def test_sae_lens_optional_adapter_encode_shape() -> None:
    release = os.getenv("SELF_GROUND_SAE_RELEASE")
    sae_id = os.getenv("SELF_GROUND_SAE_ID")
    if not release or not sae_id:
        pytest.skip("set SELF_GROUND_SAE_RELEASE and SELF_GROUND_SAE_ID to test a real SAE")

    from self_ground.sae import SAELensAdapter

    adapter = SAELensAdapter.from_pretrained(release=release, sae_id=sae_id, device="cpu")
    activations = np.zeros((2, adapter.d_in), dtype=np.float32)
    features = adapter.encode(activations)

    assert features.values.shape[0] == 2
    assert features.values.shape[1] == len(features.feature_ids)
