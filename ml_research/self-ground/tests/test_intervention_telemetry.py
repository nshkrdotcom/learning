from __future__ import annotations

import torch

from self_ground.activations import FeatureActivations
from self_ground.intervention_telemetry import (
    telemetry_has_nonfinite,
    telemetry_warnings,
)
from self_ground.sae_interventions import decoded_sae_patch_with_telemetry_from_activation


class TinyDecodedSAE:
    def encode(self, activation: torch.Tensor) -> FeatureActivations:
        return FeatureActivations(
            values=activation.detach().cpu().numpy(),
            feature_ids=[f"sae_{idx}" for idx in range(activation.shape[-1])],
        )

    def decode(self, feature_activations: FeatureActivations):
        return feature_activations.values


def test_decoded_patch_telemetry_is_finite_and_does_not_mutate() -> None:
    activation = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    original = activation.clone()

    patched, telemetry = decoded_sae_patch_with_telemetry_from_activation(
        activation=activation,
        sae_adapter=TinyDecodedSAE(),
        feature_ids=["sae_1"],
        operation="ablate",
        token_position=-1,
        patch_mode="delta",
    )

    assert torch.equal(activation, original)
    assert patched.shape == activation.shape
    assert telemetry.selected_feature_activation_abs_mean > 0
    assert telemetry.decoded_delta_norm_mean > 0
    assert telemetry_has_nonfinite(telemetry) is False


def test_norm_drift_warning_triggers() -> None:
    telemetry = {
        "relative_norm_drift_mean": 0.9,
        "decoded_delta_norm_ratio": 0.8,
    }

    warnings = telemetry_warnings(
        telemetry,
        max_relative_norm_drift_warning=0.5,
        max_decoded_delta_norm_ratio_warning=0.5,
    )

    assert warnings["norm_drift_warning"] is True
    assert warnings["decoded_delta_norm_ratio_warning"] is True
