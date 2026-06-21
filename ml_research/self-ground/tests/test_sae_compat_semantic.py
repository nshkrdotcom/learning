from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from self_ground.activations import FeatureActivations
from self_ground.sae_compat import verify_sae_compatibility


class TinySemanticSAE:
    def __init__(
        self,
        *,
        model_name: str | None = "test-local",
        hook_name: str | None = "blocks.2.hook_resid_post",
        decode_mode: str = "same",
    ) -> None:
        self.decode_mode = decode_mode
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
        return FeatureActivations(
            values=np.asarray(values, dtype=float),
            feature_ids=[f"sae_{idx}" for idx in range(values.shape[-1])],
        )

    def decode(self, feature_activations: FeatureActivations):
        values = np.asarray(feature_activations.values, dtype=float)
        if self.decode_mode == "nan":
            decoded = np.array(values, copy=True)
            decoded[0, 0, 0] = np.nan
            return decoded
        if self.decode_mode == "inf":
            decoded = np.array(values, copy=True)
            decoded[0, 0, 0] = np.inf
            return decoded
        return values


def test_semantic_compatible_metadata_and_shapes_return_compatible(
    tmp_path,
    tiny_model_adapter,
) -> None:
    out = tmp_path / "compatibility.json"

    result = verify_sae_compatibility(
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticSAE(),
        out=out,
    )

    assert result.shape_compatible is True
    assert result.metadata_compatible is True
    assert result.reconstruction_compatible is True
    assert result.compatible is True
    assert result.reconstruction_mse == 0.0
    artifact = json.loads(out.read_text())
    assert artifact["metadata_report"]["metadata_compatible"] is True
    assert artifact["reconstruction_l2_relative"] == 0.0


def test_shape_compatible_but_model_mismatch_is_incompatible(tiny_model_adapter) -> None:
    result = verify_sae_compatibility(
        model_name="EleutherAI/pythia-70m",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticSAE(model_name="pythia-70m-deduped"),
    )

    assert result.shape_compatible is True
    assert result.metadata_compatible is False
    assert result.compatible is False
    assert "different checkpoints" in str(result.error)


def test_shape_compatible_but_hook_mismatch_is_incompatible(tiny_model_adapter) -> None:
    result = verify_sae_compatibility(
        model_name="test-local",
        hook_point="blocks.3.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticSAE(hook_name="blocks.2.hook_resid_post"),
    )

    assert result.shape_compatible is True
    assert result.metadata_compatible is False
    assert result.compatible is False
    assert "hook layer mismatch" in str(result.error)


def test_shape_compatible_but_missing_metadata_fails_closed_by_default(
    tiny_model_adapter,
) -> None:
    result = verify_sae_compatibility(
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticSAE(model_name=None, hook_name=None),
    )

    assert result.shape_compatible is True
    assert result.metadata_compatible is False
    assert result.compatible is False
    assert "missing required SAE metadata fields" in str(result.error)


def test_missing_metadata_shape_only_diagnostic_never_becomes_compatible(
    tiny_model_adapter,
) -> None:
    result = verify_sae_compatibility(
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticSAE(model_name=None, hook_name=None),
        allow_shape_only_diagnostic=True,
    )

    assert result.shape_compatible is True
    assert result.metadata_compatible is False
    assert result.compatible is False
    assert result.status == "shape_only_diagnostic_not_production_compatible"


def test_metadata_mismatch_override_is_diagnostic_only(tiny_model_adapter) -> None:
    result = verify_sae_compatibility(
        model_name="EleutherAI/pythia-70m",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticSAE(model_name="pythia-70m-deduped"),
        allow_metadata_mismatch=True,
    )

    assert result.shape_compatible is True
    assert result.metadata_compatible is False
    assert result.semantically_compatible is False
    assert result.compatible is False
    assert result.allow_metadata_mismatch is True
    assert result.diagnostic_only is True
    assert "diagnostic" in result.status


def test_reconstruction_nan_or_inf_is_incompatible(tiny_model_adapter) -> None:
    result = verify_sae_compatibility(
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticSAE(decode_mode="nan"),
    )

    assert result.shape_compatible is True
    assert result.metadata_compatible is True
    assert result.reconstruction_compatible is False
    assert result.compatible is False


def test_deduped_metadata_does_not_pass_for_non_deduped_request(
    tiny_model_adapter,
) -> None:
    result = verify_sae_compatibility(
        model_name="EleutherAI/pythia-70m",
        hook_point="blocks.2.hook_resid_post",
        sae_release="pythia-70m-deduped-res-sm",
        sae_id="blocks.2.hook_resid_post",
        model_adapter=tiny_model_adapter,
        sae_adapter=TinySemanticSAE(model_name="pythia-70m-deduped"),
    )

    assert result.declared_model == "pythia-70m-deduped"
    assert result.metadata_compatible is False
    assert result.compatible is False
