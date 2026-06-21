from __future__ import annotations

import pytest


@pytest.mark.integration
def test_transformer_lens_pythia_70m_activation_capture_shape() -> None:
    from self_ground.model import TransformerLensModelAdapter

    adapter = TransformerLensModelAdapter(
        model_name="EleutherAI/pythia-70m",
        device="cpu",
    )
    activations = adapter.get_activations(
        ["The dog is not friendly.", "The dog is friendly."],
        hook_point="blocks.2.hook_resid_post",
    )

    assert activations.shape[0] == 2
    assert activations.ndim == 3
    assert activations.shape[2] == 512


@pytest.mark.integration
def test_transformer_lens_behavior_score_is_finite() -> None:
    from self_ground.model import TransformerLensModelAdapter

    adapter = TransformerLensModelAdapter(
        model_name="EleutherAI/pythia-70m",
        device="cpu",
    )
    score = adapter.score_negation_behavior("The dog is not friendly.")

    assert isinstance(score, float)
    assert score == pytest.approx(score)
