from __future__ import annotations

import pytest
import torch


@pytest.mark.integration
def test_real_residual_patch_changes_logits() -> None:
    from self_ground.hooking import run_with_residual_patch
    from self_ground.model import TransformerLensModelAdapter

    adapter = TransformerLensModelAdapter("EleutherAI/pythia-70m", device="cpu")
    texts = ["The dog is friendly.", "The dog is not friendly."]
    baseline = adapter.logits_for_texts(texts)
    patched = run_with_residual_patch(
        adapter,
        texts,
        "blocks.2.hook_resid_post",
        patch_fn=lambda activation: torch.zeros_like(activation),
    )

    assert patched.shape == baseline.shape
    assert torch.mean(torch.abs(patched - baseline)).item() > 0.0
