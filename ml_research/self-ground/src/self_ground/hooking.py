from __future__ import annotations

from collections.abc import Callable

import torch

from self_ground.model import TransformerLensModelAdapter


@torch.no_grad()
def run_with_residual_patch(
    model_adapter: TransformerLensModelAdapter,
    texts: list[str],
    hook_point: str,
    patch_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run a real TransformerLens forward pass with an activation patch.

    The patch function receives the activation tensor at `hook_point` and must
    return a tensor with the same shape.
    """

    def hook_fn(activation: torch.Tensor, hook=None) -> torch.Tensor:
        patched = patch_fn(activation)
        if patched.shape != activation.shape:
            raise ValueError(
                "patch_fn must return the same shape as the hooked activation; "
                f"got {tuple(patched.shape)} for {tuple(activation.shape)}"
            )
        return patched

    logits = model_adapter.model.run_with_hooks(
        texts,
        fwd_hooks=[(hook_point, hook_fn)],
    )
    return logits.detach()
