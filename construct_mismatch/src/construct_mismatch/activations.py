from __future__ import annotations

import torch
from transformer_lens import HookedTransformer
from transformer_lens.utilities import get_act_name


@torch.no_grad()
def get_logits(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    # Shape: [batch, pos, vocab]
    return model(tokens)


@torch.no_grad()
def get_resid_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layers: list[int] | None = None,
    position: str = "final",
) -> dict[int, torch.Tensor]:
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    names = {get_act_name("resid_post", layer) for layer in layers}
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
    activations: dict[int, torch.Tensor] = {}
    for layer in layers:
        resid = cache[get_act_name("resid_post", layer)]
        # resid shape: [batch, pos, d_model]
        if position == "final":
            activations[layer] = resid[:, -1, :].detach()
        elif position == "all":
            activations[layer] = resid.detach()
        else:
            raise ValueError(f"Unknown position mode: {position}")
    return activations


def get_target_logit_diff(
    logits: torch.Tensor,
    class_a_token_id: int,
    class_b_token_id: int,
) -> torch.Tensor:
    # Return shape: [batch], using final-token logits.
    final_logits = logits[:, -1, :]
    return final_logits[:, class_a_token_id] - final_logits[:, class_b_token_id]
