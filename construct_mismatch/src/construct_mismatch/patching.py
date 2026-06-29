from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utilities import get_act_name

from construct_mismatch.activations import get_target_logit_diff


@dataclass(frozen=True)
class PairPatchResult:
    pair_id: str
    axis: str
    layer: int
    position: int
    clean_diff: float
    corrupt_diff: float
    patched_diff: float
    recovery: float


@torch.no_grad()
def patch_resid_pair(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    class_a_token_id: int,
    class_b_token_id: int,
    pair_id: str,
    axis: str,
) -> list[PairPatchResult]:
    names = {get_act_name("resid_post", layer) for layer in range(model.cfg.n_layers)}
    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens,
        names_filter=lambda name: name in names,
    )
    corrupt_logits = model(corrupt_tokens)
    clean_diff = float(get_target_logit_diff(clean_logits, class_a_token_id, class_b_token_id)[0])
    corrupt_diff = float(get_target_logit_diff(corrupt_logits, class_a_token_id, class_b_token_id)[0])
    denominator = clean_diff - corrupt_diff
    max_position = min(clean_tokens.shape[1], corrupt_tokens.shape[1])
    results: list[PairPatchResult] = []

    for layer in range(model.cfg.n_layers):
        hook_name = get_act_name("resid_post", layer)
        clean_resid = clean_cache[hook_name]
        for position in range(max_position):

            def patch_hook(
                resid: torch.Tensor,
                hook,
                position: int = position,
                clean_resid: torch.Tensor = clean_resid,
            ) -> torch.Tensor:
                resid[:, position, :] = clean_resid[:, position, :]
                return resid

            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, patch_hook)],
            )
            patched_diff = float(
                get_target_logit_diff(patched_logits, class_a_token_id, class_b_token_id)[0]
            )
            recovery = 0.0 if abs(denominator) < 1e-8 else (patched_diff - corrupt_diff) / denominator
            results.append(
                PairPatchResult(
                    pair_id=pair_id,
                    axis=axis,
                    layer=layer,
                    position=position,
                    clean_diff=clean_diff,
                    corrupt_diff=corrupt_diff,
                    patched_diff=patched_diff,
                    recovery=float(recovery),
                )
            )
    return results


def top_site_stability(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    top_sites: list[tuple[int, int]] = []
    by_pair: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    for pair_rows in by_pair.values():
        best = max(pair_rows, key=lambda row: abs(float(row["recovery"])))
        top_sites.append((int(best["layer"]), int(best["position"])))
    if not top_sites:
        return 0.0
    values, counts = np.unique(top_sites, axis=0, return_counts=True)
    del values
    return float(counts.max() / len(top_sites))
