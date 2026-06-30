from __future__ import annotations

import torch

from attention_lab.models.attention.multi_qkv_common import MultiQKVBaseCausalSelfAttention


class MultiQKVStaticGlobalCausalSelfAttention(MultiQKVBaseCausalSelfAttention):
    attention_type = "multi_qkv_static_3track_global"
    route_formula = "layer_idx_mod_track_count"

    def active_track_indices(self, *, step: int | None, positions: torch.Tensor) -> torch.Tensor:
        del step
        return torch.tensor(self.layer_idx % self.track_count, dtype=torch.long, device=positions.device)
