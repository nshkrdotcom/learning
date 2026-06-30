from __future__ import annotations

import torch

from attention_lab.models.attention.multi_qkv_common import MultiQKVBaseCausalSelfAttention


class MultiQKVPositionRotationGlobalCausalSelfAttention(MultiQKVBaseCausalSelfAttention):
    attention_type = "multi_qkv_position_rotation_3track_global"
    route_formula = "layer_idx_plus_position_mod_track_count"
    position_routing_enabled = True

    def active_track_indices(self, *, step: int | None, positions: torch.Tensor) -> torch.Tensor:
        del step
        return (self.layer_idx + positions) % self.track_count
