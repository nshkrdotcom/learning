from __future__ import annotations

import torch

from attention_lab.models.attention.multi_qkv_common import MultiQKVBaseCausalSelfAttention


class MultiQKVTrainRotationGlobalCausalSelfAttention(MultiQKVBaseCausalSelfAttention):
    attention_type = "multi_qkv_train_rotation_3track_global"
    route_formula = "train_layer_idx_plus_step_mod_track_count_eval_layer_idx_mod_track_count"
    eval_freeze_mode = True

    def active_track_indices(self, *, step: int | None, positions: torch.Tensor) -> torch.Tensor:
        if self.training:
            if step is None:
                raise ValueError("multi_qkv_train_rotation_3track_global requires step during training")
            active = (self.layer_idx + int(step)) % self.track_count
        else:
            active = self.layer_idx % self.track_count
        return torch.tensor(active, dtype=torch.long, device=positions.device)
