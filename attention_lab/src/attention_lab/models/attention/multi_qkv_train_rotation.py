from __future__ import annotations

import torch

from attention_lab.models.attention.multi_qkv_common import MultiQKVBaseCausalSelfAttention


class MultiQKVTrainRotationGlobalCausalSelfAttention(MultiQKVBaseCausalSelfAttention):
    attention_type = "multi_qkv_train_rotation_3track_global"
    route_formula = "(layer_idx + step) % track_count during train; layer_idx % track_count during eval/generate"
    eval_freeze_mode = True

    def active_track_indices(self, *, step: int | None, positions: torch.Tensor, schedule_mode: str) -> torch.Tensor:
        if schedule_mode == "train":
            if step is None:
                raise ValueError("multi_qkv_train_rotation_3track_global requires step during training")
            active = (self.layer_idx + int(step)) % self.track_count
        elif schedule_mode in {"eval", "generate"}:
            active = self.layer_idx % self.track_count
        else:
            raise ValueError("schedule_mode must be one of: train, eval, generate")
        return torch.tensor(active, dtype=torch.long, device=positions.device)
