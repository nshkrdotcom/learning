from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from attention_lab.models.attention.multi_qkv_common import MultiQKVSharedBank
from attention_lab.models.attention.multi_qkv_position_rotation import MultiQKVPositionRotationGlobalCausalSelfAttention
from attention_lab.models.attention.multi_qkv_static import MultiQKVStaticGlobalCausalSelfAttention
from attention_lab.models.attention.multi_qkv_train_rotation import MultiQKVTrainRotationGlobalCausalSelfAttention
from attention_lab.models.attention.standard import StandardCausalSelfAttention
from attention_lab.models.gpt import config_from_dict


def multi_config(attention_type: str = "multi_qkv_static_3track_global", track_count: int = 3):
    return config_from_dict(
        {
            "attention_type": attention_type,
            "qkv_track_count": track_count,
            "qkv_global_bank": True,
            "qkv_route_formula": None,
            "block_size": 8,
            "n_layer": 3,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        {"vocab_size": 64},
    )


def standard_config():
    config = multi_config(track_count=1)
    config.attention_type = "standard"
    return config


def copy_standard_weights(target: MultiQKVStaticGlobalCausalSelfAttention, source: StandardCausalSelfAttention) -> None:
    q_weight, k_weight, v_weight = source.c_attn.weight.data.chunk(3, dim=0)
    target.qkv_bank.q_proj[0].weight.data.copy_(q_weight)
    target.qkv_bank.k_proj[0].weight.data.copy_(k_weight)
    target.qkv_bank.v_proj[0].weight.data.copy_(v_weight)
    target.c_proj.load_state_dict(deepcopy(source.c_proj.state_dict()))


def test_multi_qkv_static_forward_shape():
    torch.manual_seed(1)
    config = multi_config()
    bank = MultiQKVSharedBank(config)
    attention = MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=1, shared_qkv_bank=bank)
    y = attention(torch.randn(2, 8, 16))
    assert tuple(y.shape) == (2, 8, 16)


def test_multi_qkv_causal_mask_prevents_future_token_influence():
    torch.manual_seed(2)
    config = multi_config("multi_qkv_position_rotation_3track_global")
    bank = MultiQKVSharedBank(config)
    attention = MultiQKVPositionRotationGlobalCausalSelfAttention(config, layer_idx=0, shared_qkv_bank=bank)
    attention.eval()
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:]) * 10.0

    y1 = attention(x1)
    y2 = attention(x2)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6, rtol=1e-5)


def test_one_track_multi_qkv_matches_standard_attention_with_shared_weights():
    torch.manual_seed(3)
    standard = StandardCausalSelfAttention(standard_config())
    config = multi_config(track_count=1)
    bank = MultiQKVSharedBank(config)
    multi = MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=0, shared_qkv_bank=bank)
    copy_standard_weights(multi, standard)
    standard.eval()
    multi.eval()
    x = torch.randn(2, 8, 16)
    assert torch.allclose(multi(x), standard(x), atol=1e-6, rtol=1e-5)


def test_static_route_formula_layer_idx_mod_three():
    config = multi_config()
    bank = MultiQKVSharedBank(config)
    positions = torch.arange(8)
    assert int(
        MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=0, shared_qkv_bank=bank).active_track_indices(
            step=None, positions=positions, schedule_mode="train"
        )
    ) == 0
    assert int(
        MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=4, shared_qkv_bank=bank).active_track_indices(
            step=None, positions=positions, schedule_mode="eval"
        )
    ) == 1


def test_train_rotation_formula_and_eval_freeze():
    config = multi_config("multi_qkv_train_rotation_3track_global")
    bank = MultiQKVSharedBank(config)
    attention = MultiQKVTrainRotationGlobalCausalSelfAttention(config, layer_idx=2, shared_qkv_bank=bank)
    positions = torch.arange(8)
    attention.train()
    assert int(attention.active_track_indices(step=5, positions=positions, schedule_mode="train")) == (2 + 5) % 3
    with pytest.raises(ValueError, match="requires step"):
        attention(torch.randn(1, 8, 16))
    attention.eval()
    assert int(attention.active_track_indices(step=None, positions=positions, schedule_mode="eval")) == 2


def test_position_rotation_formula_per_position():
    config = multi_config("multi_qkv_position_rotation_3track_global")
    bank = MultiQKVSharedBank(config)
    attention = MultiQKVPositionRotationGlobalCausalSelfAttention(config, layer_idx=1, shared_qkv_bank=bank)
    positions = torch.arange(6)
    assert torch.equal(
        attention.active_track_indices(step=None, positions=positions, schedule_mode="generate"),
        torch.tensor([1, 2, 0, 1, 2, 0]),
    )


def test_inactive_hard_switch_tracks_receive_no_gradient_for_static_route():
    torch.manual_seed(4)
    config = multi_config()
    bank = MultiQKVSharedBank(config)
    attention = MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=0, shared_qkv_bank=bank)
    loss = attention(torch.randn(2, 8, 16)).pow(2).mean()
    loss.backward()
    assert bank.q_proj[0].weight.grad is not None and bank.q_proj[0].weight.grad.abs().sum() > 0
    for track in (1, 2):
        for module_list in (bank.q_proj, bank.k_proj, bank.v_proj):
            grad = module_list[track].weight.grad
            assert grad is None or grad.abs().sum() == 0
