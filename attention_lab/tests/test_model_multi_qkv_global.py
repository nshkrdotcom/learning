from __future__ import annotations

import torch

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.inspect_model_config import inspect_model_config


def tiny_model_config(attention_type: str = "multi_qkv_static_3track_global"):
    return config_from_dict(
        {
            "attention_type": attention_type,
            "multi_qkv_track_count": 3,
            "multi_qkv_global": True,
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
    config = tiny_model_config()
    config.attention_type = "standard"
    return config


def test_global_bank_is_shared_by_all_blocks():
    model = GPT(tiny_model_config())
    bank = model.multi_qkv_bank
    assert bank is not None
    assert model.transformer.h[0].attn.qkv_bank is bank
    assert model.transformer.h[1].attn.qkv_bank is bank
    assert model.transformer.h[2].attn.qkv_bank is bank


def test_multi_qkv_model_forward_shape():
    torch.manual_seed(1)
    model = GPT(tiny_model_config("multi_qkv_position_rotation_3track_global"))
    idx = torch.randint(0, 64, (2, 8))
    logits, loss = model(idx, idx, step=3)
    assert tuple(logits.shape) == (2, 8, 64)
    assert loss is not None and torch.isfinite(loss)


def test_standard_attention_outputs_are_unchanged_by_step_position_threading():
    torch.manual_seed(2)
    model = GPT(standard_config())
    model.eval()
    idx = torch.randint(0, 64, (2, 8))
    positions = torch.arange(8)
    logits_without_context, _ = model(idx)
    logits_with_context, _ = model(idx, step=123, positions=positions)
    assert torch.equal(logits_without_context, logits_with_context)


def test_e002_candidate_parameter_deltas_are_reported(repo_root):
    config_dir = repo_root / "configs" / "experiments" / "E002_multitrack_qkv_shift_register"
    baseline = config_dir / "standard_refactor_control_30m_seed1.yaml"
    static = inspect_model_config(config_dir / "multi_qkv_static_3track_global_30m_seed1.yaml", baseline_config_path=baseline)
    train_rotation = inspect_model_config(
        config_dir / "multi_qkv_train_rotation_3track_global_30m_seed1.yaml",
        baseline_config_path=baseline,
    )
    position = inspect_model_config(
        config_dir / "multi_qkv_position_rotation_3track_global_30m_seed1.yaml",
        baseline_config_path=baseline,
    )

    assert static["attention_type"] == "multi_qkv_static_3track_global"
    assert static["multi_qkv_track_count"] == 3
    assert static["parameter_delta_vs_baseline"] != 0
    assert train_rotation["parameter_delta_vs_baseline"] == static["parameter_delta_vs_baseline"]
    assert position["parameter_delta_vs_baseline"] == static["parameter_delta_vs_baseline"]
