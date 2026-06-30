from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from attention_lab.models.attention_registry import build_attention
from attention_lab.models.attention_standard import StandardCausalSelfAttention
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.inspect_model_config import inspect_model_config


def tiny_gpt_config():
    return config_from_dict(
        {
            "attention_type": "standard",
            "block_size": 8,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        {"vocab_size": 64},
    )


def test_gpt_forward_logits_and_finite_loss():
    torch.manual_seed(123)
    model = GPT(tiny_gpt_config())
    idx = torch.randint(0, 64, (2, 8))
    logits, loss = model(idx, idx)
    assert tuple(logits.shape) == (2, 8, 64)
    assert loss is not None
    assert torch.isfinite(loss)


def test_num_parameters_positive():
    model = GPT(tiny_gpt_config())
    assert model.num_parameters() > 0


def test_standard_attention_registry_path():
    attn = build_attention(tiny_gpt_config())
    assert isinstance(attn, StandardCausalSelfAttention)


def test_unknown_attention_type_raises_clear_error():
    config = SimpleNamespace(attention_type="unknown", n_embd=16, n_head=2, bias=False, dropout=0.0)
    with pytest.raises(ValueError, match="Unknown attention_type"):
        build_attention(config)


def test_trilinear_attention_is_not_runnable_baseline():
    config = SimpleNamespace(attention_type="trilinear_cp", n_embd=16, n_head=2, bias=False, dropout=0.0, cp_rank=8)
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_attention(config)


def test_tiny_forward_is_deterministic_in_eval_mode():
    torch.manual_seed(123)
    model = GPT(tiny_gpt_config())
    model.eval()
    idx = torch.randint(0, 64, (2, 8))
    logits1, _ = model(idx)
    logits2, _ = model(idx)
    assert torch.equal(logits1, logits2)


def test_baseline_config_parameter_counts_are_inspectable(repo_root):
    config_names = [
        "baseline_16m_fineweb100m.yaml",
        "baseline_30m_fineweb100m.yaml",
        "baseline_70m_fineweb300m.yaml",
        "baseline_125m_fineweb1b.yaml",
    ]
    for config_name in config_names:
        result = inspect_model_config(repo_root / "configs" / config_name)
        assert result["attention_type"] == "standard"
        assert result["parameters_excluding_positional"] > 0
        assert result["parameters_including_positional"] >= result["parameters_excluding_positional"]


def test_historical_15m_and_30m_alias_have_same_parameter_count(repo_root):
    historical = inspect_model_config(repo_root / "configs" / "baseline_15m_fineweb100m.yaml")
    canonical = inspect_model_config(repo_root / "configs" / "baseline_30m_fineweb100m.yaml")
    assert historical["parameters_excluding_positional"] == canonical["parameters_excluding_positional"]
    assert historical["parameters_including_positional"] == canonical["parameters_including_positional"]
