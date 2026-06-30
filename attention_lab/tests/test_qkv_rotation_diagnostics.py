from __future__ import annotations

import json

import torch

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.attention_diagnostics import collect_attention_diagnostics


def multi_config(attention_type: str):
    return config_from_dict(
        {
            "attention_type": attention_type,
            "qkv_track_count": 3,
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


def test_multi_qkv_diagnostics_include_required_fields_after_backward():
    torch.manual_seed(1)
    model = GPT(multi_config("multi_qkv_static_3track_global"))
    idx = torch.randint(0, 64, (2, 8))
    _, loss = model(idx, idx, step=2)
    assert loss is not None
    loss.backward()
    rows = collect_attention_diagnostics(model, step=2)
    assert rows
    row = rows[0]
    for key in (
        "active_track_index",
        "active_track_counts",
        "per_track_gradient_norm",
        "per_track_qkv_weight_norm",
        "per_track_output_norm",
        "track_entropy",
        "route_formula",
        "uses_global_bank",
        "layer_idx",
        "step",
        "schedule_mode",
        "track_count",
        "last_forward_step",
        "track_gradient_norm",
        "position_routing_enabled",
        "eval_freeze_mode",
    ):
        assert key in row
    assert row["uses_global_bank"] is True
    assert row["schedule_mode"] == "train"
    assert row["last_forward_step"] == 2
    assert row["track_gradient_norm"] is not None and row["track_gradient_norm"] > 0.0
    assert max(value or 0.0 for value in row["per_track_gradient_norm"].values()) > 0.0


def test_position_rotation_diagnostics_count_all_tracks():
    torch.manual_seed(2)
    model = GPT(multi_config("multi_qkv_position_rotation_3track_global"))
    idx = torch.randint(0, 64, (2, 8))
    _, loss = model(idx, idx, step=2)
    assert loss is not None
    loss.backward()
    rows = collect_attention_diagnostics(model, step=2)
    first = rows[0]
    assert first["active_track_index"] is None
    assert sum(first["active_track_counts"].values()) == 8
    assert all(count > 0 for count in first["active_track_counts"].values())


def test_diagnostics_rows_are_json_serializable():
    model = GPT(multi_config("multi_qkv_train_rotation_3track_global"))
    model.eval()
    idx = torch.randint(0, 64, (1, 8))
    model(idx)
    rows = collect_attention_diagnostics(model, step=0)
    json.dumps(rows)
