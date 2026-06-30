from __future__ import annotations

import json

import pytest
import torch

from attention_lab.models.gpt import GPT, GPTConfig
from attention_lab.queue.mechanism_checks import evaluate_mechanism_activity
from attention_lab.training.attention_diagnostics import collect_attention_diagnostics


ROUTES = {
    "multi_qkv_static_3track_global": "layer_mod",
    "multi_qkv_train_rotation_3track_global": "layer_plus_step_train_layer_eval",
    "multi_qkv_position_rotation_3track_global": "layer_plus_position",
}


def tiny_multi_qkv_config(attention_type: str) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=3,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type=attention_type,
        qkv_track_count=3,
        qkv_global_bank=True,
        qkv_route_formula=ROUTES[attention_type],
    )


@pytest.mark.parametrize(
    "attention_type",
    [
        "multi_qkv_static_3track_global",
        "multi_qkv_train_rotation_3track_global",
        "multi_qkv_position_rotation_3track_global",
    ],
)
def test_multi_qkv_attention_diagnostics_emit_required_fields(attention_type: str):
    torch.manual_seed(1)
    model = GPT(tiny_multi_qkv_config(attention_type))
    idx = torch.randint(0, 64, (2, 8))

    _, loss = model(idx, idx, step=0, schedule_mode="train")
    assert loss is not None
    loss.backward()

    diag = model.transformer.h[0].attn.attention_diagnostics(step=0, layer=0)
    assert diag is not None
    for key in (
        "attention_type",
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
        "last_forward_step",
        "schedule_mode",
        "track_count",
        "track_gradient_norm",
        "position_routing_enabled",
        "eval_freeze_mode",
    ):
        assert key in diag
    assert diag["attention_type"] == attention_type
    assert diag["uses_global_bank"] is True
    assert diag["track_count"] == 3
    assert diag["schedule_mode"] == "train"
    assert diag["last_forward_step"] == 0
    assert diag["track_gradient_norm"] is not None and diag["track_gradient_norm"] > 0.0


@pytest.mark.parametrize(
    "attention_type",
    [
        "multi_qkv_static_3track_global",
        "multi_qkv_train_rotation_3track_global",
    ],
)
def test_scalar_hard_routing_gradients_go_to_active_track_only(attention_type: str):
    torch.manual_seed(2)
    config = tiny_multi_qkv_config(attention_type)
    config.n_layer = 1
    model = GPT(config)
    idx = torch.randint(0, 64, (2, 8))

    _, loss = model(idx, idx, step=0, schedule_mode="train")
    assert loss is not None
    loss.backward()

    diag = model.transformer.h[0].attn.attention_diagnostics(step=0, layer=0)
    assert diag is not None
    grads = diag["per_track_gradient_norm"]

    assert grads["0"] > 0.0
    assert grads["1"] == 0.0
    assert grads["2"] == 0.0


def test_position_routing_gradients_reach_all_tracks_for_layer0():
    torch.manual_seed(3)
    model = GPT(tiny_multi_qkv_config("multi_qkv_position_rotation_3track_global"))
    idx = torch.randint(0, 64, (2, 8))

    _, loss = model(idx, idx, step=0, schedule_mode="train")
    assert loss is not None
    loss.backward()

    diag = model.transformer.h[0].attn.attention_diagnostics(step=0, layer=0)
    assert diag is not None
    grads = diag["per_track_gradient_norm"]

    assert grads["0"] > 0.0
    assert grads["1"] > 0.0
    assert grads["2"] > 0.0


def test_position_rotation_diagnostics_count_all_tracks():
    torch.manual_seed(4)
    model = GPT(tiny_multi_qkv_config("multi_qkv_position_rotation_3track_global"))
    idx = torch.randint(0, 64, (2, 8))
    _, loss = model(idx, idx, step=2, schedule_mode="train")
    assert loss is not None
    loss.backward()
    rows = collect_attention_diagnostics(model, step=2)
    first = rows[0]
    assert first["active_track_index"] is None
    assert first["active_track_counts"] == {"0": 3, "1": 3, "2": 2}
    assert all(count > 0 for count in first["active_track_counts"].values())


def test_diagnostics_rows_are_json_serializable():
    model = GPT(tiny_multi_qkv_config("multi_qkv_train_rotation_3track_global"))
    model.eval()
    idx = torch.randint(0, 64, (1, 8))
    model(idx, schedule_mode="eval")
    rows = collect_attention_diagnostics(model, step=0)
    json.dumps(rows)


def test_qkv_track_activity_mechanism_check_accepts_nonzero_diagnostics(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    diagnostics_path.write_text(
        json.dumps(
            {
                "attention_type": "multi_qkv_static_3track_global",
                "track_gradient_norm": 0.1,
                "per_track_gradient_norm": {"0": 0.1, "1": 0.0, "2": 0.0},
                "active_track_counts": {"0": 8, "1": 0, "2": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_static_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert verdict.active is True


def test_qkv_track_activity_mechanism_check_rejects_zero_diagnostics(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    diagnostics_path.write_text(
        json.dumps(
            {
                "attention_type": "multi_qkv_static_3track_global",
                "track_gradient_norm": 0.0,
                "per_track_gradient_norm": {"0": 0.0, "1": 0.0, "2": 0.0},
                "active_track_counts": {"0": 8, "1": 0, "2": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_static_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert verdict.active is False
