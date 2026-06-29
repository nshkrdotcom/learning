from __future__ import annotations

import argparse

from attention_lab.evals.generation_eval import run_generate
from attention_lab.evals.loss_eval import run_eval
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import save_checkpoint
from attention_lab.training.optim import build_optimizer


def make_tiny_checkpoint(tmp_path, tiny_config):
    model_config = config_from_dict(tiny_config["model"], tiny_config["data"])
    model = GPT(model_config)
    optimizer = build_optimizer(model, weight_decay=0.1, learning_rate=0.001, device_type="cpu", master_process=False)
    return save_checkpoint(tmp_path, model, optimizer, tiny_config, step=0)


def test_eval_loss_can_load_tiny_checkpoint(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root)
    checkpoint = make_tiny_checkpoint(tmp_path, config)

    result = run_eval(
        argparse.Namespace(
            checkpoint=str(checkpoint),
            data_root=str(data_root),
            split="val",
            val_steps=1,
            B=2,
            T=8,
            dtype="float32",
            device="cpu",
            out=str(tmp_path / "eval.json"),
        )
    )
    assert result["loss"] > 0
    assert result["perplexity"] > 1
    assert (tmp_path / "eval.json").exists()


def test_generation_eval_writes_nonempty_sample(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root)
    checkpoint = make_tiny_checkpoint(tmp_path, config)
    out_path = tmp_path / "sample.txt"

    samples = run_generate(
        argparse.Namespace(
            checkpoint=str(checkpoint),
            prompt="!",
            num_samples=1,
            max_new_tokens=2,
            top_k=10,
            temperature=1.0,
            seed=42,
            dtype="float32",
            device="cpu",
            out=str(out_path),
        )
    )
    assert samples
    assert out_path.read_text(encoding="utf-8").strip()
