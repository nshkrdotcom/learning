from __future__ import annotations

from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.config import save_config
from attention_lab.training.train import train


def test_resume_continues_from_checkpoint_and_appends_metrics(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)

    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / "tiny_test_run"
    checkpoint_path = run_dir / "checkpoints" / "ckpt_last.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["step"] == 1
    assert checkpoint["train_loader_state"] is not None
    metric_lines_before = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()

    resumed_config = tiny_config(tmp_path, data_root, max_steps=2)
    resumed_config["train"]["save_every"] = 1
    resumed_config["sample"]["sample_every"] = 2
    save_config(resumed_config, config_path)
    train(config_path, resume_path=str(checkpoint_path))

    resumed_checkpoint = load_checkpoint(checkpoint_path)
    assert resumed_checkpoint["step"] == 2
    metric_lines_after = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metric_lines_after) > len(metric_lines_before)

