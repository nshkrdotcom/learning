from __future__ import annotations

import pytest

from attention_lab.training.config import save_config
from attention_lab.training.summarize_run import summarize_run
from attention_lab.training.train import train
from attention_lab.training.verify_run import RunVerificationError


def test_summary_works_on_tiny_run(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root)
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / "tiny_test_run"
    summary = summarize_run(run_dir)
    assert summary["run_name"] == "tiny_test_run"
    assert summary["max_step"] == 2
    assert summary["train_event_count"] >= 2
    assert summary["val_event_count"] >= 2
    assert summary["checkpoint_count"] >= 1
    assert (run_dir / "evals" / "run_summary.json").exists()


def test_summary_fails_clearly_on_missing_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RunVerificationError, match="Missing metrics"):
        summarize_run(run_dir)

