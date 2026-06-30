from __future__ import annotations

import json

from attention_lab.queue.screener import (
    ScreenVerdict,
    classify_screen_result,
    mechanism_active_from_diagnostics,
    screen_config_with_overrides,
)


def test_screen_config_overrides_training_budget(tiny_config, tmp_path):
    config = tiny_config(tmp_path, tmp_path / "data")
    screened = screen_config_with_overrides(config, tmp_path / "runs" / "screen" / "candidate")
    assert screened["run"]["out_dir"].endswith("runs/screen/candidate")
    assert screened["train"]["max_steps"] == 150
    assert screened["train"]["val_every"] == 50
    assert screened["train"]["save_every"] == 150


def test_screen_classifier_kill_criteria():
    metrics = [
        {"event": "val", "step": 10, "val_loss": 10.0},
        {"event": "val", "step": 150, "val_loss": 9.8},
    ]
    assert classify_screen_result(returncode=1, stderr="import error", metrics=[], attention_type="standard").failure_class == "COMPILE_ERROR"
    assert classify_screen_result(returncode=1, stderr="CUDA out of memory", metrics=[], attention_type="standard").failure_class == "OOM"
    assert classify_screen_result(
        returncode=0,
        stderr="",
        metrics=[{"event": "val", "step": 50, "val_loss": float("nan")}],
        attention_type="standard",
    ).failure_class == "NAN"
    assert classify_screen_result(returncode=0, stderr="", metrics=metrics, attention_type="standard").failure_class == "FLAT_LOSS"
    slow = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=[{"event": "train", "step": 1, "tokens_per_sec": 10}, {"event": "val", "step": 150, "val_loss": 5.0}],
        attention_type="standard",
        baseline_tokens_per_sec=100.0,
    )
    assert slow.failure_class == "SLOW"


def test_screen_classifier_passes_nonflat_standard_run():
    verdict = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=[
            {"event": "train", "step": 1, "tokens_per_sec": 100.0},
            {"event": "val", "step": 10, "val_loss": 10.0},
            {"event": "val", "step": 150, "val_loss": 9.0},
        ],
        attention_type="standard",
        baseline_tokens_per_sec=100.0,
    )
    assert verdict == ScreenVerdict(passed=True, failure_class=None, mechanism_active=None, step_reached=150)


def test_mechanism_active_from_diagnostics(tmp_path):
    missing = mechanism_active_from_diagnostics("cp_trilinear", tmp_path / "missing.jsonl")
    assert missing is None

    path = tmp_path / "attention_diagnostics.jsonl"
    path.write_text(json.dumps({"cp_gradient_norm": 0.0}) + "\n", encoding="utf-8")
    assert mechanism_active_from_diagnostics("cp_trilinear", path) is False

    path.write_text(json.dumps({"cp_gradient_norm": 1e-3}) + "\n", encoding="utf-8")
    assert mechanism_active_from_diagnostics("cp_trilinear", path) is True
    assert mechanism_active_from_diagnostics("standard", path) is None
