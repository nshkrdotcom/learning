from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from attention_lab.training.data_manifest import load_data_manifest


class ResumeCompatibilityError(ValueError):
    pass


def _require_equal(current: Any, checkpoint: Any, label: str) -> None:
    if current != checkpoint:
        raise ResumeCompatibilityError(
            f"Resume config mismatch for {label}: current={current!r}, checkpoint={checkpoint!r}"
        )


def validate_resume_compatibility(
    current_config: dict[str, Any],
    checkpoint_config: dict[str, Any],
    *,
    checkpoint_step: int,
) -> None:
    _require_equal(current_config["model"], checkpoint_config["model"], "model config")
    for key in ("tokenizer", "vocab_size"):
        _require_equal(current_config["data"].get(key), checkpoint_config["data"].get(key), f"data.{key}")
    for key in ("B", "T", "total_batch_size"):
        _require_equal(current_config["train"].get(key), checkpoint_config["train"].get(key), f"train.{key}")
    for key in ("weight_decay", "learning_rate", "min_lr", "warmup_steps", "grad_clip"):
        _require_equal(current_config["train"].get(key), checkpoint_config["train"].get(key), f"train.{key}")
    if int(current_config["train"]["max_steps"]) <= checkpoint_step:
        raise ResumeCompatibilityError(
            f"train.max_steps must be greater than checkpoint step {checkpoint_step} when resuming"
        )


def assert_model_state_compatible(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> None:
    current_state = model.state_dict()
    current_keys = set(current_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(current_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - current_keys)
    if missing or unexpected:
        raise ResumeCompatibilityError(f"Checkpoint model keys differ: missing={missing}, unexpected={unexpected}")
    mismatched = [
        key
        for key in sorted(current_keys)
        if tuple(current_state[key].shape) != tuple(checkpoint_state[key].shape)
    ]
    if mismatched:
        details = ", ".join(
            f"{key}: current={tuple(current_state[key].shape)} checkpoint={tuple(checkpoint_state[key].shape)}"
            for key in mismatched[:5]
        )
        raise ResumeCompatibilityError(f"Checkpoint model shape mismatch: {details}")


def _manifest_comparison_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "tokenizer": manifest.get("tokenizer"),
        "dataset": manifest.get("dataset"),
        "dataset_config": manifest.get("dataset_config"),
        "shards": manifest.get("shards"),
        "total_train_tokens": manifest.get("total_train_tokens"),
        "total_val_tokens": manifest.get("total_val_tokens"),
    }


def validate_resume_data_manifest(run_dir: str | Path, data_root: str | Path) -> None:
    run_manifest_path = Path(run_dir) / "data_manifest.json"
    current_manifest_path = Path(data_root) / "manifest.json"
    if not run_manifest_path.is_file() or not current_manifest_path.is_file():
        return
    run_payload = _manifest_comparison_payload(load_data_manifest(run_manifest_path))
    current_payload = _manifest_comparison_payload(load_data_manifest(current_manifest_path))
    if run_payload != current_payload:
        raise ResumeCompatibilityError(
            "Resume data manifest mismatch:\n"
            f"run={json.dumps(run_payload, sort_keys=True)}\n"
            f"current={json.dumps(current_payload, sort_keys=True)}"
        )
