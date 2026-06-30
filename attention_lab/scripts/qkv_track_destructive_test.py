from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from attention_lab.models.attention.multi_qkv_common import is_multi_qkv_attention
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.data_loader import TokenShardLoader
from attention_lab.training.runtime import autocast_context, device_type_from_device, dtype_from_name


def _default_output(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "evals" / "qkv_track_destructive_test.json"
    return Path("reports") / "qkv_track_destructive_test.json"


@torch.no_grad()
def _loss_and_logits(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device_type: str,
    dtype: torch.dtype,
) -> tuple[float, torch.Tensor]:
    with autocast_context(device_type, dtype):
        logits, loss = model(x, y, schedule_mode="eval")
    assert loss is not None
    return float(loss.item()), logits.detach().float()


def run_destructive_test(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    device_type = device_type_from_device(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config = checkpoint["config"]
    model_config = config_from_dict(config["model"], config["data"])
    if not is_multi_qkv_attention(model_config.attention_type):
        raise ValueError(f"qkv_track_destructive_test requires a Multi-QKV checkpoint, got {model_config.attention_type}")

    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    dtype = dtype_from_name(args.dtype or config["train"].get("dtype", "bfloat16"))

    loader = TokenShardLoader(args.data_root or config["data"]["data_root"], args.B, args.T, 0, 1, args.split, True)
    x, y = loader.next_batch()
    x = x.to(device)
    y = y.to(device)

    bank = model.multi_qkv_bank
    if bank is None:
        raise ValueError("Loaded model has no multi_qkv_bank")

    natural_loss, natural_logits = _loss_and_logits(model, x, y, device_type=device_type, dtype=dtype)
    forced_results = []
    for track in range(bank.track_count):
        bank.forced_track = track
        forced_loss, forced_logits = _loss_and_logits(model, x, y, device_type=device_type, dtype=dtype)
        forced_results.append(
            {
                "forced_track": track,
                "loss": forced_loss,
                "loss_delta": forced_loss - natural_loss,
                "mean_abs_logit_delta": float((forced_logits - natural_logits).abs().mean().item()),
            }
        )
    bank.forced_track = None

    swap_results = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        bank.swap_tracks = (a, b)
        swap_loss, swap_logits = _loss_and_logits(model, x, y, device_type=device_type, dtype=dtype)
        swap_results.append(
            {
                "swap_tracks": [a, b],
                "loss": swap_loss,
                "loss_delta": swap_loss - natural_loss,
                "mean_abs_logit_delta": float((swap_logits - natural_logits).abs().mean().item()),
            }
        )
    bank.swap_tracks = None

    result = {
        "checkpoint": str(checkpoint_path),
        "attention_type": model_config.attention_type,
        "split": args.split,
        "B": args.B,
        "T": args.T,
        "natural_loss": natural_loss,
        "forced_track_results": forced_results,
        "swap_track_results": swap_results,
    }
    out_path = Path(args.out) if args.out else _default_output(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(json.dumps(run_destructive_test(args), indent=2))


if __name__ == "__main__":
    main()
