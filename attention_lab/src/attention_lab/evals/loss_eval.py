import argparse
import json
import math
from pathlib import Path

import torch

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.data_loader import TokenShardLoader
from attention_lab.training.runtime import device_type_from_device, dtype_from_name
from attention_lab.training.train import evaluate_loss


def default_eval_output(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "evals" / "val_loss.json"
    return Path("reports") / "val_loss.json"


def run_eval(args: argparse.Namespace) -> dict:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    device_type = device_type_from_device(device)

    checkpoint = load_checkpoint(args.checkpoint, device=device)
    config = checkpoint["config"]
    data_config = dict(config["data"])
    if args.data_root:
        data_config["data_root"] = args.data_root

    model_config = config_from_dict(config["model"], data_config)
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    train_config = config["train"]
    B = args.B or int(train_config["B"])
    T = args.T or int(train_config["T"])
    val_steps = args.val_steps or int(train_config["val_steps"])
    dtype = dtype_from_name(args.dtype or train_config.get("dtype", "bfloat16"))

    loader = TokenShardLoader(data_config["data_root"], B, T, 0, 1, args.split, True)
    loss = evaluate_loss(model, loader, val_steps, device, device_type, dtype, ddp=False)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "steps": val_steps,
        "loss": loss,
        "perplexity": math.exp(loss),
    }

    out_path = Path(args.out) if args.out else default_eval_output(Path(args.checkpoint))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--val_steps", type=int, default=None)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = run_eval(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
