from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.config import load_config


def inspect_model_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    model_config = config_from_dict(config["model"], config["data"])
    model = GPT(model_config)
    train_config = config["train"]
    micro_tokens = int(train_config["B"]) * int(train_config["T"])
    total_batch_size = int(train_config["total_batch_size"])
    return {
        "config": str(config_path),
        "run_name": config["run"]["name"],
        "attention_type": model_config.attention_type,
        "n_layer": model_config.n_layer,
        "n_head": model_config.n_head,
        "n_embd": model_config.n_embd,
        "block_size": model_config.block_size,
        "vocab_size": model_config.vocab_size,
        "parameters_excluding_positional": model.num_parameters(non_embedding=True),
        "parameters_including_positional": model.num_parameters(non_embedding=False),
        "estimated_tokens_per_step": total_batch_size,
        "estimated_gradient_accum_steps": total_batch_size // micro_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    result = inspect_model_config(args.config)
    for key, value in result.items():
        print(f"{key}: {value}")
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
