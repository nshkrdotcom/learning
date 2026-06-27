from __future__ import annotations

from typing import Any

import torch


def dtype_from_config(config: dict[str, Any]) -> torch.dtype:
    dtype_name = str(config.get("model", {}).get("dtype", "float32"))
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return mapping[dtype_name]


def requested_device(config: dict[str, Any]) -> str:
    return str(config.get("model", {}).get("device", "cuda"))


def resolve_device(config: dict[str, Any]) -> str:
    device = requested_device(config)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false")
    return device


def load_hooked_transformer(config: dict[str, Any]):
    from transformer_lens import HookedTransformer

    model_name = str(config["model"]["name"])
    device = resolve_device(config)
    dtype = dtype_from_config(config)
    return HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)


def n_layers(model: Any) -> int:
    if hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
        return int(model.cfg.n_layers)
    if hasattr(model, "blocks"):
        return len(model.blocks)
    raise ValueError("Could not determine number of layers for model")


def vocab_size(model: Any) -> int:
    if hasattr(model, "cfg") and hasattr(model.cfg, "d_vocab"):
        return int(model.cfg.d_vocab)
    if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "vocab_size"):
        return int(model.tokenizer.vocab_size)
    raise ValueError("Could not determine vocab size for model")
