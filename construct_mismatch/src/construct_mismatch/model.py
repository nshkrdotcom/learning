from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
import yaml
from transformer_lens import HookedTransformer


def choose_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str = "gpt2-small", device: str = "auto") -> HookedTransformer:
    resolved_device = choose_device(device)
    model = HookedTransformer.from_pretrained(model_name, device=resolved_device)
    model.cfg.default_prepend_bos = False
    model.eval()
    return model


def load_model_from_config(path: Path) -> HookedTransformer:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return load_model(config.get("model_name", "gpt2-small"), config.get("device", "auto"))


def encode_text(model: HookedTransformer, text: str) -> list[int]:
    tokens = model.to_tokens(text, prepend_bos=False)
    return [int(token) for token in tokens[0].tolist()]


def decode_tokens(model: HookedTransformer, token_ids: Iterable[int]) -> str:
    return model.tokenizer.decode(list(token_ids))


def token_id_for_target(model: HookedTransformer, target: str) -> int:
    token_ids = encode_text(model, target)
    decoded = decode_tokens(model, token_ids)
    if len(token_ids) != 1 or decoded != target:
        raise ValueError(f"Target {target!r} is not a validated single GPT-2 token: {token_ids}, {decoded!r}")
    return token_ids[0]


def prompt_to_tokens(model: HookedTransformer, prompt: str) -> torch.Tensor:
    return model.to_tokens(prompt, prepend_bos=False)
