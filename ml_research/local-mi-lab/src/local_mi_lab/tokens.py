from __future__ import annotations

from typing import Any

import numpy as np


def encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(token_id) for token_id in ids]


def decode_token(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode([int(token_id)]))
    if hasattr(tokenizer, "id_to_token"):
        return str(tokenizer.id_to_token[int(token_id)])
    return str(token_id)


def token_id_for_single_token(tokenizer: Any, token_text: str) -> int:
    ids = encode_text(tokenizer, token_text)
    if len(ids) == 0:
        raise ValueError(f"Expected token {token_text!r} is missing from tokenizer output")
    if len(ids) != 1:
        raise ValueError(
            f"Expected token {token_text!r} must encode to one token, got ids {ids}"
        )
    return ids[0]


def validate_expected_token(tokenizer: Any, expected_next_token: str) -> int:
    return token_id_for_single_token(tokenizer, expected_next_token)


def top_token_ids(logits: Any, k: int = 5) -> list[int]:
    values = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    if values.ndim != 1:
        raise ValueError("top_token_ids expects one-dimensional logits")
    k = min(k, values.shape[0])
    return [int(i) for i in np.argsort(values)[::-1][:k]]


def top_tokens(tokenizer: Any, logits: Any, k: int = 5) -> list[dict[str, int | float | str]]:
    values = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    return [
        {
            "token_id": token_id,
            "token": decode_token(tokenizer, token_id),
            "logit": float(values[token_id]),
        }
        for token_id in top_token_ids(values, k)
    ]
