from __future__ import annotations

from typing import Any

import numpy as np

from local_mi_lab.tokens import token_id_for_single_token
from local_mi_lab.types import PromptRecord


def logits_at_position(logits: Any, position: int = -1) -> np.ndarray:
    array = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array[position]
    if array.ndim == 3:
        return array[0, position]
    raise ValueError(f"Expected logits with 1, 2, or 3 dimensions, got shape {array.shape}")


def target_logit_score(logits: Any, target_token_id: int, position: int = -1) -> float:
    values = logits_at_position(logits, position)
    _validate_token_id(values, target_token_id)
    return float(values[target_token_id])


def logit_diff_score(
    logits: Any,
    positive_token_id: int,
    negative_token_id: int,
    position: int = -1,
) -> float:
    values = logits_at_position(logits, position)
    _validate_token_id(values, positive_token_id)
    _validate_token_id(values, negative_token_id)
    return float(values[positive_token_id] - values[negative_token_id])


def probability_score(logits: Any, target_token_id: int, position: int = -1) -> float:
    values = logits_at_position(logits, position)
    _validate_token_id(values, target_token_id)
    shifted = values - np.max(values)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    return float(probs[target_token_id])


def rank_score(logits: Any, target_token_id: int, position: int = -1) -> int:
    values = logits_at_position(logits, position)
    _validate_token_id(values, target_token_id)
    return int(1 + np.sum(values > values[target_token_id]))


def normalized_effect_size(
    clean_score: float,
    corrupt_score: float,
    patched_score: float,
) -> dict[str, float | str | None]:
    denominator = clean_score - corrupt_score
    if denominator == 0:
        return {"effect_size": None, "effect_size_status": "denominator_zero"}
    return {
        "effect_size": float((patched_score - corrupt_score) / denominator),
        "effect_size_status": "ok",
    }


def paired_induction_score(
    positive_true_score: float,
    positive_control_score: float,
    family_true_score: float,
    family_control_score: float,
) -> float:
    return float((positive_true_score - positive_control_score) - (family_true_score - family_control_score))


def metric_output(
    metric_name: str,
    score: float | int,
    position: int,
    positive_token_id: int,
    negative_token_id: int | None = None,
    status: str = "ok",
) -> dict[str, float | int | str | None]:
    return {
        "metric_name": metric_name,
        "score": score,
        "position": position,
        "positive_token_id": int(positive_token_id),
        "negative_token_id": None if negative_token_id is None else int(negative_token_id),
        "status": status,
    }


def resolve_induction_token_ids(tokenizer: Any, record: PromptRecord) -> dict[str, int | str]:
    true_token = record.true_expected_next_token or record.expected_next_token
    wrong_or_control = record.wrong_or_control_token
    if not wrong_or_control:
        raise ValueError(f"Record {record.example_id} is missing wrong_or_control_token")
    return {
        "true_expected_next_token": true_token,
        "expected_next_token": record.expected_next_token,
        "wrong_or_control_token": wrong_or_control,
        "true_token_id": token_id_for_single_token(tokenizer, true_token),
        "expected_token_id": token_id_for_single_token(tokenizer, record.expected_next_token),
        "wrong_or_control_token_id": token_id_for_single_token(tokenizer, wrong_or_control),
    }


def _validate_token_id(values: np.ndarray, token_id: int) -> None:
    if token_id < 0 or token_id >= values.shape[-1]:
        raise ValueError(f"token_id {token_id} out of range for vocab {values.shape[-1]}")
