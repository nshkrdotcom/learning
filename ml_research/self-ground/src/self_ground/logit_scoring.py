from __future__ import annotations

from typing import Any

import torch

from self_ground.activations import CONDITIONS


def contrast_from_logits(
    *,
    model_adapter,
    logits: torch.Tensor,
    positive_tokens: list[str],
    negative_tokens: list[str],
) -> list[float]:
    pos_ids = model_adapter.token_ids_for_strings(positive_tokens)
    neg_ids = model_adapter.token_ids_for_strings(negative_tokens)
    final_logits = logits[:, -1, :]
    pos = final_logits[:, pos_ids].mean(dim=-1)
    neg = final_logits[:, neg_ids].mean(dim=-1)
    return (pos - neg).detach().cpu().tolist()


def condition_dict(values: list[float]) -> dict[str, float]:
    if len(values) != len(CONDITIONS):
        raise ValueError(f"expected {len(CONDITIONS)} condition values, got {len(values)}")
    return dict(zip(CONDITIONS, values, strict=True))


def delta_dict(
    baseline: dict[str, float],
    patched: dict[str, float],
) -> dict[str, float]:
    return {condition: patched[condition] - baseline[condition] for condition in CONDITIONS}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def delta_metrics(delta: dict[str, float]) -> dict[str, Any]:
    signed_negation_delta_mean = _mean([delta["x_pos"], delta["x_para"]])
    signed_control_delta_mean = _mean([delta["x_neg"], delta["x_decoy"]])
    absolute_negation_delta_mean = _mean([abs(delta["x_pos"]), abs(delta["x_para"])])
    absolute_control_delta_mean = _mean([abs(delta["x_neg"]), abs(delta["x_decoy"])])
    return {
        "signed_negation_delta_mean": signed_negation_delta_mean,
        "signed_control_delta_mean": signed_control_delta_mean,
        "absolute_negation_delta_mean": absolute_negation_delta_mean,
        "absolute_control_delta_mean": absolute_control_delta_mean,
        "signed_specificity_score": signed_negation_delta_mean - signed_control_delta_mean,
        "absolute_specificity_score": absolute_negation_delta_mean - absolute_control_delta_mean,
    }
