from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict

from self_ground.activations import CONDITIONS


class TokenContrastScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_tokens: list[str]
    foil_tokens: list[str]
    target_token_ids: list[int]
    foil_token_ids: list[int]
    target_score: float
    foil_score: float
    contrast: float


class BehavioralTaskScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: str
    prompt_result: TokenContrastScore
    control_result: TokenContrastScore


def token_id_for_single_token_string(model_adapter, token_text: str) -> int:
    if not token_text:
        raise ValueError("token string must be non-empty")
    model = getattr(model_adapter, "model", None)
    if model is not None and hasattr(model, "to_tokens"):
        tokens = model.to_tokens(token_text, prepend_bos=False).flatten()
        if int(tokens.numel()) != 1:
            raise ValueError(
                f"expected {token_text!r} to map to exactly one token; got {int(tokens.numel())}"
            )
        return int(tokens[0])
    ids = model_adapter.token_ids_for_strings([token_text])
    if len(ids) != 1:
        raise ValueError(f"expected {token_text!r} to map to exactly one token")
    return int(ids[0])


def _final_logits(logits: torch.Tensor, position: int) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits[:, position, :]
    raise ValueError("logits must be 2D or 3D")


def token_group_logit_score(
    logits: torch.Tensor,
    token_ids: list[int],
    *,
    position: int = -1,
    reduction: Literal["mean", "max"] = "mean",
) -> torch.Tensor:
    if not token_ids:
        raise ValueError("token_ids must contain at least one token id")
    if reduction not in {"mean", "max"}:
        raise ValueError("reduction must be 'mean' or 'max'")
    selected = _final_logits(logits, position)[:, token_ids]
    if reduction == "mean":
        return selected.mean(dim=-1)
    return selected.max(dim=-1).values


def contrast_score_from_logits(
    logits: torch.Tensor,
    *,
    target_token_ids: list[int],
    foil_token_ids: list[int],
    position: int = -1,
    reduction: Literal["mean", "max"] = "mean",
) -> torch.Tensor:
    target = token_group_logit_score(
        logits,
        target_token_ids,
        position=position,
        reduction=reduction,
    )
    foil = token_group_logit_score(
        logits,
        foil_token_ids,
        position=position,
        reduction=reduction,
    )
    return target - foil


def _single_float(value: torch.Tensor, label: str) -> float:
    flat = value.detach().cpu().flatten()
    if flat.numel() != 1:
        raise ValueError(f"{label} must contain exactly one batch item; got {flat.numel()}")
    return float(flat[0].item())


def _score_one(
    *,
    target_tokens: list[str],
    foil_tokens: list[str],
    target_token_ids: list[int],
    foil_token_ids: list[int],
    logits: torch.Tensor,
    reduction: Literal["mean", "max"],
) -> TokenContrastScore:
    target_score = token_group_logit_score(
        logits,
        target_token_ids,
        reduction=reduction,
    )
    foil_score = token_group_logit_score(logits, foil_token_ids, reduction=reduction)
    return TokenContrastScore(
        target_tokens=target_tokens,
        foil_tokens=foil_tokens,
        target_token_ids=target_token_ids,
        foil_token_ids=foil_token_ids,
        target_score=_single_float(target_score, "target_score"),
        foil_score=_single_float(foil_score, "foil_score"),
        contrast=_single_float(target_score - foil_score, "contrast"),
    )


def score_behavioral_task_logits(
    *,
    task,
    validation,
    prompt_logits: torch.Tensor,
    control_logits: torch.Tensor,
    reduction: Literal["mean", "max"] = "mean",
) -> BehavioralTaskScore:
    return BehavioralTaskScore(
        task_id=task.id,
        family=task.family,
        prompt_result=_score_one(
            target_tokens=task.target_tokens,
            foil_tokens=task.foil_tokens,
            target_token_ids=validation.target_token_ids,
            foil_token_ids=validation.foil_token_ids,
            logits=prompt_logits,
            reduction=reduction,
        ),
        control_result=_score_one(
            target_tokens=task.control_target_tokens,
            foil_tokens=task.control_foil_tokens,
            target_token_ids=validation.control_target_token_ids,
            foil_token_ids=validation.control_foil_token_ids,
            logits=control_logits,
            reduction=reduction,
        ),
    )


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
