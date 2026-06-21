from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from self_ground.hooking import run_with_residual_patch
from self_ground.model import TransformerLensModelAdapter


@dataclass(frozen=True)
class ResidualInterventionSpec:
    feature_ids: list[str]
    operation: Literal["zero", "amplify"]
    factor: float = 1.0
    hook_point: str = "blocks.2.hook_resid_post"
    pooling: Literal["final_token"] = "final_token"


def parse_residual_feature_id(feature_id: str) -> int:
    prefix = "resid_"
    if not feature_id.startswith(prefix):
        raise ValueError(f"residual feature id must start with {prefix!r}: {feature_id!r}")
    raw_index = feature_id[len(prefix) :]
    if not raw_index.isdigit():
        raise ValueError(
            "residual feature id must end with a non-negative integer: "
            f"{feature_id!r}"
        )
    return int(raw_index)


def _normalise_token_position(token_position: int, sequence_length: int) -> int:
    position = token_position if token_position >= 0 else sequence_length + token_position
    if position < 0 or position >= sequence_length:
        raise ValueError(
            f"token_position {token_position} is out of range for sequence length {sequence_length}"
        )
    return position


def patch_residual_dimensions(
    activation: torch.Tensor,
    feature_indices: list[int],
    *,
    operation: Literal["zero", "amplify"],
    factor: float,
    token_position: int = -1,
) -> torch.Tensor:
    if operation not in {"zero", "amplify"}:
        raise ValueError("operation must be 'zero' or 'amplify'")
    if activation.ndim != 3:
        raise ValueError("residual activation must have shape [batch, position, d_model]")
    if not feature_indices:
        raise ValueError("feature_indices must contain at least one residual dimension")

    _, sequence_length, d_model = activation.shape
    position = _normalise_token_position(token_position, sequence_length)
    for index in feature_indices:
        if index < 0 or index >= d_model:
            raise ValueError(
                f"residual dimension index {index} is out of range for d_model {d_model}"
            )

    patched = activation.clone()
    if operation == "zero":
        patched[:, position, feature_indices] = 0.0
    else:
        patched[:, position, feature_indices] = patched[:, position, feature_indices] * factor

    if patched.shape != activation.shape:
        raise ValueError("patched activation shape changed unexpectedly")
    return patched


def run_residual_intervention_logits(
    model_adapter: TransformerLensModelAdapter,
    texts: list[str],
    hook_point: str,
    feature_ids: list[str],
    operation: Literal["zero", "amplify"],
    factor: float = 1.0,
    token_position: int = -1,
) -> torch.Tensor:
    feature_indices = [parse_residual_feature_id(feature_id) for feature_id in feature_ids]
    return run_with_residual_patch(
        model_adapter,
        texts,
        hook_point,
        patch_fn=lambda activation: patch_residual_dimensions(
            activation,
            feature_indices,
            operation=operation,
            factor=factor,
            token_position=token_position,
        ),
    )
