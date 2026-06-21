from __future__ import annotations

from typing import Literal

import torch

from self_ground.activations import FeatureActivations
from self_ground.hooking import run_with_residual_patch
from self_ground.model import TransformerLensModelAdapter
from self_ground.sae import SAELensAdapter


def parse_sae_feature_id(feature_id: str) -> int:
    prefix = "sae_"
    if not feature_id.startswith(prefix):
        raise ValueError(f"SAE feature id must start with {prefix!r}: {feature_id!r}")
    raw_index = feature_id[len(prefix) :]
    if not raw_index.isdigit():
        raise ValueError(f"SAE feature id must end with a non-negative integer: {feature_id!r}")
    return int(raw_index)


def _normalise_token_position(token_position: int, sequence_length: int) -> int:
    position = token_position if token_position >= 0 else sequence_length + token_position
    if position < 0 or position >= sequence_length:
        raise ValueError(
            f"token_position {token_position} is out of range for sequence length {sequence_length}"
        )
    return position


def modify_sae_features(
    encoded: torch.Tensor,
    feature_indices: list[int],
    *,
    operation: Literal["ablate", "amplify"],
    factor: float = 1.0,
    token_position: int | None = -1,
) -> torch.Tensor:
    if operation not in {"ablate", "amplify"}:
        raise ValueError("operation must be 'ablate' or 'amplify'")
    if operation == "amplify" and factor == 1.0:
        raise ValueError("operation='amplify' requires factor != 1.0")
    if encoded.ndim not in {2, 3}:
        raise ValueError("encoded SAE activations must be 2D or 3D")
    if not feature_indices:
        raise ValueError("feature_indices must contain at least one SAE feature")

    d_sae = encoded.shape[-1]
    for index in feature_indices:
        if index < 0 or index >= d_sae:
            raise ValueError(f"SAE feature index {index} is out of range for d_sae {d_sae}")

    modified = encoded.clone()
    if encoded.ndim == 2:
        target = modified[:, feature_indices]
        if operation == "ablate":
            modified[:, feature_indices] = 0.0
        else:
            modified[:, feature_indices] = target * factor
        return modified

    if token_position is None:
        target = modified[:, :, feature_indices]
        if operation == "ablate":
            modified[:, :, feature_indices] = 0.0
        else:
            modified[:, :, feature_indices] = target * factor
        return modified

    position = _normalise_token_position(token_position, encoded.shape[1])
    target = modified[:, position, feature_indices]
    if operation == "ablate":
        modified[:, position, feature_indices] = 0.0
    else:
        modified[:, position, feature_indices] = target * factor
    return modified


def _to_tensor(value, *, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _encode_to_tensor(activation: torch.Tensor, sae_adapter: SAELensAdapter) -> torch.Tensor:
    encoded = sae_adapter.encode(activation)
    return _to_tensor(encoded.values, like=activation)


def _decode_to_tensor(encoded: torch.Tensor, sae_adapter: SAELensAdapter) -> torch.Tensor:
    feature_ids = [f"sae_{idx}" for idx in range(encoded.shape[-1])]
    decoded = sae_adapter.decode(
        FeatureActivations(
            values=encoded.detach().cpu().numpy(),
            feature_ids=feature_ids,
        )
    )
    return _to_tensor(decoded, like=encoded)


def _align_decoded_to_activation(
    *,
    activation: torch.Tensor,
    decoded: torch.Tensor,
    token_position: int | None,
) -> torch.Tensor:
    if decoded.shape == activation.shape:
        return decoded
    if decoded.ndim == 2 and activation.ndim == 3:
        if decoded.shape[0] != activation.shape[0] or decoded.shape[1] != activation.shape[2]:
            raise ValueError(
                "decoded 2D SAE output is not compatible with activation shape: "
                f"decoded={tuple(decoded.shape)}, activation={tuple(activation.shape)}"
            )
        if token_position is None:
            raise ValueError("2D decoded SAE output requires a token_position")
        position = _normalise_token_position(token_position, activation.shape[1])
        aligned = activation.clone()
        aligned[:, position, :] = decoded
        return aligned
    raise ValueError(
        "decoded SAE output is not compatible with activation shape: "
        f"decoded={tuple(decoded.shape)}, activation={tuple(activation.shape)}"
    )


def decoded_sae_patch_from_activation(
    *,
    activation: torch.Tensor,
    sae_adapter: SAELensAdapter,
    feature_ids: list[str],
    operation: Literal["ablate", "amplify"],
    factor: float = 1.0,
    token_position: int | None = -1,
    patch_mode: Literal["replace", "delta"] = "replace",
) -> torch.Tensor:
    if patch_mode not in {"replace", "delta"}:
        raise ValueError("patch_mode must be 'replace' or 'delta'")
    feature_indices = [parse_sae_feature_id(feature_id) for feature_id in feature_ids]
    encoded_original = _encode_to_tensor(activation, sae_adapter)
    encoded_modified = modify_sae_features(
        encoded_original,
        feature_indices,
        operation=operation,
        factor=factor,
        token_position=token_position,
    )
    decoded_modified = _decode_to_tensor(encoded_modified, sae_adapter)
    modified_residual = _align_decoded_to_activation(
        activation=activation,
        decoded=decoded_modified,
        token_position=token_position,
    )
    if patch_mode == "replace":
        return modified_residual

    decoded_original = _decode_to_tensor(encoded_original, sae_adapter)
    original_residual = _align_decoded_to_activation(
        activation=activation,
        decoded=decoded_original,
        token_position=token_position,
    )
    return activation + (modified_residual - original_residual)


def run_sae_decoded_intervention_logits(
    model_adapter: TransformerLensModelAdapter,
    sae_adapter: SAELensAdapter,
    texts: list[str],
    hook_point: str,
    feature_ids: list[str],
    operation: Literal["ablate", "amplify"],
    factor: float = 1.0,
    token_position: int | None = -1,
    patch_mode: Literal["replace", "delta"] = "delta",
) -> torch.Tensor:
    return run_with_residual_patch(
        model_adapter,
        texts,
        hook_point,
        patch_fn=lambda activation: decoded_sae_patch_from_activation(
            activation=activation,
            sae_adapter=sae_adapter,
            feature_ids=feature_ids,
            operation=operation,
            factor=factor,
            token_position=token_position,
            patch_mode=patch_mode,
        ),
    )
