from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from self_ground.io import write_config
from self_ground.real_model_check import DEFAULT_CHECK_TEXTS
from self_ground.sae_metadata import (
    extract_sae_identity_metadata,
    metadata_matches_requested_target,
)


@dataclass(frozen=True)
class SAECompatibilityResult:
    model_name: str
    hook_point: str
    sae_release: str
    sae_id: str
    activation_shape: list[int]
    encoded_shape: list[int]
    decoded_shape: list[int]
    d_model: int
    d_sae: int
    shape_compatible: bool = False
    metadata_compatible: bool = False
    reconstruction_compatible: bool = False
    compatible: bool = False
    status: str = "error"
    error: str | None = None
    declared_model: str | None = None
    declared_hook_point: str | None = None
    declared_hook_layer: int | None = None
    declared_hook_type: str | None = None
    requested_hook_layer: int | None = None
    requested_hook_type: str | None = None
    reconstruction_mse: float | None = None
    reconstruction_l2_relative: float | None = None
    reconstruction_max_abs_error: float | None = None
    metadata_report: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def model_dump(self) -> dict:
        return asdict(self)


def _shape(value) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    return [int(dim) for dim in shape]


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def sae_shapes_are_compatible(activation_shape: list[int], decoded_shape: list[int]) -> bool:
    if len(activation_shape) != 3:
        return False
    if decoded_shape == activation_shape:
        return True
    if len(decoded_shape) == 2:
        return (
            decoded_shape[0] == activation_shape[0]
            and decoded_shape[1] == activation_shape[2]
        )
    return False


def sae_encoded_shape_is_compatible(
    activation_shape: list[int],
    encoded_shape: list[int],
) -> bool:
    if len(activation_shape) != 3 or len(encoded_shape) not in {2, 3}:
        return False
    if encoded_shape[0] != activation_shape[0]:
        return False
    if len(encoded_shape) == 3 and encoded_shape[1] != activation_shape[1]:
        return False
    return True


def _aligned_reconstruction_arrays(
    *,
    activations,
    decoded,
) -> tuple[np.ndarray, np.ndarray]:
    activation_values = _to_numpy(activations)
    decoded_values = _to_numpy(decoded)
    if decoded_values.shape == activation_values.shape:
        return activation_values, decoded_values
    if (
        activation_values.ndim == 3
        and decoded_values.ndim == 2
        and decoded_values.shape[0] == activation_values.shape[0]
        and decoded_values.shape[1] == activation_values.shape[2]
    ):
        return activation_values[:, -1, :], decoded_values
    raise ValueError(
        "decoded SAE output is not shape-compatible with hook activation: "
        f"decoded={list(decoded_values.shape)}, activation={list(activation_values.shape)}"
    )


def _compute_reconstruction_metrics(
    *,
    activations,
    decoded,
) -> tuple[float, float, float]:
    expected, reconstructed = _aligned_reconstruction_arrays(
        activations=activations,
        decoded=decoded,
    )
    diff = reconstructed - expected
    mse = float(np.mean(diff**2))
    l2_relative = float(np.linalg.norm(diff) / (np.linalg.norm(expected) + 1e-12))
    max_abs_error = float(np.max(np.abs(diff)))
    return mse, l2_relative, max_abs_error


def _compatibility_error(
    *,
    shape_errors: list[str],
    metadata_report: dict[str, Any] | None,
    reconstruction_errors: list[str],
) -> str | None:
    errors: list[str] = []
    errors.extend(shape_errors)
    if metadata_report is not None:
        errors.extend(metadata_report.get("errors", []))
    errors.extend(reconstruction_errors)
    return "; ".join(errors) if errors else None


def verify_sae_compatibility(
    *,
    model_name: str = "EleutherAI/pythia-70m-deduped",
    hook_point: str = "blocks.2.hook_resid_post",
    sae_release: str = "",
    sae_id: str = "",
    device: str | None = "cpu",
    texts: list[str] | None = None,
    out: str | Path | None = None,
    require_metadata_match: bool = True,
    allow_shape_only_diagnostic: bool = False,
    max_reconstruction_l2_relative: float | None = None,
    max_reconstruction_mse: float | None = None,
    model_adapter=None,
    sae_adapter=None,
) -> SAECompatibilityResult:
    check_texts = texts or DEFAULT_CHECK_TEXTS
    if not sae_release or not sae_id:
        result = SAECompatibilityResult(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            activation_shape=[],
            encoded_shape=[],
            decoded_shape=[],
            d_model=0,
            d_sae=0,
            shape_compatible=False,
            metadata_compatible=False,
            reconstruction_compatible=False,
            compatible=False,
            status="error",
            error="sae_release and sae_id are required",
        )
        if out is not None:
            write_config(result.model_dump(), out)
        return result
    activation_shape: list[int] = []
    encoded_shape: list[int] = []
    decoded_shape: list[int] = []
    d_model = 0
    d_sae = 0
    declared_model: str | None = None
    declared_hook_point: str | None = None
    declared_hook_layer: int | None = None
    declared_hook_type: str | None = None
    requested_hook_layer: int | None = None
    requested_hook_type: str | None = None
    metadata_report: dict[str, Any] | None = None
    warnings: list[str] = []
    shape_compatible = False
    metadata_compatible = False
    reconstruction_compatible = False
    reconstruction_mse: float | None = None
    reconstruction_l2_relative: float | None = None
    reconstruction_max_abs_error: float | None = None

    try:
        if model_adapter is None:
            from self_ground.model import TransformerLensModelAdapter

            model = TransformerLensModelAdapter(model_name=model_name, device=device)
        else:
            model = model_adapter
        activations = model.get_activations(check_texts, hook_point=hook_point)
        activation_shape = _shape(activations)
        if len(activation_shape) != 3:
            raise ValueError(
                "activation must have shape [batch, position, d_model]; "
                f"got {activation_shape}"
            )
        d_model = activation_shape[-1]

        if sae_adapter is None:
            from self_ground.sae import SAELensAdapter

            sae = SAELensAdapter.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device or getattr(model, "device", "cpu"),
            )
        else:
            sae = sae_adapter
        identity = extract_sae_identity_metadata(
            sae_adapter=sae,
            sae_release=sae_release,
            sae_id=sae_id,
        )
        declared_model = identity.declared_model
        declared_hook_point = identity.declared_hook_point
        declared_hook_layer = identity.declared_hook_layer
        declared_hook_type = identity.declared_hook_type
        metadata_report = metadata_matches_requested_target(
            metadata=identity,
            requested_model_name=model_name,
            requested_hook_point=hook_point,
            require_metadata=require_metadata_match,
        )
        metadata_compatible = bool(metadata_report["metadata_compatible"])
        requested_hook_layer = metadata_report["requested_hook_layer"]
        requested_hook_type = metadata_report["requested_hook_type"]
        warnings.extend(metadata_report.get("warnings", []))

        encoded = sae.encode(activations)
        encoded_shape = _shape(encoded.values)
        shape_errors: list[str] = []
        encoded_compatible = sae_encoded_shape_is_compatible(activation_shape, encoded_shape)
        if not encoded_compatible:
            shape_errors.append(
                "encoded SAE activations are not shape-compatible with hook activation: "
                f"encoded={encoded_shape}, activation={activation_shape}"
            )
        d_sae = encoded_shape[-1]

        decoded = sae.decode(encoded)
        decoded_shape = _shape(decoded)
        decoded_compatible = sae_shapes_are_compatible(activation_shape, decoded_shape)
        if not decoded_compatible:
            shape_errors.append(
                "decoded SAE output is not shape-compatible with hook activation: "
                f"decoded={decoded_shape}, activation={activation_shape}"
            )
        if identity.d_in is not None and identity.d_in != d_model:
            shape_errors.append(
                f"SAE d_in {identity.d_in} does not match hook d_model {d_model}"
            )
        if identity.d_sae is not None and identity.d_sae != d_sae:
            shape_errors.append(
                f"SAE d_sae {identity.d_sae} does not match encoded width {d_sae}"
            )
        shape_compatible = not shape_errors

        reconstruction_errors: list[str] = []
        if shape_compatible:
            (
                reconstruction_mse,
                reconstruction_l2_relative,
                reconstruction_max_abs_error,
            ) = _compute_reconstruction_metrics(activations=activations, decoded=decoded)
            metrics = [
                reconstruction_mse,
                reconstruction_l2_relative,
                reconstruction_max_abs_error,
            ]
            if not all(np.isfinite(metric) for metric in metrics):
                reconstruction_errors.append("SAE reconstruction metrics contain NaN or Inf")
            if (
                max_reconstruction_l2_relative is not None
                and reconstruction_l2_relative > max_reconstruction_l2_relative
            ):
                reconstruction_errors.append(
                    "SAE reconstruction relative L2 exceeds threshold: "
                    f"{reconstruction_l2_relative} > {max_reconstruction_l2_relative}"
                )
            if max_reconstruction_mse is not None and reconstruction_mse > max_reconstruction_mse:
                reconstruction_errors.append(
                    "SAE reconstruction MSE exceeds threshold: "
                    f"{reconstruction_mse} > {max_reconstruction_mse}"
                )
            if reconstruction_l2_relative > 1.0:
                warnings.append(
                    "SAE reconstruction relative L2 is greater than 1.0; "
                    "reconstruction error is high and should be reviewed"
                )
        else:
            reconstruction_errors.append("shape compatibility failed before reconstruction check")
        reconstruction_compatible = not reconstruction_errors

        error = _compatibility_error(
            shape_errors=shape_errors,
            metadata_report=metadata_report,
            reconstruction_errors=reconstruction_errors,
        )
        compatible = (
            shape_compatible
            and metadata_compatible
            and reconstruction_compatible
            and not allow_shape_only_diagnostic
        )
        status = "ok" if compatible else "error"
        if allow_shape_only_diagnostic:
            status = "shape_only_diagnostic_not_production_compatible"
            compatible = False

        result = SAECompatibilityResult(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            activation_shape=activation_shape,
            encoded_shape=encoded_shape,
            decoded_shape=decoded_shape,
            d_model=d_model,
            d_sae=d_sae,
            shape_compatible=shape_compatible,
            metadata_compatible=metadata_compatible,
            reconstruction_compatible=reconstruction_compatible,
            compatible=compatible,
            status=status,
            error=error,
            declared_model=declared_model,
            declared_hook_point=declared_hook_point,
            declared_hook_layer=declared_hook_layer,
            declared_hook_type=declared_hook_type,
            requested_hook_layer=requested_hook_layer,
            requested_hook_type=requested_hook_type,
            reconstruction_mse=reconstruction_mse,
            reconstruction_l2_relative=reconstruction_l2_relative,
            reconstruction_max_abs_error=reconstruction_max_abs_error,
            metadata_report=metadata_report,
            warnings=warnings,
        )
    except Exception as exc:
        result = SAECompatibilityResult(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            activation_shape=activation_shape,
            encoded_shape=encoded_shape,
            decoded_shape=decoded_shape,
            d_model=d_model,
            d_sae=d_sae,
            shape_compatible=shape_compatible,
            metadata_compatible=metadata_compatible,
            reconstruction_compatible=reconstruction_compatible,
            compatible=False,
            status="error",
            error=str(exc),
            declared_model=declared_model,
            declared_hook_point=declared_hook_point,
            declared_hook_layer=declared_hook_layer,
            declared_hook_type=declared_hook_type,
            requested_hook_layer=requested_hook_layer,
            requested_hook_type=requested_hook_type,
            reconstruction_mse=reconstruction_mse,
            reconstruction_l2_relative=reconstruction_l2_relative,
            reconstruction_max_abs_error=reconstruction_max_abs_error,
            metadata_report=metadata_report,
            warnings=warnings,
        )

    if out is not None:
        write_config(result.model_dump(), out)
    return result
