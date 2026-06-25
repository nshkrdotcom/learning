from __future__ import annotations

import re
from typing import Any

from mechledger.assessments.common import AssessmentConditionResult
from mechledger.core.debt import DebtSeverity


def normalize_model_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().lower()
    if not normalized:
        return None
    if "/" in normalized:
        normalized = normalized.rsplit("/", maxsplit=1)[-1]
    return normalized


def parse_transformerlens_hook_point(hook_point: str) -> dict[str, Any]:
    match = re.fullmatch(r"blocks\.(?P<layer>\d+)\.hook_(?P<hook_type>[a-zA-Z0-9_]+)", hook_point)
    if match is None:
        raise ValueError(f"unsupported TransformerLens hook point: {hook_point!r}")
    layer = int(match.group("layer"))
    hook_type = match.group("hook_type")
    return {
        "parse_status": "ok",
        "layer": layer,
        "hook_type": hook_type,
        "canonical_hook_point": f"blocks.{layer}.hook_{hook_type}",
    }


def sae_shapes_are_compatible(activation_shape: list[int], decoded_shape: list[int]) -> bool:
    if len(activation_shape) != 3:
        return False
    if decoded_shape == activation_shape:
        return True
    return (
        len(decoded_shape) == 2
        and decoded_shape[0] == activation_shape[0]
        and decoded_shape[1] == activation_shape[2]
    )


def sae_encoded_shape_is_compatible(
    activation_shape: list[int],
    encoded_shape: list[int],
) -> bool:
    if len(activation_shape) != 3 or len(encoded_shape) not in {2, 3}:
        return False
    if encoded_shape[0] != activation_shape[0]:
        return False
    return not (len(encoded_shape) == 3 and encoded_shape[1] != activation_shape[1])


def metadata_matches_requested_target(
    *,
    metadata: dict[str, Any],
    requested_model_name: str,
    requested_hook_point: str,
    require_metadata: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    requested_hook = parse_transformerlens_hook_point(requested_hook_point)
    requested_model = normalize_model_name(requested_model_name)
    declared_model = normalize_model_name(metadata.get("declared_model"))
    declared_hook_point = metadata.get("declared_hook_point")
    missing_fields = list(metadata.get("missing_fields", []))
    missing_required = [
        field_name
        for field_name in ("declared_model", "declared_hook_point")
        if field_name in missing_fields
    ]
    if missing_required:
        message = f"missing required SAE metadata fields: {missing_required}"
        warnings.append(message)
        if require_metadata:
            errors.append(message)
    model_match = declared_model is not None and requested_model == declared_model
    if declared_model is not None and not model_match:
        errors.append(
            f"SAE metadata declares model {metadata.get('declared_model')!r}, "
            f"but requested model is {requested_model_name!r}"
        )
    declared_hook = None
    if declared_hook_point is not None:
        declared_hook = parse_transformerlens_hook_point(str(declared_hook_point))
    hook_point_match = (
        declared_hook is not None
        and declared_hook["canonical_hook_point"] == requested_hook["canonical_hook_point"]
    )
    if declared_hook is not None and not hook_point_match:
        errors.append(
            f"SAE metadata declares hook point {declared_hook_point!r}, "
            f"but requested hook point is {requested_hook_point!r}"
        )
    declared_layer = (
        declared_hook["layer"]
        if declared_hook is not None
        else _as_int(metadata.get("declared_hook_layer"))
    )
    declared_type = (
        declared_hook["hook_type"]
        if declared_hook is not None
        else metadata.get("declared_hook_type")
    )
    layer_match = declared_layer is not None and declared_layer == requested_hook["layer"]
    hook_type_match = declared_type is not None and declared_type == requested_hook["hook_type"]
    metadata_compatible = (
        not errors and model_match and hook_point_match and layer_match and hook_type_match
    )
    return {
        "metadata_compatible": bool(metadata_compatible),
        "model_match": bool(model_match),
        "hook_point_match": bool(hook_point_match),
        "hook_layer_match": bool(layer_match),
        "hook_type_match": bool(hook_type_match),
        "declared_model": metadata.get("declared_model"),
        "requested_model": requested_model_name,
        "declared_hook_point": declared_hook_point,
        "requested_hook_point": requested_hook_point,
        "declared_hook_layer": declared_layer,
        "requested_hook_layer": requested_hook["layer"],
        "declared_hook_type": declared_type,
        "requested_hook_type": requested_hook["hook_type"],
        "missing_metadata_fields": missing_fields,
        "warnings": warnings,
        "errors": errors,
    }


def evaluate_compatibility_record(record: dict[str, Any]) -> AssessmentConditionResult:
    compatible = bool(record.get("compatible")) and not bool(record.get("diagnostic_only"))
    return AssessmentConditionResult(
        condition_id="metadata_compatibility",
        condition_type="no_metadata_override",
        passed=compatible,
        parameters={
            "compatible": record.get("compatible"),
            "diagnostic_only": record.get("diagnostic_only"),
        },
        failure_message=str(record.get("error") or "Compatibility failed or is diagnostic-only."),
        default_consequence="blocker",
        debt_type="metadata_mismatch_override",
        severity=DebtSeverity.BLOCKING,
    )


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
