from __future__ import annotations

import inspect
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SAEIdentityMetadata:
    sae_release: str
    sae_id: str
    declared_model: str | None
    declared_hook_point: str | None
    declared_hook_layer: int | None
    declared_hook_type: str | None
    d_in: int | None
    d_sae: int | None
    architecture: str | None
    raw_metadata: dict[str, Any]
    missing_fields: list[str]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


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
    if not match:
        raise ValueError(f"unsupported TransformerLens hook point: {hook_point!r}")
    layer = int(match.group("layer"))
    hook_type = match.group("hook_type")
    return {
        "parse_status": "ok",
        "layer": layer,
        "hook_type": hook_type,
        "canonical_hook_point": f"blocks.{layer}.hook_{hook_type}",
    }


def _is_simple(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _safe_call(value: Any) -> Any:
    if not callable(value):
        return value
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return repr(value)
    if signature.parameters:
        return repr(value)
    try:
        return value()
    except Exception:
        return repr(value)


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    elif hasattr(value, "__dict__"):
        items = vars(value).items()
    else:
        return {}

    result: dict[str, Any] = {}
    for key, raw in items:
        if str(key).startswith("_"):
            continue
        resolved = _safe_call(raw)
        if _is_simple(resolved):
            result[str(key)] = resolved
        elif isinstance(resolved, dict):
            result[str(key)] = {
                str(nested_key): nested_value
                for nested_key, nested_value in resolved.items()
                if _is_simple(nested_value)
            }
        else:
            result[str(key)] = repr(resolved)
    return result


def _candidate_metadata_objects(sae_adapter) -> list[tuple[str, Any]]:
    sae = getattr(sae_adapter, "sae", None)
    cfg = getattr(sae, "cfg", None)
    config = getattr(sae, "config", None)
    return [
        ("adapter.metadata", getattr(sae_adapter, "metadata", None)),
        ("adapter.cfg", getattr(sae_adapter, "cfg", None)),
        ("adapter.config", getattr(sae_adapter, "config", None)),
        ("sae.metadata", getattr(sae, "metadata", None)),
        ("sae.cfg.metadata", getattr(cfg, "metadata", None)),
        ("sae.config.metadata", getattr(config, "metadata", None)),
        ("sae.cfg", cfg),
        ("sae.config", config),
        ("sae", sae),
    ]


def _first_value(candidates: list[dict[str, Any]], names: tuple[str, ...]) -> Any:
    for candidate in candidates:
        for name in names:
            value = candidate.get(name)
            if value not in (None, ""):
                return value
    return None


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _extract_hook_parts(
    *,
    declared_hook_point: str | None,
    raw_layer: Any,
    raw_hook_type: Any,
) -> tuple[int | None, str | None]:
    layer = _as_int(raw_layer)
    hook_type = _as_str(raw_hook_type)
    if declared_hook_point:
        try:
            parsed = parse_transformerlens_hook_point(declared_hook_point)
            layer = layer if layer is not None else parsed["layer"]
            hook_type = hook_type or parsed["hook_type"]
        except ValueError:
            pass
    return layer, hook_type


def extract_sae_identity_metadata(
    *,
    sae_adapter,
    sae_release: str,
    sae_id: str,
) -> SAEIdentityMetadata:
    source_dicts: list[dict[str, Any]] = []
    raw_metadata: dict[str, Any] = {}
    for source, value in _candidate_metadata_objects(sae_adapter):
        as_dict = _object_to_dict(value)
        if as_dict:
            raw_metadata[source] = as_dict
            source_dicts.append(as_dict)

    declared_model = _as_str(
        _first_value(
            source_dicts,
            (
                "model_name",
                "model",
                "model_from_pretrained",
                "model_name_or_path",
            ),
        )
    )
    declared_hook_point = _as_str(
        _first_value(
            source_dicts,
            (
                "hook_name",
                "hook_point",
                "hook",
            ),
        )
    )
    raw_layer = _first_value(source_dicts, ("hook_layer", "layer"))
    raw_hook_type = _first_value(source_dicts, ("hook_type", "hook_name_type"))
    declared_hook_layer, declared_hook_type = _extract_hook_parts(
        declared_hook_point=declared_hook_point,
        raw_layer=raw_layer,
        raw_hook_type=raw_hook_type,
    )
    architecture = _as_str(_first_value(source_dicts, ("architecture", "sae_architecture")))
    d_in = _as_int(_first_value(source_dicts, ("d_in", "input_width", "activation_width")))
    d_sae = _as_int(_first_value(source_dicts, ("d_sae", "dict_size", "n_features")))

    missing_fields = []
    for field_name, value in [
        ("declared_model", declared_model),
        ("declared_hook_point", declared_hook_point),
        ("declared_hook_layer", declared_hook_layer),
        ("declared_hook_type", declared_hook_type),
        ("d_in", d_in),
        ("d_sae", d_sae),
        ("architecture", architecture),
    ]:
        if value is None:
            missing_fields.append(field_name)

    return SAEIdentityMetadata(
        sae_release=sae_release,
        sae_id=sae_id,
        declared_model=declared_model,
        declared_hook_point=declared_hook_point,
        declared_hook_layer=declared_hook_layer,
        declared_hook_type=declared_hook_type,
        d_in=d_in,
        d_sae=d_sae,
        architecture=architecture,
        raw_metadata=raw_metadata,
        missing_fields=missing_fields,
    )


def metadata_matches_requested_target(
    *,
    metadata: SAEIdentityMetadata,
    requested_model_name: str,
    requested_hook_point: str,
    require_metadata: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    requested_hook = parse_transformerlens_hook_point(requested_hook_point)
    requested_model = normalize_model_name(requested_model_name)
    declared_model = normalize_model_name(metadata.declared_model)

    missing_required = [
        field_name
        for field_name in ("declared_model", "declared_hook_point")
        if field_name in metadata.missing_fields
    ]
    if missing_required:
        message = f"missing required SAE metadata fields: {missing_required}"
        if require_metadata:
            errors.append(message)
        warnings.append(message)

    model_match = declared_model is not None and requested_model == declared_model
    if declared_model is not None and not model_match:
        errors.append(
            "SAE metadata declares model "
            f"{metadata.declared_model!r}, but requested model is {requested_model_name!r}; "
            "these are different checkpoints"
        )

    declared_hook = None
    if metadata.declared_hook_point is not None:
        try:
            declared_hook = parse_transformerlens_hook_point(metadata.declared_hook_point)
        except ValueError as exc:
            errors.append(str(exc))

    hook_point_match = (
        declared_hook is not None
        and declared_hook["canonical_hook_point"] == requested_hook["canonical_hook_point"]
    )
    if declared_hook is not None and not hook_point_match:
        errors.append(
            "SAE metadata declares hook point "
            f"{metadata.declared_hook_point!r}, but requested hook point is "
            f"{requested_hook_point!r}"
        )

    declared_hook_layer = metadata.declared_hook_layer
    declared_hook_type = metadata.declared_hook_type
    if declared_hook is not None:
        declared_hook_layer = declared_hook["layer"]
        declared_hook_type = declared_hook["hook_type"]

    hook_layer_match = (
        declared_hook_layer is not None and declared_hook_layer == requested_hook["layer"]
    )
    if declared_hook_layer is not None and not hook_layer_match:
        errors.append(
            f"hook layer mismatch: SAE declares {declared_hook_layer}, "
            f"requested {requested_hook['layer']}"
        )

    hook_type_match = (
        declared_hook_type is not None and declared_hook_type == requested_hook["hook_type"]
    )
    if declared_hook_type is not None and not hook_type_match:
        errors.append(
            f"hook type mismatch: SAE declares {declared_hook_type!r}, "
            f"requested {requested_hook['hook_type']!r}"
        )

    metadata_compatible = not errors and model_match and hook_layer_match and hook_type_match
    if metadata.declared_hook_point is not None:
        metadata_compatible = metadata_compatible and hook_point_match

    return {
        "metadata_compatible": bool(metadata_compatible),
        "model_match": bool(model_match),
        "hook_point_match": bool(hook_point_match),
        "hook_layer_match": bool(hook_layer_match),
        "hook_type_match": bool(hook_type_match),
        "declared_model": metadata.declared_model,
        "requested_model": requested_model_name,
        "declared_hook_point": metadata.declared_hook_point,
        "requested_hook_point": requested_hook_point,
        "declared_hook_layer": declared_hook_layer,
        "requested_hook_layer": requested_hook["layer"],
        "declared_hook_type": declared_hook_type,
        "requested_hook_type": requested_hook["hook_type"],
        "missing_metadata_fields": list(metadata.missing_fields),
        "warnings": warnings,
        "errors": errors,
    }
