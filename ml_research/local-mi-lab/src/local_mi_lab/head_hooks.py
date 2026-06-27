from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HeadPatchSite:
    hook_name: str
    head_specific_possible: bool
    head_axis: int | None
    seq_axis: int | None
    feature_axis: int | None
    actual_patch_scope: str


def hook_z_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_z"


def hook_result_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_result"


def hook_attn_out_name(layer: int) -> str:
    return f"blocks.{layer}.hook_attn_out"


def hook_pattern_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_pattern"


def candidate_hook_names(layer: int) -> list[str]:
    return [
        hook_z_name(layer),
        hook_result_name(layer),
        hook_attn_out_name(layer),
        hook_pattern_name(layer),
    ]


def inspect_head_hooks(model: Any, prompt: str, layers: list[int]) -> list[dict[str, Any]]:
    _enable_attn_result_if_available(model)
    tokens = model.to_tokens(prompt)
    names = [name for layer in layers for name in candidate_hook_names(layer)]
    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens, names_filter=names)
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for hook_name in candidate_hook_names(layer):
            tensor = _cache_get(cache, hook_name)
            rows.append(describe_hook_tensor(hook_name, tensor))
    return rows


def describe_hook_tensor(hook_name: str, tensor: Any | None) -> dict[str, Any]:
    if tensor is None:
        return {
            "hook_name": hook_name,
            "exists_or_capturable": False,
            "tensor_shape": None,
            "interpretation": "not captured",
            "can_index_head_dimension": False,
            "head_dimension_axis": None,
            "seq_axis": None,
            "d_head_or_hidden_axis": None,
            "notes": "Hook was not captured for this model/run.",
        }
    shape = [int(dim) for dim in tensor.shape]
    metadata = shape_metadata_for_hook(hook_name, shape)
    return {
        "hook_name": hook_name,
        "exists_or_capturable": True,
        "tensor_shape": shape,
        **metadata,
    }


def shape_metadata_for_hook(hook_name: str, shape: list[int]) -> dict[str, Any]:
    if hook_name.endswith(".attn.hook_z") and len(shape) == 4:
        return {
            "interpretation": "attention head z output",
            "can_index_head_dimension": True,
            "head_dimension_axis": 2,
            "seq_axis": 1,
            "d_head_or_hidden_axis": 3,
            "notes": "Preferred head-specific patch site.",
        }
    if hook_name.endswith(".attn.hook_result") and len(shape) == 4:
        return {
            "interpretation": "attention head result output",
            "can_index_head_dimension": True,
            "head_dimension_axis": 2,
            "seq_axis": 1,
            "d_head_or_hidden_axis": 3,
            "notes": "Head-specific fallback patch site.",
        }
    if hook_name.endswith(".hook_attn_out") and len(shape) == 3:
        return {
            "interpretation": "full attention layer output after heads are combined",
            "can_index_head_dimension": False,
            "head_dimension_axis": None,
            "seq_axis": 1,
            "d_head_or_hidden_axis": 2,
            "notes": "Layer-level fallback only; not head-specific.",
        }
    if hook_name.endswith(".attn.hook_pattern") and len(shape) == 4:
        return {
            "interpretation": "attention pattern",
            "can_index_head_dimension": True,
            "head_dimension_axis": 1,
            "seq_axis": 2,
            "d_head_or_hidden_axis": 3,
            "notes": "Descriptive pattern hook, not an output patch site.",
        }
    return {
        "interpretation": "unsupported or unexpected hook shape",
        "can_index_head_dimension": False,
        "head_dimension_axis": None,
        "seq_axis": None,
        "d_head_or_hidden_axis": None,
        "notes": f"Unexpected shape {shape} for {hook_name}.",
    }


def resolve_head_patch_site(model: Any, layer: int, preferred: str = "hook_z") -> dict[str, Any]:
    rows = inspect_head_hooks(model, "A B C", [layer])
    return asdict(resolve_head_patch_site_from_metadata(rows, layer, preferred=preferred))


def resolve_head_patch_site_from_metadata(
    rows: list[dict[str, Any]],
    layer: int,
    preferred: str = "hook_z",
) -> HeadPatchSite:
    by_name = {str(row["hook_name"]): row for row in rows}
    preferred_names = _preferred_hook_names(layer, preferred)
    for hook_name in preferred_names:
        row = by_name.get(hook_name)
        if row and row.get("exists_or_capturable") and row.get("can_index_head_dimension"):
            scope = "single_head_z" if hook_name.endswith(".attn.hook_z") else "single_head_result"
            return HeadPatchSite(
                hook_name=hook_name,
                head_specific_possible=True,
                head_axis=int(row["head_dimension_axis"]),
                seq_axis=int(row["seq_axis"]),
                feature_axis=int(row["d_head_or_hidden_axis"]),
                actual_patch_scope=scope,
            )
    attn_out = by_name.get(hook_attn_out_name(layer))
    if attn_out and attn_out.get("exists_or_capturable"):
        return HeadPatchSite(
            hook_name=hook_attn_out_name(layer),
            head_specific_possible=False,
            head_axis=None,
            seq_axis=int(attn_out["seq_axis"]),
            feature_axis=int(attn_out["d_head_or_hidden_axis"]),
            actual_patch_scope="full_attn_out_layer",
        )
    return HeadPatchSite(
        hook_name="",
        head_specific_possible=False,
        head_axis=None,
        seq_axis=None,
        feature_axis=None,
        actual_patch_scope="unsupported",
    )


def _preferred_hook_names(layer: int, preferred: str) -> list[str]:
    if preferred == "hook_result":
        return [hook_result_name(layer), hook_z_name(layer)]
    return [hook_z_name(layer), hook_result_name(layer)]


def _enable_attn_result_if_available(model: Any) -> None:
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)


def _cache_get(cache: Any, hook_name: str) -> Any | None:
    cache_dict = getattr(cache, "cache_dict", None)
    if isinstance(cache_dict, dict):
        return cache_dict.get(hook_name)
    try:
        return cache[hook_name]
    except KeyError:
        return None
