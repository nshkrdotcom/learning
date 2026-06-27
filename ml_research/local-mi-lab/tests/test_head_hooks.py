from __future__ import annotations

import torch

from local_mi_lab.head_hooks import (
    describe_hook_tensor,
    hook_attn_out_name,
    hook_result_name,
    hook_z_name,
    resolve_head_patch_site_from_metadata,
)


def test_hook_z_shape_is_head_specific() -> None:
    row = describe_hook_tensor(hook_z_name(0), torch.zeros(1, 7, 12, 64))
    assert row["can_index_head_dimension"] is True
    assert row["head_dimension_axis"] == 2
    assert row["seq_axis"] == 1
    site = resolve_head_patch_site_from_metadata([row], layer=0)
    assert site.head_specific_possible is True
    assert site.actual_patch_scope == "single_head_z"


def test_hook_result_shape_is_head_specific_fallback() -> None:
    row = describe_hook_tensor(hook_result_name(1), torch.zeros(1, 7, 12, 768))
    site = resolve_head_patch_site_from_metadata([row], layer=1)
    assert site.head_specific_possible is True
    assert site.actual_patch_scope == "single_head_result"
    assert site.head_axis == 2


def test_hook_attn_out_shape_is_not_head_specific() -> None:
    row = describe_hook_tensor(hook_attn_out_name(2), torch.zeros(1, 7, 768))
    site = resolve_head_patch_site_from_metadata([row], layer=2)
    assert site.head_specific_possible is False
    assert site.actual_patch_scope == "full_attn_out_layer"
    assert site.head_axis is None


def test_unsupported_hook_returns_unsupported_scope() -> None:
    row = describe_hook_tensor("blocks.3.mlp.hook_post", torch.zeros(1, 7, 3072))
    site = resolve_head_patch_site_from_metadata([row], layer=3)
    assert site.head_specific_possible is False
    assert site.actual_patch_scope == "unsupported"


def test_preferred_hook_result_order_is_explicit() -> None:
    rows = [
        describe_hook_tensor(hook_z_name(4), torch.zeros(1, 7, 12, 64)),
        describe_hook_tensor(hook_result_name(4), torch.zeros(1, 7, 12, 768)),
    ]
    site = resolve_head_patch_site_from_metadata(rows, layer=4, preferred="hook_result")
    assert site.hook_name == hook_result_name(4)
    assert site.feature_axis == 3
