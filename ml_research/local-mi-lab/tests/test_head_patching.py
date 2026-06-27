from __future__ import annotations

import pandas as pd
import pytest
import torch

from local_mi_lab.head_hooks import HeadPatchSite
from local_mi_lab.head_patching import (
    aggregate_head_patching_by_family,
    aggregate_head_patching_by_head,
    apply_head_intervention,
    classify_head_specificity,
    parse_heads,
    resolve_position_index_for_record,
)
from local_mi_lab.types import PromptRecord


def test_patches_only_selected_head_dimension_in_hook_z_tensor() -> None:
    site = _site(head_specific=True)
    corrupt = torch.zeros(1, 3, 2, 4)
    clean = torch.ones(1, 3, 2, 4)
    patched = apply_head_intervention(
        corrupt,
        clean,
        site=site,
        head=1,
        clean_position=2,
        corrupt_position=2,
        intervention="head_clean_to_corrupt_patch",
    )
    assert torch.all(patched[:, 2, 1, :] == 1)
    assert torch.all(patched[:, 2, 0, :] == 0)
    assert torch.all(patched[:, 0, :, :] == 0)


def test_zero_ablation_zeros_only_selected_head() -> None:
    site = _site(head_specific=True)
    corrupt = torch.ones(1, 3, 2, 4)
    patched = apply_head_intervention(
        corrupt,
        None,
        site=site,
        head=0,
        clean_position=1,
        corrupt_position=1,
        intervention="head_zero_ablation",
    )
    assert torch.all(patched[:, 1, 0, :] == 0)
    assert torch.all(patched[:, 1, 1, :] == 1)


def test_mean_ablation_replaces_only_selected_head() -> None:
    site = _site(head_specific=True)
    corrupt = torch.ones(1, 3, 2, 4)
    mean_act = torch.arange(4, dtype=corrupt.dtype)
    patched = apply_head_intervention(
        corrupt,
        None,
        site=site,
        head=1,
        clean_position=1,
        corrupt_position=1,
        intervention="head_mean_ablation",
        mean_act=mean_act,
    )
    assert torch.all(patched[:, 1, 1, :] == mean_act)
    assert torch.all(patched[:, 1, 0, :] == 1)


def test_unsupported_intervention_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unsupported intervention"):
        apply_head_intervention(
            torch.ones(1, 3, 2, 4),
            None,
            site=_site(head_specific=True),
            head=0,
            clean_position=1,
            corrupt_position=1,
            intervention="unknown",
        )


def test_full_attn_out_fallback_is_marked_not_head_specific() -> None:
    status = classify_head_specificity(
        _results_df(),
        head_specific_patch=False,
        positive_mean=0.5,
        max_control_mean=0.1,
        gap=0.4,
    )
    assert status == "not_head_specific"


def test_aggregation_by_family_and_head() -> None:
    by_family = aggregate_head_patching_by_family(_results_df(control_effect=0.2))
    assert set(by_family["family"]) == {"positive_repeat_sequence", "distractor_repeat_control"}
    by_head = aggregate_head_patching_by_head(_results_df(control_effect=0.2))
    row = by_head.iloc[0]
    assert row["positive_mean_effect_size"] == 0.6
    assert row["max_control_mean_effect_size"] == 0.2
    assert row["positive_minus_control_effect_gap"] == 0.39999999999999997
    assert row["specificity_status"] == "head_specific_positive_candidate"


def test_nonspecific_classification_when_controls_move_as_much() -> None:
    status = classify_head_specificity(
        _results_df(control_effect=0.7),
        head_specific_patch=True,
        positive_mean=0.6,
        max_control_mean=0.7,
        gap=-0.1,
    )
    assert status == "nonspecific_moves_controls"


def test_denominator_zero_handling() -> None:
    df = _results_df()
    df["effect_size_status"] = "denominator_zero"
    df["effect_size"] = None
    status = classify_head_specificity(
        df,
        head_specific_patch=True,
        positive_mean=None,
        max_control_mean=None,
        gap=None,
    )
    assert status == "denominator_problem"


def test_parse_heads_deduplicates_and_sorts() -> None:
    assert parse_heads("L2H3,L0H1,L2H3") == [(0, 1), (2, 3)]


def test_previous_occurrence_position_status_is_explicit() -> None:
    record = PromptRecord(
        example_id="p0",
        task="induction_heldout",
        family="heldout_symbolic_longer",
        prompt="A B C D E F A B C D E",
        expected_next_token=" F",
        control_prompt="A C E B D F C A E B D",
        notes="",
        prompt_tokens_text=["A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E"],
        expected_source_position_hint=4,
    )
    position, status = resolve_position_index_for_record(record, "previous_occurrence", 12)
    assert position == 5
    assert status == "ok"

    control = PromptRecord(
        example_id="c0",
        task="induction_heldout",
        family="heldout_no_structure_same_tokens",
        prompt="A C E B D F C A E B D",
        expected_next_token=" F",
        control_prompt="A B C D E F A B C D E",
        notes="",
        prompt_tokens_text=["A", "C", "E", "B", "D", "F", "C", "A", "E", "B", "D"],
        expected_source_position_hint=None,
        should_show_induction_behavior=False,
    )
    position, status = resolve_position_index_for_record(control, "previous_occurrence", 12)
    assert position is None
    assert status == "unavailable_for_family"


def _site(head_specific: bool) -> HeadPatchSite:
    return HeadPatchSite(
        hook_name="blocks.0.attn.hook_z" if head_specific else "blocks.0.hook_attn_out",
        head_specific_possible=head_specific,
        head_axis=2 if head_specific else None,
        seq_axis=1,
        feature_axis=3 if head_specific else 2,
        actual_patch_scope="single_head_z" if head_specific else "full_attn_out_layer",
    )


def _results_df(control_effect: float = 0.2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("p0", "positive_repeat_sequence", True, 0.6),
            _row("p1", "positive_repeat_sequence", True, 0.6),
            _row("c0", "distractor_repeat_control", False, control_effect),
            _row("c1", "distractor_repeat_control", False, control_effect),
        ]
    )


def _row(example_id: str, family: str, should_show: bool, effect_size: float | None) -> dict[str, object]:
    return {
        "run_id": "run",
        "seed": 0,
        "example_id": example_id,
        "family": family,
        "control_family": "" if should_show else family,
        "should_show_induction_behavior": should_show,
        "layer": 0,
        "head": 1,
        "hook_name": "blocks.0.attn.hook_z",
        "head_specific_patch": True,
        "actual_patch_scope": "single_head_z",
        "intervention": "head_clean_to_corrupt_patch",
        "position_label": "final",
        "patch_position": 6,
        "clean_prompt": "A B C D A B C",
        "corrupt_prompt": "A B C D A B X",
        "true_expected_next_token": " D",
        "wrong_or_control_token": " X",
        "metric": "true_vs_control_logit_diff",
        "clean_score": 3.0,
        "corrupt_score": 1.0,
        "patched_score": 2.2,
        "effect_size": effect_size,
        "effect_size_status": "ok" if effect_size is not None else "denominator_zero",
    }
