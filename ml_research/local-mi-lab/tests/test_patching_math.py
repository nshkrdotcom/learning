from __future__ import annotations

import pytest

from local_mi_lab.patching import (
    patching_effect,
    patching_effect_size,
    resolve_patch_positions,
    validate_clean_corrupt_token_lengths,
)


def test_patching_effect_calculation() -> None:
    assert patching_effect_size(clean_score=5.0, corrupt_score=1.0, patched_score=3.0) == pytest.approx(
        0.5
    )
    assert patching_effect(clean_score=5.0, corrupt_score=1.0, patched_score=3.0) == {
        "effect_size": pytest.approx(0.5),
        "effect_size_status": "ok",
    }


def test_patching_effect_marks_zero_denominator_undefined() -> None:
    assert patching_effect_size(clean_score=1.0, corrupt_score=1.0, patched_score=2.0) is None
    assert patching_effect(clean_score=1.0, corrupt_score=1.0, patched_score=2.0) == {
        "effect_size": None,
        "effect_size_status": "denominator_zero",
    }


def test_mismatched_token_lengths_fail_unless_allowed() -> None:
    with pytest.raises(ValueError, match="same length"):
        validate_clean_corrupt_token_lengths(5, 6, allow_length_mismatch=False)
    validate_clean_corrupt_token_lengths(5, 6, allow_length_mismatch=True)


def test_invalid_patch_positions_fail_clearly() -> None:
    with pytest.raises(ValueError, match="not valid"):
        resolve_patch_positions([5], clean_seq_len=5, corrupt_seq_len=6, allow_length_mismatch=True)


def test_patch_positions_resolve_when_valid() -> None:
    assert resolve_patch_positions(["final"], 5, 5, allow_length_mismatch=False) == [4]
