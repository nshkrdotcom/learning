from __future__ import annotations

import pytest

from local_mi_lab.patching import patching_effect_size


def test_patching_effect_calculation() -> None:
    assert patching_effect_size(clean_score=5.0, corrupt_score=1.0, patched_score=3.0) == pytest.approx(
        0.5
    )


def test_patching_effect_handles_zero_denominator() -> None:
    assert patching_effect_size(clean_score=1.0, corrupt_score=1.0, patched_score=2.0) == 0.0
