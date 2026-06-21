from __future__ import annotations

import pytest

from self_ground.ravel_adapter.scoring import (
    cause_minus_isolation,
    cause_score,
    isolation_score,
    isolation_to_cause_ratio,
    summarize_negation_ravel_scores,
)


def test_specificity_gap_is_cause_minus_isolation_alias() -> None:
    assert cause_score(0.25) == 0.25
    assert isolation_score(0.05) == 0.05
    assert cause_minus_isolation(0.25, 0.05) == pytest.approx(0.2)


def test_collateral_ratio_is_isolation_to_cause_alias() -> None:
    assert isolation_to_cause_ratio(0.25, 0.05) == pytest.approx(0.2)
    assert isolation_to_cause_ratio(0.0, 0.05) is None


def test_summary_uses_ravel_shaped_names() -> None:
    scores = summarize_negation_ravel_scores(
        target_absolute_delta=0.3,
        control_absolute_delta=0.1,
    )

    assert scores.cause_score == 0.3
    assert scores.isolation_score == 0.1
    assert scores.cause_minus_isolation == pytest.approx(0.2)
    assert scores.isolation_to_cause_ratio == pytest.approx(1 / 3)
