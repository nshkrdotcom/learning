from __future__ import annotations

import pytest

from local_mi_lab.metrics import (
    aggregate_baseline,
    aggregate_baseline_by_family,
    controlled_baseline_summary,
    expected_token_stats,
    hardest_control_family,
    positive_vs_control_gap,
)


def test_expected_token_rank_probability_math() -> None:
    stats = expected_token_stats([1.0, 3.0, 2.0], target_token_id=1)
    assert stats["target_rank"] == 1
    assert stats["target_probability"] == pytest.approx(0.6652409557)


def test_expected_token_rank_for_second_best() -> None:
    stats = expected_token_stats([1.0, 3.0, 2.0], target_token_id=2)
    assert stats["target_rank"] == 2


def test_baseline_aggregation() -> None:
    rows = [
        {
            "example_id": "a",
            "expected_probability": 0.5,
            "expected_rank": 1,
            "logit_diff_vs_control": 2.0,
            "probability_diff_vs_control": 0.2,
        },
        {
            "example_id": "b",
            "expected_probability": 0.25,
            "expected_rank": 12,
            "logit_diff_vs_control": -1.0,
            "probability_diff_vs_control": -0.1,
        },
    ]
    summary = aggregate_baseline(rows)
    assert summary["n_examples"] == 2
    assert summary["mean_expected_probability"] == pytest.approx(0.375)
    assert summary["median_expected_rank"] == pytest.approx(6.5)
    assert summary["n_rank_at_most_10"] == 1
    assert summary["failing_examples"] == ["b"]


def test_baseline_aggregation_by_family() -> None:
    rows = [
        _row("p1", "positive_repeat_sequence", True, 0.8, 1, 1.0),
        _row("p2", "positive_repeat_sequence", True, 0.6, 2, 0.8),
        _row("c1", "shuffled_repeat_control", False, 0.2, 20, 0.1),
    ]
    by_family = aggregate_baseline_by_family(rows)
    positive = next(row for row in by_family if row["family"] == "positive_repeat_sequence")
    assert positive["n_examples"] == 2
    assert positive["mean_expected_probability"] == pytest.approx(0.7)
    assert positive["fraction_rank_at_most_10"] == pytest.approx(1.0)


def test_positive_vs_control_gap_calculation() -> None:
    family_rows = [
        {
            "family": "positive_repeat_sequence",
            "mean_expected_probability": 0.7,
            "fraction_rank_at_most_10": 1.0,
            "should_show_induction_behavior": True,
        },
        {
            "family": "shuffled_repeat_control",
            "mean_expected_probability": 0.2,
            "fraction_rank_at_most_10": 0.25,
            "should_show_induction_behavior": False,
        },
    ]
    gap = positive_vs_control_gap(family_rows)
    assert gap["gap_mean_expected_probability"] == pytest.approx(0.5)
    assert gap["gap_fraction_rank_at_most_10"] == pytest.approx(0.75)


def test_hardest_control_family_selection() -> None:
    family_rows = [
        {
            "family": "positive_repeat_sequence",
            "mean_expected_probability": 0.7,
            "should_show_induction_behavior": True,
        },
        {
            "family": "weak_control",
            "mean_expected_probability": 0.1,
            "should_show_induction_behavior": False,
        },
        {
            "family": "hard_control",
            "mean_expected_probability": 0.4,
            "should_show_induction_behavior": False,
        },
    ]
    assert hardest_control_family(family_rows)["family"] == "hard_control"


def test_controlled_baseline_summary_contains_gap() -> None:
    rows = [
        _row("p1", "positive_repeat_sequence", True, 0.8, 1, 1.0),
        _row("c1", "shuffled_repeat_control", False, 0.2, 20, 0.1),
    ]
    summary = controlled_baseline_summary(rows)
    assert summary["positive_vs_control_gap"]["gap_mean_expected_probability"] == pytest.approx(
        0.6
    )
    assert summary["hardest_control_family"]["family"] == "shuffled_repeat_control"


def _row(
    example_id: str,
    family: str,
    should_show: bool,
    probability: float,
    rank: int,
    logit_diff: float,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "family": family,
        "should_show_induction_behavior": should_show,
        "expected_probability": probability,
        "expected_rank": rank,
        "logit_diff_vs_control": logit_diff,
        "probability_diff_vs_control": probability / 10,
    }
