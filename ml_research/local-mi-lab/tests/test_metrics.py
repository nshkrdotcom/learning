from __future__ import annotations

import pytest

from local_mi_lab.metrics import aggregate_baseline, expected_token_stats


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
