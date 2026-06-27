from __future__ import annotations

import pandas as pd

from local_mi_lab.metric_calibration_report import (
    render_metric_calibration_learning_note,
    render_metric_calibration_markdown,
)


def test_metric_calibration_report_refuses_mechanism_claims() -> None:
    summary = _summary("metric_needs_revision")
    markdown = render_metric_calibration_markdown(summary, _family_rows())

    assert "does not show an induction head" in markdown
    assert "circuit" in markdown
    assert "Search allowed: `False`" in markdown


def test_learning_note_records_hardest_control() -> None:
    note = render_metric_calibration_learning_note(_summary("metric_needs_revision"), _family_rows())

    assert "Hardest Control" in note
    assert "calib_target_swap_control" in note
    assert "I will not claim an induction head" in note


def test_passing_report_still_limits_claims() -> None:
    markdown = render_metric_calibration_markdown(
        _summary("metric_calibrated_for_next_spec"),
        _family_rows(control_diff=-0.5),
    )

    assert "passed the pre-registered calibration thresholds" in markdown
    assert "Calibration success would only permit a tighter next spec" in markdown


def _summary(status: str) -> dict[str, object]:
    return {
        "status": status,
        "search_allowed": status == "metric_calibrated_for_next_spec",
        "thresholds": {
            "min_positive_minus_max_control_gap": 0.25,
            "min_positive_fraction_diff_positive": 0.8,
            "max_control_fraction_diff_positive": 0.2,
        },
        "positive_mean_true_vs_control_logit_diff": 0.5,
        "max_control_mean_true_vs_control_logit_diff": 0.4,
        "positive_minus_max_control_gap": 0.1,
        "weakest_positive_family_mean": 0.45,
        "hardest_control_family": {
            "family": "calib_target_swap_control",
            "mean_true_vs_control_logit_diff": 0.4,
            "fraction_diff_positive": 0.5,
        },
        "positive_domain_means": {"symbolic": 0.5, "word": 0.4, "number": 0.6},
        "positive_length_means": {"short": 0.5, "medium": 0.4, "long": 0.6},
        "command": "uv run python scripts/run_metric_calibration.py",
    }


def _family_rows(control_diff: float = 0.4) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": "calib_clean_repeat_symbolic",
                "should_show_induction_behavior": True,
                "mean_true_vs_control_logit_diff": 0.5,
                "fraction_diff_positive": 1.0,
                "median_target_rank": 1,
            },
            {
                "family": "calib_target_swap_control",
                "should_show_induction_behavior": False,
                "mean_true_vs_control_logit_diff": control_diff,
                "fraction_diff_positive": 0.5,
                "median_target_rank": 3,
            },
        ]
    )
