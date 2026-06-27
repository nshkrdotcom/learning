from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy_1d(values: Any) -> np.ndarray:
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected one-dimensional logits, got shape {array.shape}")
    return array


def softmax(values: Any) -> np.ndarray:
    array = to_numpy_1d(values)
    shifted = array - np.max(array)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def expected_token_stats(logits: Any, target_token_id: int) -> dict[str, float | int]:
    array = to_numpy_1d(logits)
    if target_token_id < 0 or target_token_id >= array.shape[0]:
        raise ValueError(f"target_token_id {target_token_id} out of range for vocab {array.shape[0]}")
    probs = softmax(array)
    target_logit = float(array[target_token_id])
    rank = int(1 + np.sum(array > target_logit))
    return {
        "target_token_id": int(target_token_id),
        "target_logit": target_logit,
        "target_probability": float(probs[target_token_id]),
        "target_rank": rank,
    }


def aggregate_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_examples": 0,
            "mean_expected_probability": None,
            "median_expected_rank": None,
            "mean_logit_diff_vs_control": None,
            "mean_probability_diff_vs_control": None,
            "n_rank_at_most_10": 0,
            "failing_examples": [],
        }
    probabilities = np.array([float(row["expected_probability"]) for row in rows])
    ranks = np.array([int(row["expected_rank"]) for row in rows])
    logit_diffs = np.array([float(row["logit_diff_vs_control"]) for row in rows])
    prob_diffs = np.array([float(row["probability_diff_vs_control"]) for row in rows])
    failing = [
        str(row["example_id"])
        for row in rows
        if int(row["expected_rank"]) > 10 or float(row["probability_diff_vs_control"]) <= 0
    ]
    return {
        "n_examples": len(rows),
        "mean_expected_probability": float(np.mean(probabilities)),
        "median_expected_rank": float(np.median(ranks)),
        "mean_expected_rank": float(np.mean(ranks)),
        "mean_logit_diff_vs_control": float(np.mean(logit_diffs)),
        "mean_probability_diff_vs_control": float(np.mean(prob_diffs)),
        "n_rank_at_most_10": int(np.sum(ranks <= 10)),
        "failing_examples": failing,
    }


def aggregate_baseline_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(row.get("family", "positive_repeat_sequence")) for row in rows})
    family_rows: list[dict[str, Any]] = []
    for family in families:
        subset = [row for row in rows if str(row.get("family", "positive_repeat_sequence")) == family]
        summary = aggregate_baseline(subset)
        should_values = {bool(row.get("should_show_induction_behavior", True)) for row in subset}
        family_rows.append(
            {
                "family": family,
                "n_examples": summary["n_examples"],
                "mean_expected_probability": summary["mean_expected_probability"],
                "median_expected_rank": summary["median_expected_rank"],
                "mean_logit_diff_vs_control": summary["mean_logit_diff_vs_control"],
                "mean_probability_diff_vs_control": summary["mean_probability_diff_vs_control"],
                "n_rank_at_most_10": summary["n_rank_at_most_10"],
                "fraction_rank_at_most_10": (
                    int(summary["n_rank_at_most_10"]) / int(summary["n_examples"])
                    if summary["n_examples"]
                    else 0.0
                ),
                "should_show_induction_behavior": should_values == {True},
            }
        )
    return family_rows


def positive_vs_control_gap(family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [
        row for row in family_rows if bool(row.get("should_show_induction_behavior", False))
    ]
    control_rows = [
        row for row in family_rows if not bool(row.get("should_show_induction_behavior", False))
    ]
    if not positive_rows or not control_rows:
        return {
            "positive_mean_expected_probability": None,
            "max_control_mean_expected_probability": None,
            "gap_mean_expected_probability": None,
            "positive_fraction_rank_at_most_10": None,
            "max_control_fraction_rank_at_most_10": None,
            "gap_fraction_rank_at_most_10": None,
        }
    positive_mean = float(
        np.mean([float(row["mean_expected_probability"]) for row in positive_rows])
    )
    max_control_mean = max(float(row["mean_expected_probability"]) for row in control_rows)
    positive_fraction = float(
        np.mean([float(row["fraction_rank_at_most_10"]) for row in positive_rows])
    )
    max_control_fraction = max(float(row["fraction_rank_at_most_10"]) for row in control_rows)
    return {
        "positive_mean_expected_probability": positive_mean,
        "max_control_mean_expected_probability": max_control_mean,
        "gap_mean_expected_probability": positive_mean - max_control_mean,
        "positive_fraction_rank_at_most_10": positive_fraction,
        "max_control_fraction_rank_at_most_10": max_control_fraction,
        "gap_fraction_rank_at_most_10": positive_fraction - max_control_fraction,
    }


def hardest_control_family(family_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    control_rows = [
        row for row in family_rows if not bool(row.get("should_show_induction_behavior", False))
    ]
    if not control_rows:
        return None
    return max(control_rows, key=lambda row: float(row["mean_expected_probability"]))


def controlled_baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_rows = aggregate_baseline_by_family(rows)
    positive_rows = [row for row in rows if bool(row.get("should_show_induction_behavior", True))]
    control_rows = [row for row in rows if not bool(row.get("should_show_induction_behavior", True))]
    return {
        "overall_summary": aggregate_baseline(rows),
        "positive_family_summary": aggregate_baseline(positive_rows),
        "control_family_summary": aggregate_baseline(control_rows),
        "positive_vs_control_gap": positive_vs_control_gap(family_rows),
        "hardest_control_family": hardest_control_family(family_rows),
        "by_family": family_rows,
    }


def behavior_is_worth_activation_analysis(summary: dict[str, Any]) -> bool:
    if not summary or not summary.get("n_examples"):
        return False
    rank_hits = int(summary.get("n_rank_at_most_10") or 0)
    n_examples = int(summary.get("n_examples") or 0)
    mean_prob_diff = float(summary.get("mean_probability_diff_vs_control") or 0.0)
    return rank_hits > 0 and mean_prob_diff > 0 and rank_hits / max(n_examples, 1) >= 0.10
