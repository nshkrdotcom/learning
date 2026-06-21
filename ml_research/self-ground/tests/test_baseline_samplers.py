from __future__ import annotations

import csv

import pytest

from self_ground.baseline_samplers import (
    MatchedControlConfig,
    load_feature_activation_stats_from_ranking,
    select_activation_density_matched_features,
    select_multiple_activation_density_matched_feature_sets,
)


def _write_density_ranking(path, *, off_by: float = 0.0) -> None:
    rows = [
        ("sae_0", 10.0, [1.0, 1.0, 1.0, 1.0]),
        ("sae_1", 9.0, [1.0, 1.0, 1.0, 1.0]),
        ("sae_2", 3.0, [1.0 + off_by, 1.0 + off_by, 1.0 + off_by, 1.0 + off_by]),
        ("sae_3", 2.0, [1.0 + off_by, 1.0 + off_by, 1.0 + off_by, 1.0 + off_by]),
        ("sae_4", 1.0, [0.1, 0.0, 0.0, 0.0]),
        ("sae_5", 0.8, [0.2, 0.0, 0.0, 0.0]),
        ("sae_6", 0.5, [1.0 + off_by, 1.0 + off_by, 1.0 + off_by, 1.0 + off_by]),
        ("sae_7", 0.3, [1.0 + off_by, 1.0 + off_by, 1.0 + off_by, 1.0 + off_by]),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_id",
                "score",
                "mean_pos",
                "mean_neg",
                "mean_para",
                "mean_decoy",
                "abs_score",
            ],
        )
        writer.writeheader()
        for feature_id, score, means in rows:
            writer.writerow(
                {
                    "feature_id": feature_id,
                    "score": score,
                    "mean_pos": means[0],
                    "mean_neg": means[1],
                    "mean_para": means[2],
                    "mean_decoy": means[3],
                    "abs_score": abs(score),
                }
            )


def test_load_stats_records_per_condition_approximation(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_density_ranking(ranking)

    stats = load_feature_activation_stats_from_ranking(ranking)

    assert stats[0].feature_id == "sae_0"
    assert stats[0].activation_abs_mean == 1.0
    assert stats[0].activation_nonzero_fraction == 1.0
    assert stats[0].stats_source == "per_condition_mean_approximation"


def test_density_matched_selection_is_deterministic_and_excludes_top(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_density_ranking(ranking)

    config = MatchedControlConfig(top_k=2, seed=7)
    first = select_activation_density_matched_features(
        ranking,
        top_feature_ids=["sae_0", "sae_1"],
        config=config,
    )
    second = select_activation_density_matched_features(
        ranking,
        top_feature_ids=["sae_0", "sae_1"],
        config=config,
    )

    assert first.feature_ids == second.feature_ids
    assert not set(first.feature_ids) & {"sae_0", "sae_1"}
    assert first.control_stats_summary["activation_abs_mean"] == pytest.approx(
        first.top_stats_summary["activation_abs_mean"]
    )
    assert first.stats_source == "per_condition_mean_approximation"


def test_multiple_density_matched_sets_have_expected_labels(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_density_ranking(ranking)

    results = select_multiple_activation_density_matched_feature_sets(
        ranking,
        top_feature_ids=["sae_0", "sae_1"],
        top_k=2,
        seeds=[7, 11],
    )

    assert set(results) == {"density_matched_seed_7", "density_matched_seed_11"}
    assert all(
        result.selection_method == "activation_density_matched"
        for result in results.values()
    )


def test_relaxation_metadata_is_recorded_when_strict_tolerance_fails(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_density_ranking(ranking, off_by=0.25)

    result = select_activation_density_matched_features(
        ranking,
        top_feature_ids=["sae_0", "sae_1"],
        config=MatchedControlConfig(
            top_k=2,
            seed=7,
            density_tolerance=0.01,
            abs_mean_tolerance=0.01,
            allow_relaxed_tolerance=True,
        ),
    )

    assert result.relaxed is True
    assert result.tolerance_used["abs_mean_tolerance"] > 0.01


def test_strict_matching_failure_is_explicit(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_density_ranking(ranking, off_by=0.25)

    with pytest.raises(ValueError, match="insufficient activation-density matched controls"):
        select_activation_density_matched_features(
            ranking,
            top_feature_ids=["sae_0", "sae_1"],
            config=MatchedControlConfig(
                top_k=2,
                seed=7,
                density_tolerance=0.01,
                abs_mean_tolerance=0.01,
                allow_relaxed_tolerance=False,
            ),
        )
