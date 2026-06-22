from __future__ import annotations

import csv

import pytest

from self_ground.baselines import (
    build_feature_sets,
    select_bottom_active_features,
    select_features_by_mode,
    select_multiple_seeded_random_feature_sets,
    select_seeded_random_features,
    select_top_features,
)


def _write_ranking(path, prefix: str = "sae_") -> None:
    rows = [
        ("0", 10.0, [1.0, 1.0, 1.0, 1.0]),
        ("1", -8.0, [1.0, 1.0, 1.0, 1.0]),
        ("2", 0.0, [0.1, 0.0, 0.0, 0.0]),
        ("3", 0.3, [1.0, 1.0, 1.0, 1.0]),
        ("4", -0.2, [1.0, 1.0, 1.0, 1.0]),
        ("5", 0.1, [0.2, 0.0, 0.0, 0.0]),
        ("6", 0.05, [1.0, 1.0, 1.0, 1.0]),
        ("7", -0.04, [1.0, 1.0, 1.0, 1.0]),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_id",
                "score",
                "abs_score",
                "mean_pos",
                "mean_neg",
                "mean_para",
                "mean_decoy",
            ],
        )
        writer.writeheader()
        for suffix, score, means in rows:
            writer.writerow(
                {
                    "feature_id": f"{prefix}{suffix}",
                    "score": score,
                    "abs_score": abs(score),
                    "mean_pos": means[0],
                    "mean_neg": means[1],
                    "mean_para": means[2],
                    "mean_decoy": means[3],
                }
            )


def _write_rich_ranking(path) -> None:
    rows = [
        ("sae_0", 10.0, 0.8, 0.1, 0.7, 8.0, 3, 0.9, 0.3),
        ("sae_1", -8.0, 0.2, 0.9, -0.7, 0.2, 0, 0.8, 0.9),
        ("sae_2", 3.0, 0.7, 0.2, 0.5, 3.5, 2, 0.7, 0.4),
        ("sae_3", 2.0, 0.4, 0.05, 0.35, 8.0, 1, 0.6, 0.1),
        ("sae_4", 1.0, 0.3, 0.02, 0.28, 15.0, 3, 0.5, 0.05),
    ]
    fieldnames = [
        "feature_id",
        "score",
        "abs_score",
        "mean_pos",
        "mean_neg",
        "mean_para",
        "mean_decoy",
        "mean_target_prompt_activation",
        "mean_control_prompt_activation",
        "target_minus_control_activation",
        "target_control_ratio",
        "family_consistency_count",
        "positive_family_count",
        "negative_family_count",
        "activation_nonzero_rate_target",
        "activation_nonzero_rate_control",
        "target_minus_control_activation_sentiment_negation",
        "target_minus_control_activation_property_negation",
        "target_minus_control_activation_state_negation",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_id, score, target, control, gap, ratio, consistency, nz_t, nz_c in rows:
            family_gaps = [gap if idx < consistency else -abs(gap) for idx in range(3)]
            writer.writerow(
                {
                    "feature_id": feature_id,
                    "score": score,
                    "abs_score": abs(score),
                    "mean_pos": target,
                    "mean_neg": control,
                    "mean_para": target,
                    "mean_decoy": control,
                    "mean_target_prompt_activation": target,
                    "mean_control_prompt_activation": control,
                    "target_minus_control_activation": gap,
                    "target_control_ratio": ratio,
                    "family_consistency_count": consistency,
                    "positive_family_count": consistency,
                    "negative_family_count": 3 - consistency,
                    "activation_nonzero_rate_target": nz_t,
                    "activation_nonzero_rate_control": nz_c,
                    "target_minus_control_activation_sentiment_negation": family_gaps[0],
                    "target_minus_control_activation_property_negation": family_gaps[1],
                    "target_minus_control_activation_state_negation": family_gaps[2],
                }
            )


def test_top_bottom_and_random_selection_are_deterministic(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    assert select_top_features(ranking, top_k=2) == ["sae_0", "sae_1"]
    assert select_bottom_active_features(ranking, top_k=2) == ["sae_7", "sae_6"]
    assert select_seeded_random_features(ranking, top_k=2, seed=7) == select_seeded_random_features(
        ranking,
        top_k=2,
        seed=7,
    )
    assert not set(select_seeded_random_features(ranking, top_k=2, seed=7)) & {
        "sae_0",
        "sae_1",
    }


def test_multiple_seeded_random_feature_sets_have_labels(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    sets = select_multiple_seeded_random_feature_sets(ranking, top_k=2, seeds=[7, 11])

    assert set(sets) == {"random_seed_7", "random_seed_11"}
    assert sets["random_seed_7"] != sets["random_seed_11"]


def test_residual_rankings_are_rejected_for_sae_baselines(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking, prefix="resid_")

    with pytest.raises(ValueError, match="SAE feature ids"):
        select_top_features(ranking, top_k=1)


def test_insufficient_feature_count_raises_clear_error(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    with pytest.raises(ValueError, match="insufficient"):
        select_seeded_random_features(ranking, top_k=10, seed=7)


def test_feature_set_artifact_shape(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    artifact = build_feature_sets(
        ranking,
        top_k=2,
        baseline_mode="top-vs-random-multiseed",
        random_seeds=[7, 11],
    )

    labels = [row["label"] for row in artifact["feature_sets"]]
    assert labels == ["top", "random_seed_7", "random_seed_11"]
    assert artifact["feature_sets"][0]["selection_method"] == "ranking_abs_score_top_k"
    assert artifact["feature_sets"][0]["feature_selection_mode"] == "top"


def test_density_matched_feature_set_artifact_shape(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    artifact = build_feature_sets(
        ranking,
        top_k=2,
        baseline_mode="top-vs-density-matched-multiseed",
        random_seeds=[7, 11],
    )

    labels = [row["label"] for row in artifact["feature_sets"]]
    assert labels == ["top", "density_matched_seed_7", "density_matched_seed_11"]
    density_row = artifact["feature_sets"][1]
    assert density_row["selection_method"] == "activation_density_matched"
    assert density_row["matched_control_metadata"]["stats_source"] == (
        "per_condition_mean_approximation"
    )
    assert not set(density_row["feature_ids"]) & {"sae_0", "sae_1"}


def test_top_positive_feature_selection_excludes_wrong_sign_features(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    assert select_features_by_mode(
        ranking,
        top_k=3,
        feature_selection_mode="top-positive",
    ) == ["sae_0", "sae_3", "sae_5"]


def test_family_consistent_mode_blocks_without_family_columns(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    with pytest.raises(ValueError, match="per-family ranking score columns"):
        select_features_by_mode(
            ranking,
            top_k=2,
            feature_selection_mode="top-family-consistent",
        )


def test_feature_selection_mode_is_recorded(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    artifact = build_feature_sets(
        ranking,
        top_k=2,
        baseline_mode="top",
        feature_selection_mode="top-positive",
    )

    top = artifact["feature_sets"][0]
    assert artifact["feature_selection_mode"] == "top-positive"
    assert top["selection_method"] == "ranking_positive_score_top_k"
    assert top["feature_ids"] == ["sae_0", "sae_3"]


def test_specificity_feature_selection_modes_are_deterministic(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_rich_ranking(ranking)

    assert select_features_by_mode(
        ranking,
        top_k=2,
        feature_selection_mode="top-target-control-gap",
    ) == ["sae_0", "sae_2"]
    assert select_features_by_mode(
        ranking,
        top_k=2,
        feature_selection_mode="top-target-control-ratio",
    ) == ["sae_4", "sae_0"]
    assert select_features_by_mode(
        ranking,
        top_k=2,
        feature_selection_mode="top-family-consistent-gap",
        min_family_consistency=2,
    ) == ["sae_0", "sae_2"]
    assert select_features_by_mode(
        ranking,
        top_k=2,
        feature_selection_mode="top-low-control-activation",
        max_control_activation_quantile=0.5,
    ) == ["sae_0", "sae_3"]
    assert select_features_by_mode(
        ranking,
        top_k=2,
        feature_selection_mode="ensemble-specificity",
        max_control_activation_quantile=0.5,
    ) == ["sae_0", "sae_4"]


def test_specificity_mode_blocks_without_required_columns(tmp_path) -> None:
    ranking = tmp_path / "feature_rankings.csv"
    _write_ranking(ranking)

    with pytest.raises(ValueError, match="required columns"):
        select_features_by_mode(
            ranking,
            top_k=2,
            feature_selection_mode="top-target-control-gap",
        )
