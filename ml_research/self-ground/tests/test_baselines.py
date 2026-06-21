from __future__ import annotations

import csv

import pytest

from self_ground.baselines import (
    build_feature_sets,
    select_bottom_active_features,
    select_multiple_seeded_random_feature_sets,
    select_seeded_random_features,
    select_top_features,
)


def _write_ranking(path, prefix: str = "sae_") -> None:
    rows = [
        ("0", 10.0),
        ("1", -8.0),
        ("2", 0.0),
        ("3", 0.3),
        ("4", -0.2),
        ("5", 0.1),
        ("6", 0.05),
        ("7", -0.04),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_id", "score", "abs_score"])
        writer.writeheader()
        for suffix, score in rows:
            writer.writerow(
                {
                    "feature_id": f"{prefix}{suffix}",
                    "score": score,
                    "abs_score": abs(score),
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
