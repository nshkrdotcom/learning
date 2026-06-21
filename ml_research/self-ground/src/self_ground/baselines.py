from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from self_ground.baseline_samplers import (
    MatchedControlConfig,
    matched_control_result_to_dict,
    select_activation_density_matched_features,
)
from self_ground.behavioral_tasks import TASK_FAMILY_ORDER


def _read_ranking(ranking_path: Path) -> list[dict[str, Any]]:
    path = Path(ranking_path)
    if path.is_dir():
        path = path / "feature_rankings.csv"
    if not path.exists():
        raise ValueError(f"ranking file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"ranking file is empty: {path}")
    invalid = [
        row.get("feature_id", "")
        for row in rows
        if not row.get("feature_id", "").startswith("sae_")
    ]
    if invalid:
        raise ValueError(f"SAE decoded intervention requires SAE feature ids; got {invalid[:3]}")
    for row in rows:
        row["abs_score_float"] = float(row.get("abs_score") or abs(float(row.get("score", 0.0))))
    return rows


def select_top_features(ranking_path: Path, *, top_k: int) -> list[str]:
    return select_features_by_mode(ranking_path, top_k=top_k, feature_selection_mode="top")


def _score(row: dict[str, Any]) -> float:
    return float(row.get("score") or 0.0)


def _family_score_columns(rows: list[dict[str, Any]]) -> dict[str, str]:
    if not rows:
        return {}
    columns = set(rows[0])
    mapping: dict[str, str] = {}
    for family in TASK_FAMILY_ORDER:
        candidates = [
            f"score_{family}",
            f"{family}_score",
            f"ranking_score_{family}",
            f"{family}_ranking_score",
        ]
        for candidate in candidates:
            if candidate in columns:
                mapping[family] = candidate
                break
    return mapping


def select_features_by_mode(
    ranking_path: Path,
    *,
    top_k: int,
    feature_selection_mode: str = "top",
    min_family_consistency: int = 3,
) -> list[str]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    rows = _read_ranking(ranking_path)
    if feature_selection_mode in {"top", "top-absolute"}:
        selected = rows[:top_k]
    elif feature_selection_mode == "top-positive":
        selected = [row for row in rows if _score(row) > 0][:top_k]
    elif feature_selection_mode == "top-family-consistent":
        if min_family_consistency < 1:
            raise ValueError("min_family_consistency must be >= 1")
        family_columns = _family_score_columns(rows)
        missing = [family for family in TASK_FAMILY_ORDER if family not in family_columns]
        if missing:
            raise ValueError(
                "top-family-consistent requires per-family ranking score columns; "
                f"missing {missing}"
            )
        selected = [
            row
            for row in rows
            if sum(float(row[family_columns[family]]) > 0 for family in TASK_FAMILY_ORDER)
            >= min_family_consistency
        ][:top_k]
    else:
        raise ValueError(f"unknown feature_selection_mode: {feature_selection_mode}")
    if len(selected) < top_k:
        raise ValueError(
            f"insufficient SAE features for requested top_k under {feature_selection_mode}"
        )
    return [str(row["feature_id"]) for row in selected]


def select_bottom_active_features(
    ranking_path: Path,
    *,
    top_k: int,
    min_abs_activation_score: float | None = None,
) -> list[str]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    rows = _read_ranking(ranking_path)
    threshold = 0.0 if min_abs_activation_score is None else min_abs_activation_score
    active = [row for row in rows if float(row["abs_score_float"]) > threshold]
    pool = active if len(active) >= top_k else rows
    if len(pool) < top_k:
        raise ValueError("insufficient active SAE features for requested top_k")
    ordered = sorted(pool, key=lambda row: (float(row["abs_score_float"]), str(row["feature_id"])))
    return [str(row["feature_id"]) for row in ordered[:top_k]]


def select_seeded_random_features(
    ranking_path: Path,
    *,
    top_k: int,
    seed: int = 7,
    exclude_top_k: int = 0,
    exclude_top_fraction: float = 0.1,
) -> list[str]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    rows = _read_ranking(ranking_path)
    top_exclusion = max(exclude_top_k, int(len(rows) * exclude_top_fraction))
    top_exclusion = max(top_exclusion, top_k)
    pool = rows[top_exclusion:]
    if len(pool) < top_k:
        raise ValueError("insufficient SAE features after excluding top-ranked features")
    rng = random.Random(seed)
    selected = rng.sample(pool, top_k)
    return [str(row["feature_id"]) for row in selected]


def select_multiple_seeded_random_feature_sets(
    ranking_path: Path,
    *,
    top_k: int,
    seeds: list[int],
    exclude_top_fraction: float = 0.1,
) -> dict[str, list[str]]:
    return {
        f"random_seed_{seed}": select_seeded_random_features(
            ranking_path,
            top_k=top_k,
            seed=seed,
            exclude_top_fraction=exclude_top_fraction,
        )
        for seed in seeds
    }


def build_feature_sets(
    ranking_path: Path,
    *,
    top_k: int,
    baseline_mode: str = "top-vs-random-multiseed",
    random_seeds: list[int] | None = None,
    density_tolerance: float = 0.10,
    abs_mean_tolerance: float = 0.10,
    allow_relaxed_density_matching: bool = True,
    feature_selection_mode: str = "top",
    min_family_consistency: int = 3,
) -> dict[str, Any]:
    if baseline_mode not in {
        "top",
        "top-vs-random",
        "top-vs-random-multiseed",
        "top-vs-bottom-active",
        "top-vs-random-and-bottom-active",
        "top-vs-density-matched",
        "top-vs-density-matched-multiseed",
        "top-vs-random-and-density-matched",
        "top-vs-random-density-and-bottom-active",
    }:
        raise ValueError("unknown baseline_mode")
    seeds = random_seeds or [7, 11, 13]
    top_feature_ids = select_features_by_mode(
        ranking_path,
        top_k=top_k,
        feature_selection_mode=feature_selection_mode,
        min_family_consistency=min_family_consistency,
    )
    selection_method = {
        "top": "ranking_abs_score_top_k",
        "top-absolute": "ranking_abs_score_top_k",
        "top-positive": "ranking_positive_score_top_k",
        "top-family-consistent": "ranking_positive_family_consistent_top_k",
    }[feature_selection_mode]
    rows: list[dict[str, Any]] = [
        {
            "label": "top",
            "selection_method": selection_method,
            "feature_selection_mode": feature_selection_mode,
            "min_family_consistency": min_family_consistency
            if feature_selection_mode == "top-family-consistent"
            else None,
            "feature_ids": top_feature_ids,
            "seed": None,
        }
    ]
    random_modes = {
        "top-vs-random",
        "top-vs-random-multiseed",
        "top-vs-random-and-bottom-active",
        "top-vs-random-and-density-matched",
        "top-vs-random-density-and-bottom-active",
    }
    if baseline_mode in random_modes:
        used_seeds = seeds[:1] if baseline_mode == "top-vs-random" else seeds
        for seed in used_seeds:
            rows.append(
                {
                    "label": f"random_seed_{seed}",
                    "selection_method": "seeded_random_excluding_top_fraction",
                    "feature_ids": select_seeded_random_features(
                        ranking_path,
                        top_k=top_k,
                        seed=seed,
                    ),
                    "seed": seed,
                }
            )
    density_modes = {
        "top-vs-density-matched",
        "top-vs-density-matched-multiseed",
        "top-vs-random-and-density-matched",
        "top-vs-random-density-and-bottom-active",
    }
    if baseline_mode in density_modes:
        used_seeds = seeds[:1] if baseline_mode == "top-vs-density-matched" else seeds
        for seed in used_seeds:
            matched = select_activation_density_matched_features(
                ranking_path,
                top_feature_ids=top_feature_ids,
                config=MatchedControlConfig(
                    top_k=top_k,
                    seed=seed,
                    density_tolerance=density_tolerance,
                    abs_mean_tolerance=abs_mean_tolerance,
                    allow_relaxed_tolerance=allow_relaxed_density_matching,
                ),
            )
            rows.append(
                {
                    "label": matched.label,
                    "selection_method": matched.selection_method,
                    "feature_ids": matched.feature_ids,
                    "seed": seed,
                    "matched_control_metadata": matched_control_result_to_dict(matched),
                }
            )
    if baseline_mode in {"top-vs-bottom-active", "top-vs-random-and-bottom-active"}:
        rows.append(
            {
                "label": "bottom_active",
                "selection_method": "bottom_active_abs_score",
                "feature_ids": select_bottom_active_features(ranking_path, top_k=top_k),
                "seed": None,
            }
        )
    if baseline_mode == "top-vs-random-density-and-bottom-active":
        rows.append(
            {
                "label": "bottom_active",
                "selection_method": "bottom_active_abs_score",
                "feature_ids": select_bottom_active_features(ranking_path, top_k=top_k),
                "seed": None,
            }
        )
    return {
        "feature_selection_mode": feature_selection_mode,
        "min_family_consistency": min_family_consistency,
        "feature_sets": rows,
    }
