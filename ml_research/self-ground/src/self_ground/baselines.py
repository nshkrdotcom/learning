from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any


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
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    rows = _read_ranking(ranking_path)
    if len(rows) < top_k:
        raise ValueError("insufficient SAE features for requested top_k")
    return [str(row["feature_id"]) for row in rows[:top_k]]


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
) -> dict[str, Any]:
    if baseline_mode not in {
        "top",
        "top-vs-random",
        "top-vs-random-multiseed",
        "top-vs-bottom-active",
        "top-vs-random-and-bottom-active",
    }:
        raise ValueError("unknown baseline_mode")
    seeds = random_seeds or [7, 11, 13]
    rows: list[dict[str, Any]] = [
        {
            "label": "top",
            "selection_method": "ranking_abs_score_top_k",
            "feature_ids": select_top_features(ranking_path, top_k=top_k),
            "seed": None,
        }
    ]
    random_modes = {
        "top-vs-random",
        "top-vs-random-multiseed",
        "top-vs-random-and-bottom-active",
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
    if baseline_mode in {"top-vs-bottom-active", "top-vs-random-and-bottom-active"}:
        rows.append(
            {
                "label": "bottom_active",
                "selection_method": "bottom_active_abs_score",
                "feature_ids": select_bottom_active_features(ranking_path, top_k=top_k),
                "seed": None,
            }
        )
    return {"feature_sets": rows}
