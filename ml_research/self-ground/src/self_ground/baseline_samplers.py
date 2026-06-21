from __future__ import annotations

import csv
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureActivationStats:
    feature_id: str
    abs_score: float
    activation_mean: float
    activation_abs_mean: float
    activation_nonzero_fraction: float
    activation_std: float
    stats_source: str = "per_condition_mean_approximation"


@dataclass(frozen=True)
class MatchedControlConfig:
    top_k: int
    seed: int
    candidate_pool_exclude_top_fraction: float = 0.1
    density_tolerance: float = 0.10
    abs_mean_tolerance: float = 0.10
    allow_relaxed_tolerance: bool = True
    max_relaxation_steps: int = 5


@dataclass(frozen=True)
class MatchedControlResult:
    label: str
    feature_ids: list[str]
    seed: int
    selection_method: str
    matched_on: list[str]
    tolerance_used: dict[str, float]
    top_stats_summary: dict[str, float]
    control_stats_summary: dict[str, float]
    candidate_pool_size: int
    relaxed: bool
    stats_source: str


CONDITION_MEAN_COLUMNS = ["mean_pos", "mean_neg", "mean_para", "mean_decoy"]
TRUE_DENSITY_COLUMNS = [
    "activation_mean",
    "activation_abs_mean",
    "activation_nonzero_fraction",
    "activation_std",
]


def _ranking_file(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "feature_rankings.csv"
    if not path.exists():
        raise ValueError(f"ranking file does not exist: {path}")
    return path


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def _stats_from_row(row: dict[str, Any]) -> FeatureActivationStats:
    feature_id = str(row.get("feature_id") or "")
    if not feature_id.startswith("sae_"):
        raise ValueError(f"SAE feature id required for matched controls: {feature_id!r}")
    abs_score = _float(row, "abs_score", abs(_float(row, "score")))

    if all(row.get(column) not in {None, ""} for column in TRUE_DENSITY_COLUMNS):
        return FeatureActivationStats(
            feature_id=feature_id,
            abs_score=abs_score,
            activation_mean=_float(row, "activation_mean"),
            activation_abs_mean=_float(row, "activation_abs_mean"),
            activation_nonzero_fraction=_float(row, "activation_nonzero_fraction"),
            activation_std=_float(row, "activation_std"),
            stats_source="per_example_activation_density",
        )

    if all(row.get(column) not in {None, ""} for column in CONDITION_MEAN_COLUMNS):
        means = [_float(row, column) for column in CONDITION_MEAN_COLUMNS]
        return FeatureActivationStats(
            feature_id=feature_id,
            abs_score=abs_score,
            activation_mean=sum(means) / len(means),
            activation_abs_mean=sum(abs(value) for value in means) / len(means),
            activation_nonzero_fraction=sum(1 for value in means if value != 0.0)
            / len(means),
            activation_std=statistics.pstdev(means),
            stats_source="per_condition_mean_approximation",
        )

    score = _float(row, "score")
    return FeatureActivationStats(
        feature_id=feature_id,
        abs_score=abs_score,
        activation_mean=score,
        activation_abs_mean=abs_score,
        activation_nonzero_fraction=1.0 if abs_score > 0.0 else 0.0,
        activation_std=0.0,
        stats_source="ranking_score_fallback",
    )


def load_feature_activation_stats_from_ranking(ranking_path: Path) -> list[FeatureActivationStats]:
    path = _ranking_file(ranking_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"ranking file is empty: {path}")
    return [_stats_from_row(row) for row in rows]


def _summary(stats: list[FeatureActivationStats]) -> dict[str, float]:
    if not stats:
        return {
            "activation_mean": 0.0,
            "activation_abs_mean": 0.0,
            "activation_nonzero_fraction": 0.0,
            "activation_std": 0.0,
            "abs_score": 0.0,
        }
    return {
        "activation_mean": sum(item.activation_mean for item in stats) / len(stats),
        "activation_abs_mean": sum(item.activation_abs_mean for item in stats) / len(stats),
        "activation_nonzero_fraction": sum(
            item.activation_nonzero_fraction for item in stats
        )
        / len(stats),
        "activation_std": sum(item.activation_std for item in stats) / len(stats),
        "abs_score": sum(item.abs_score for item in stats) / len(stats),
    }


def _within_tolerance(value: float, target: float, tolerance: float) -> bool:
    scale = max(abs(target), 1e-12)
    return abs(value - target) <= tolerance * scale


def _candidate_pool(
    *,
    stats: list[FeatureActivationStats],
    top_feature_ids: list[str],
    exclude_top_fraction: float,
    top_k: int,
) -> list[FeatureActivationStats]:
    excluded = set(top_feature_ids)
    rank_exclusion_count = max(top_k, int(len(stats) * exclude_top_fraction))
    rank_excluded = {item.feature_id for item in stats[:rank_exclusion_count]}
    return [item for item in stats if item.feature_id not in excluded | rank_excluded]


def _match_distance(
    item: FeatureActivationStats,
    top_summary: dict[str, float],
) -> float:
    density_scale = max(abs(top_summary["activation_nonzero_fraction"]), 1e-12)
    abs_mean_scale = max(abs(top_summary["activation_abs_mean"]), 1e-12)
    density_distance = abs(
        item.activation_nonzero_fraction - top_summary["activation_nonzero_fraction"]
    ) / density_scale
    abs_mean_distance = abs(item.activation_abs_mean - top_summary["activation_abs_mean"]) / (
        abs_mean_scale
    )
    return density_distance + abs_mean_distance


def _serialize_result(result: MatchedControlResult) -> dict[str, Any]:
    return asdict(result)


def select_activation_density_matched_features(
    ranking_path: Path,
    *,
    top_feature_ids: list[str],
    config: MatchedControlConfig,
) -> MatchedControlResult:
    if config.top_k < 1:
        raise ValueError("top_k must be >= 1")
    if len(top_feature_ids) < config.top_k:
        raise ValueError("top_feature_ids must contain at least top_k ids")

    stats = load_feature_activation_stats_from_ranking(ranking_path)
    by_id = {item.feature_id: item for item in stats}
    missing = [
        feature_id
        for feature_id in top_feature_ids[: config.top_k]
        if feature_id not in by_id
    ]
    if missing:
        raise ValueError(f"top feature ids are missing from ranking stats: {missing}")

    top_stats = [by_id[feature_id] for feature_id in top_feature_ids[: config.top_k]]
    top_summary = _summary(top_stats)
    pool = _candidate_pool(
        stats=stats,
        top_feature_ids=top_feature_ids,
        exclude_top_fraction=config.candidate_pool_exclude_top_fraction,
        top_k=config.top_k,
    )
    if len(pool) < config.top_k:
        raise ValueError("insufficient candidate pool for activation-density matched controls")

    rng = random.Random(config.seed)

    density_tolerance = config.density_tolerance
    abs_mean_tolerance = config.abs_mean_tolerance
    relaxed = False
    selected: list[FeatureActivationStats] = []
    for step in range(config.max_relaxation_steps + 1):
        matching = [
            item
            for item in pool
            if _within_tolerance(
                item.activation_nonzero_fraction,
                top_summary["activation_nonzero_fraction"],
                density_tolerance,
            )
            and _within_tolerance(
                item.activation_abs_mean,
                top_summary["activation_abs_mean"],
                abs_mean_tolerance,
            )
        ]
        selected = [
            item
            for _, _, item in sorted(
                (
                    (_match_distance(item, top_summary), rng.random(), item)
                    for item in matching
                ),
                key=lambda value: (value[0], value[1], value[2].feature_id),
            )
        ]
        if len(selected) >= config.top_k:
            break
        if not config.allow_relaxed_tolerance or step >= config.max_relaxation_steps:
            raise ValueError(
                "insufficient activation-density matched controls; "
                f"matched {len(selected)} of {config.top_k} with "
                f"density_tolerance={density_tolerance} and "
                f"abs_mean_tolerance={abs_mean_tolerance}"
            )
        density_tolerance *= 2.0
        abs_mean_tolerance *= 2.0
        relaxed = True

    chosen = selected[: config.top_k]
    stats_sources = sorted({item.stats_source for item in [*top_stats, *chosen]})
    stats_source = stats_sources[0] if len(stats_sources) == 1 else "mixed"
    return MatchedControlResult(
        label=f"density_matched_seed_{config.seed}",
        feature_ids=[item.feature_id for item in chosen],
        seed=config.seed,
        selection_method="activation_density_matched",
        matched_on=["activation_nonzero_fraction", "activation_abs_mean"],
        tolerance_used={
            "density_tolerance": density_tolerance,
            "abs_mean_tolerance": abs_mean_tolerance,
        },
        top_stats_summary=top_summary,
        control_stats_summary=_summary(chosen),
        candidate_pool_size=len(pool),
        relaxed=relaxed,
        stats_source=stats_source,
    )


def select_multiple_activation_density_matched_feature_sets(
    ranking_path: Path,
    *,
    top_feature_ids: list[str],
    top_k: int,
    seeds: list[int],
    density_tolerance: float = 0.10,
    abs_mean_tolerance: float = 0.10,
) -> dict[str, MatchedControlResult]:
    return {
        f"density_matched_seed_{seed}": select_activation_density_matched_features(
            ranking_path,
            top_feature_ids=top_feature_ids,
            config=MatchedControlConfig(
                top_k=top_k,
                seed=seed,
                density_tolerance=density_tolerance,
                abs_mean_tolerance=abs_mean_tolerance,
            ),
        )
        for seed in seeds
    }


def matched_control_result_to_dict(result: MatchedControlResult) -> dict[str, Any]:
    return _serialize_result(result)
