from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": _mean(ordered),
        "max": ordered[-1],
    }


def _ranking_by_id(ranking_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    enriched: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(ranking_rows, start=1):
        feature_id = str(row.get("feature_id"))
        means = [
            value
            for key in ["mean_pos", "mean_neg", "mean_para", "mean_decoy"]
            if (value := _float(row.get(key))) is not None
        ]
        enriched[feature_id] = {
            **row,
            "ranking_position": index,
            "score_float": _float(row.get("score")) or 0.0,
            "abs_score_float": _float(row.get("abs_score")) or abs(_float(row.get("score")) or 0.0),
            "activation_abs_mean_estimate": _mean([abs(value) for value in means]),
            "activation_mean_estimate": _mean(means),
            "density_stats_source": (
                "per_condition_mean_approximation"
                if means
                else "ranking_artifact_missing_condition_means"
            ),
        }
    return enriched


def _feature_rows(
    feature_sets: list[dict[str, Any]],
    ranking_lookup: dict[str, dict[str, Any]],
    labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        label = str(feature_set.get("label"))
        if labels is not None and label not in labels:
            continue
        for feature_id in feature_set.get("feature_ids", []):
            ranking = ranking_lookup.get(str(feature_id), {})
            rows.append(
                {
                    "feature_set_label": label,
                    "selection_method": feature_set.get("selection_method"),
                    "feature_id": feature_id,
                    "ranking_position": ranking.get("ranking_position"),
                    "score": ranking.get("score_float"),
                    "abs_score": ranking.get("abs_score_float"),
                    "activation_abs_mean_estimate": ranking.get("activation_abs_mean_estimate"),
                    "density_stats_source": ranking.get("density_stats_source"),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _all_summary_rows(eval_dir: Path) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(eval_dir / "behavioral_summary.csv")
        if row.get("family") == "__all__"
    ]


def analyze_feature_selection(ranking_dir: Path, eval_dir: Path, out_dir: Path) -> dict[str, Any]:
    ranking_rows = _read_csv(Path(ranking_dir) / "feature_rankings.csv")
    feature_sets = _read_json(Path(eval_dir) / "feature_sets.json").get("feature_sets", [])
    summary_rows = _all_summary_rows(Path(eval_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_lookup = _ranking_by_id(ranking_rows)

    scores = [_float(row.get("score")) or 0.0 for row in ranking_rows]
    abs_scores = [
        _float(row.get("abs_score")) or abs(_float(row.get("score")) or 0.0)
        for row in ranking_rows
    ]
    ranking_distribution_rows = [
        {"metric": "score", **_summary(scores)},
        {"metric": "abs_score", **_summary(abs_scores)},
    ]
    _write_csv(
        out_dir / "ranking_distribution.csv",
        ranking_distribution_rows,
        ["metric", "min", "median", "mean", "max"],
    )

    top_labels = {"top"}
    control_labels = {str(row.get("label")) for row in feature_sets if row.get("label") != "top"}
    selected_rows = _feature_rows(feature_sets, ranking_lookup, top_labels)
    control_rows = _feature_rows(feature_sets, ranking_lookup, control_labels)
    feature_fieldnames = [
        "feature_set_label",
        "selection_method",
        "feature_id",
        "ranking_position",
        "score",
        "abs_score",
        "activation_abs_mean_estimate",
        "density_stats_source",
    ]
    _write_csv(out_dir / "selected_feature_table.csv", selected_rows, feature_fieldnames)
    _write_csv(out_dir / "control_feature_table.csv", control_rows, feature_fieldnames)

    effect_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        label = str(row.get("feature_set_label"))
        features = [
            value
            for item in (selected_rows + control_rows)
            if item["feature_set_label"] == label
            if (value := _float(item.get("abs_score"))) is not None
        ]
        effect_rows.append(
            {
                "feature_set_label": label,
                "operation": row.get("operation"),
                "factor": row.get("factor"),
                "target_absolute_delta_mean": row.get("target_absolute_delta_mean"),
                "control_absolute_delta_mean": row.get("control_absolute_delta_mean"),
                "specificity_gap_mean": row.get("specificity_gap_mean"),
                "mean_selected_abs_score": _mean(features),
            }
        )
    _write_csv(
        out_dir / "feature_set_effects.csv",
        effect_rows,
        [
            "feature_set_label",
            "operation",
            "factor",
            "target_absolute_delta_mean",
            "control_absolute_delta_mean",
            "specificity_gap_mean",
            "mean_selected_abs_score",
        ],
    )

    top_abs_mean = _mean(
        [value for row in selected_rows if (value := _float(row.get("abs_score"))) is not None]
    )
    density_abs_values = [
        value
        for row in control_rows
        if str(row["feature_set_label"]).startswith("density_matched_seed_")
        if (value := _float(row.get("abs_score"))) is not None
    ]
    density_abs_mean = _mean(density_abs_values)
    top_effect = next((row for row in effect_rows if row["feature_set_label"] == "top"), {})
    labels: list[str] = []
    if (
        top_abs_mean is not None
        and density_abs_mean is not None
        and top_abs_mean <= density_abs_mean * 1.05
    ):
        labels.append("top_features_not_separable_from_density_controls")
    if density_abs_values and max(density_abs_values) >= (top_abs_mean or 0.0) * 0.8:
        labels.append("density_controls_high_ranking")
    target = _float(top_effect.get("target_absolute_delta_mean")) or 0.0
    control = _float(top_effect.get("control_absolute_delta_mean")) or 0.0
    specificity = _float(top_effect.get("specificity_gap_mean")) or 0.0
    if target > 0 and specificity <= 0:
        labels.append("target_effect_present_but_not_specific")
    if control > target:
        labels.append("control_effect_dominates")
    if len(selected_rows) < 5:
        labels.append("insufficient_feature_count")
    if ranking_rows and "mean_pos" not in ranking_rows[0]:
        labels.append("ranking_artifact_missing_density_columns")
    if len(effect_rows) >= 2:
        pairs = [
            (
                _float(row.get("mean_selected_abs_score")),
                _float(row.get("target_absolute_delta_mean")),
            )
            for row in effect_rows
        ]
        valid_pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(valid_pairs) >= 2:
            high_score = max(valid_pairs, key=lambda item: item[0])
            high_effect = max(valid_pairs, key=lambda item: item[1])
            if high_score != high_effect:
                labels.append("ranking_metric_not_predictive_of_intervention_effect")

    diagnosis = {
        "diagnosis_labels": labels,
        "top_abs_score_mean": top_abs_mean,
        "density_control_abs_score_mean": density_abs_mean,
        "top_target_absolute_delta_mean": target,
        "top_control_absolute_delta_mean": control,
        "top_specificity_gap_mean": specificity,
        "top_features": [row["feature_id"] for row in selected_rows],
        "density_control_labels": sorted(
            {
                str(row["feature_set_label"])
                for row in control_rows
                if str(row["feature_set_label"]).startswith("density_matched_seed_")
            }
        ),
    }
    summary = {
        "ranking_dir": str(ranking_dir),
        "eval_dir": str(eval_dir),
        "n_ranked_features": len(ranking_rows),
        "n_feature_sets": len(feature_sets),
        "diagnosis_labels": labels,
    }
    (out_dir / "feature_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "feature_specificity_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme = f"""# E002 Feature Selection Analysis

This artifact-only analysis compares selected top SAE features to random,
density-matched, and bottom-active controls.

- top mean abs ranking score: `{top_abs_mean}`
- density-control mean abs ranking score: `{density_abs_mean}`
- diagnosis labels: `{labels}`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return diagnosis


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase 3 feature selection artifacts.")
    parser.add_argument("--ranking-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze_feature_selection(Path(args.ranking_dir), Path(args.eval_dir), Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
