from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from self_ground.io import write_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "count_nonzero": 0,
            "n": 0,
        }
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "count_nonzero": sum(1 for value in values if abs(value) > 1e-12),
        "n": len(values),
    }


def _ranking_by_id(ranking_dir: Path) -> dict[str, dict[str, str]]:
    path = ranking_dir / "feature_rankings.csv" if ranking_dir.is_dir() else ranking_dir
    rows = _read_csv(path)
    return {str(row.get("feature_id")): row for row in rows}


def _feature_activity_rows(
    *,
    feature_sets: list[dict[str, Any]],
    ranking: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        metadata = feature_set.get("matched_control_metadata", {})
        for feature_id in feature_set.get("feature_ids", []):
            ranking_row = ranking.get(str(feature_id), {})
            rows.append(
                {
                    "feature_set_label": feature_set.get("label"),
                    "selection_method": feature_set.get("selection_method"),
                    "feature_id": feature_id,
                    "ranking_abs_score": _float(ranking_row.get("abs_score")),
                    "mean_pos": _float(ranking_row.get("mean_pos")),
                    "mean_neg": _float(ranking_row.get("mean_neg")),
                    "mean_para": _float(ranking_row.get("mean_para")),
                    "mean_decoy": _float(ranking_row.get("mean_decoy")),
                    "density_stats_source": metadata.get("stats_source"),
                    "density_relaxed": metadata.get("relaxed"),
                }
            )
    return rows


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return grouped


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def diagnose_zero_effect_run(
    *,
    run_dir: str | Path,
    ranking_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    ranking_path = Path(ranking_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    feature_sets_path = run_path / "feature_sets.json"
    row_path = run_path / "behavioral_intervention_results.jsonl"
    baseline_path = run_path / "baseline_task_scores.jsonl"
    summary_path = run_path / "behavioral_summary.csv"
    missing = [
        str(path.name)
        for path in [feature_sets_path, row_path, baseline_path, summary_path]
        if not path.exists()
    ]

    feature_sets = _read_json(feature_sets_path).get("feature_sets", []) if not missing else []
    rows = _read_jsonl(row_path) if row_path.exists() else []
    baseline_rows = _read_jsonl(baseline_path) if baseline_path.exists() else []
    summary_rows = _read_csv(summary_path)
    ranking = _ranking_by_id(ranking_path) if Path(ranking_path).exists() else {}

    feature_activity = _feature_activity_rows(feature_sets=feature_sets, ranking=ranking)
    _write_csv(
        out_path / "feature_activity_summary.csv",
        feature_activity,
        [
            "feature_set_label",
            "selection_method",
            "feature_id",
            "ranking_abs_score",
            "mean_pos",
            "mean_neg",
            "mean_para",
            "mean_decoy",
            "density_stats_source",
            "density_relaxed",
        ],
    )

    row_delta_distribution: list[dict[str, Any]] = []
    telemetry_summary: list[dict[str, Any]] = []
    for label, grouped in sorted(_group_rows(rows, "feature_set_label").items()):
        target_values = [_float(row.get("target_absolute_delta")) for row in grouped]
        control_values = [_float(row.get("control_absolute_delta")) for row in grouped]
        decoded_norms = [_float(row.get("decoded_delta_norm_mean")) for row in grouped]
        norm_drifts = [_float(row.get("relative_norm_drift_mean")) for row in grouped]
        row_delta_distribution.append(
            {
                "feature_set_label": label,
                **{f"target_{k}": v for k, v in _stats(target_values).items()},
                **{f"control_{k}": v for k, v in _stats(control_values).items()},
            }
        )
        telemetry_summary.append(
            {
                "feature_set_label": label,
                "relative_norm_drift_mean": _stats(norm_drifts)["mean"],
                "relative_norm_drift_max": _stats(norm_drifts)["max"],
                "decoded_delta_norm_mean": _stats(decoded_norms)["mean"],
                "decoded_delta_norm_max": _stats(decoded_norms)["max"],
                "decoded_delta_norm_zero_count": sum(
                    1 for value in decoded_norms if abs(value) <= 1e-12
                ),
                "missing_telemetry_rows": sum(
                    1
                    for row in grouped
                    if "decoded_delta_norm_mean" not in row
                    or "relative_norm_drift_mean" not in row
                ),
            }
        )
    _write_csv(
        out_path / "row_delta_distribution.csv",
        row_delta_distribution,
        [
            "feature_set_label",
            "target_min",
            "target_max",
            "target_mean",
            "target_median",
            "target_count_nonzero",
            "target_n",
            "control_min",
            "control_max",
            "control_mean",
            "control_median",
            "control_count_nonzero",
            "control_n",
        ],
    )
    _write_csv(
        out_path / "patch_telemetry_summary.csv",
        telemetry_summary,
        [
            "feature_set_label",
            "relative_norm_drift_mean",
            "relative_norm_drift_max",
            "decoded_delta_norm_mean",
            "decoded_delta_norm_max",
            "decoded_delta_norm_zero_count",
            "missing_telemetry_rows",
        ],
    )

    nonzero_row_count = sum(
        1
        for row in rows
        if abs(_float(row.get("target_absolute_delta"))) > 1e-12
        or abs(_float(row.get("control_absolute_delta"))) > 1e-12
    )
    all_summary_zero = all(
        abs(_float(row.get("target_absolute_delta_mean"))) <= 1e-12
        and abs(_float(row.get("control_absolute_delta_mean"))) <= 1e-12
        for row in summary_rows
        if row.get("family") == "__all__"
    )
    ranking_scores = [_float(row.get("ranking_abs_score")) for row in feature_activity]
    decoded_norm_values = [
        _float(row.get("decoded_delta_norm_mean"))
        for row in rows
        if "decoded_delta_norm_mean" in row
    ]
    pass_rates = [
        bool(row.get("intended_direction_pass"))
        for row in baseline_rows
        if "intended_direction_pass" in row
    ]
    intended_pass_rate = (
        sum(1 for value in pass_rates if value) / len(pass_rates)
        if pass_rates
        else None
    )

    labels: list[str] = []
    if missing:
        labels.append("insufficient_artifacts_for_diagnosis")
    if rows and nonzero_row_count == 0:
        labels.append("all_row_deltas_zero")
    if all_summary_zero and nonzero_row_count > 0:
        labels.append("aggregation_hides_nonzero_rows")
    if ranking_scores and max(ranking_scores) <= 1e-12:
        labels.append("selected_features_have_zero_or_nearzero_ranking_activity")
    if not decoded_norm_values or all(abs(value) <= 1e-12 for value in decoded_norm_values):
        labels.append("decoded_delta_norm_zero_or_missing")
    if intended_pass_rate is not None and intended_pass_rate < 0.5:
        labels.append("task_baseline_not_calibrated")

    if "insufficient_artifacts_for_diagnosis" in labels:
        next_action = "blocked_missing_artifacts"
    elif "decoded_delta_norm_zero_or_missing" in labels:
        next_action = "inspect_sae_decode_patch"
    elif "task_baseline_not_calibrated" in labels:
        next_action = "fix_task_calibration"
    elif "all_row_deltas_zero" in labels:
        next_action = "include_amplify_operation"
    elif "selected_features_have_zero_or_nearzero_ranking_activity" in labels:
        next_action = "increase_ranking_scale"
    else:
        next_action = "run_serious_gpu_e002"

    diagnosis = {
        "run_dir": str(run_path),
        "ranking_dir": str(ranking_path),
        "missing_artifacts": missing,
        "diagnosis_labels": labels,
        "recommended_next_action": next_action,
        "n_behavioral_rows": len(rows),
        "n_nonzero_row_deltas": nonzero_row_count,
        "all_summary_rows_zero": all_summary_zero,
        "baseline_intended_direction_pass_rate": intended_pass_rate,
        "selected_feature_ids": {
            str(row.get("label")): list(row.get("feature_ids", []))
            for row in feature_sets
        },
    }
    write_config(diagnosis, out_path / "zero_effect_diagnosis.json")
    readme = f"""# Zero-Effect Diagnosis

- source run: `{run_path}`
- ranking: `{ranking_path}`
- labels: `{labels}`
- recommended next action: `{next_action}`

This diagnostic reads artifacts only. It does not rerun the model.
"""
    (out_path / "README.md").write_text(readme, encoding="utf-8")
    return diagnosis


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose zero or hidden Phase 3 effects.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ranking-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    diagnosis = diagnose_zero_effect_run(
        run_dir=args.run_dir,
        ranking_dir=args.ranking_dir,
        out_dir=args.out,
    )
    print(json.dumps(diagnosis, indent=2, sort_keys=True))
    return 0 if "insufficient_artifacts_for_diagnosis" not in diagnosis["diagnosis_labels"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
