from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _group_metrics(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    output = []
    for label, group in sorted(grouped.items()):
        target_values = [_float(row.get("target_absolute_delta")) for row in group]
        control_values = [_float(row.get("control_absolute_delta")) for row in group]
        gaps = [
            target - control
            for target, control in zip(target_values, control_values, strict=True)
        ]
        output.append(
            {
                key: label,
                "n_rows": len(group),
                "n_tasks": len({str(row.get("task_id")) for row in group}),
                "target_delta_mean": _mean(target_values),
                "control_delta_mean": _mean(control_values),
                "specificity_gap_mean": _mean(gaps),
                "target_control_ratio": (
                    _mean(target_values) / _mean(control_values)
                    if _mean(control_values) > 0
                    else None
                ),
                "control_dominant_rows": sum(
                    control > target
                    for target, control in zip(target_values, control_values, strict=True)
                ),
            }
        )
    return output


def _task_rows(
    *,
    behavioral_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = {str(row.get("task_id")): row for row in baseline_rows}
    task_by_id = {str(row.get("id")): row for row in tasks}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavioral_rows:
        grouped[str(row.get("task_id"))].append(row)
    output = []
    for task_id, group in sorted(grouped.items()):
        target_values = [_float(row.get("target_absolute_delta")) for row in group]
        control_values = [_float(row.get("control_absolute_delta")) for row in group]
        task = task_by_id.get(task_id, {})
        baseline = baseline_by_id.get(task_id, {})
        output.append(
            {
                "task_id": task_id,
                "family": group[0].get("family"),
                "template_id": (task.get("metadata") or {}).get("template_id")
                or (task.get("metadata") or {}).get("template_family")
                or "unknown",
                "target_tokens": ",".join(
                    task.get("target_tokens") or group[0].get("target_tokens") or []
                ),
                "foil_tokens": ",".join(
                    task.get("foil_tokens") or group[0].get("foil_tokens") or []
                ),
                "baseline_margin": _float(baseline.get("baseline_prompt_contrast")),
                "target_delta_mean": _mean(target_values),
                "control_delta_mean": _mean(control_values),
                "specificity_gap_mean": _mean(target_values) - _mean(control_values),
                "control_dominates": _mean(control_values) > _mean(target_values),
            }
        )
    return output


def _feature_set_rows(behavioral_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in behavioral_rows:
        key = (
            str(row.get("feature_set_label") or "unknown"),
            str(row.get("control_suite") or "matched_non_negation_current"),
        )
        grouped[key].append(row)
    output = []
    for (label, suite), group in sorted(grouped.items()):
        target_values = [_float(row.get("target_absolute_delta")) for row in group]
        control_values = [_float(row.get("control_absolute_delta")) for row in group]
        output.append(
            {
                "feature_set_label": label,
                "control_suite": suite,
                "n_rows": len(group),
                "target_delta_mean": _mean(target_values),
                "control_delta_mean": _mean(control_values),
                "specificity_gap_mean": _mean(target_values) - _mean(control_values),
                "target_control_ratio": (
                    _mean(target_values) / _mean(control_values)
                    if _mean(control_values) > 0
                    else None
                ),
            }
        )
    return output


def diagnose(
    run_dir: Path,
    ranking_dir: Path,
    calibration_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    behavioral_rows = _read_jsonl(run_dir / "behavioral_intervention_results.jsonl")
    baseline_rows = _read_jsonl(run_dir / "baseline_task_scores.jsonl")
    tasks = _read_jsonl(run_dir / "behavioral_tasks.jsonl")
    summary_rows = _read_csv(run_dir / "behavioral_summary.csv")
    ranking_rows = _read_csv(ranking_dir / "feature_rankings.csv")
    calibration = _read_json(calibration_dir / "calibration_summary.json")
    report = _read_json(run_dir / "mechanism_report.json")
    config = _read_json(run_dir / "config.json")

    all_summary = [
        row
        for row in summary_rows
        if row.get("family") == "__all__" and row.get("feature_set_label") == "top"
    ]
    target_mean = _mean([_float(row.get("target_absolute_delta_mean")) for row in all_summary])
    control_mean = _mean([_float(row.get("control_absolute_delta_mean")) for row in all_summary])
    specificity_gap = target_mean - control_mean
    ratio = target_mean / control_mean if control_mean > 0 else None

    by_family = _group_metrics(behavioral_rows, "family")
    task_effects = _task_rows(
        behavioral_rows=behavioral_rows,
        baseline_rows=baseline_rows,
        tasks=tasks,
    )
    by_feature_set = _feature_set_rows(behavioral_rows)
    by_template = _group_metrics(
        [
            {
                **row,
                "template_id": next(
                    (
                        (task.get("metadata") or {}).get("template_id")
                        or (task.get("metadata") or {}).get("template_family")
                        for task in tasks
                        if task.get("id") == row.get("task_id")
                    ),
                    "unknown",
                ),
            }
            for row in behavioral_rows
        ],
        "template_id",
    )

    top_feature_ids = set()
    feature_sets = _read_json(run_dir / "feature_sets.json").get("feature_sets", [])
    for feature_set in feature_sets:
        if feature_set.get("label") == "top":
            top_feature_ids.update(feature_set.get("feature_ids", []))
    feature_by_id = {row.get("feature_id"): row for row in ranking_rows}
    by_feature = []
    for feature_id in sorted(top_feature_ids):
        ranking = feature_by_id.get(feature_id, {})
        by_feature.append(
            {
                "feature_id": feature_id,
                "score": ranking.get("score"),
                "abs_score": ranking.get("abs_score"),
                "target_minus_control_activation": ranking.get("target_minus_control_activation"),
                "target_control_ratio": ranking.get("target_control_ratio"),
                "family_consistency_count": ranking.get("family_consistency_count"),
                "activation_nonzero_rate_target": ranking.get("activation_nonzero_rate_target"),
                "activation_nonzero_rate_control": ranking.get("activation_nonzero_rate_control"),
            }
        )

    feature_score_vs_effect = []
    set_effect_by_label = {row["feature_set_label"]: row for row in by_feature_set}
    for feature_set in feature_sets:
        effect = set_effect_by_label.get(feature_set.get("label"), {})
        for feature_id in feature_set.get("feature_ids", []):
            ranking = feature_by_id.get(feature_id, {})
            feature_score_vs_effect.append(
                {
                    "feature_set_label": feature_set.get("label"),
                    "feature_id": feature_id,
                    "ranking_score": ranking.get("score"),
                    "abs_score": ranking.get("abs_score"),
                    "target_minus_control_activation": ranking.get(
                        "target_minus_control_activation"
                    ),
                    "feature_set_specificity_gap": effect.get("specificity_gap_mean"),
                    "feature_set_target_delta": effect.get("target_delta_mean"),
                    "feature_set_control_delta": effect.get("control_delta_mean"),
                }
            )

    control_dominant = sorted(
        [row for row in task_effects if row["control_dominates"]],
        key=lambda row: row["specificity_gap_mean"],
    )
    target_dominant = sorted(
        [row for row in task_effects if not row["control_dominates"]],
        key=lambda row: -row["specificity_gap_mean"],
    )
    outliers = [
        {**row, "outlier_type": "control_dominates"} for row in control_dominant[:20]
    ] + [
        {**row, "outlier_type": "target_dominates"} for row in target_dominant[:20]
    ]

    labels = []
    if control_mean > target_mean:
        labels.append("control_dominates_globally")
    family_failures = [row for row in by_family if row["specificity_gap_mean"] < 0]
    if family_failures:
        labels.append("control_dominates_specific_families")
    if target_mean > 0 and specificity_gap <= 0:
        labels.append("target_effect_present_but_nonspecific")
    density_rows = [
        row
        for row in by_feature_set
        if str(row["feature_set_label"]).startswith("density_matched_seed_")
    ]
    if any(row["target_delta_mean"] > 0 for row in density_rows):
        labels.append("density_controls_also_move_targets")
    if feature_score_vs_effect and len(
        {row.get("feature_set_specificity_gap") for row in feature_score_vs_effect}
    ) <= 1:
        labels.append("insufficient_per_feature_attribution")
    if family_failures and len(family_failures) == 1:
        labels.append("specificity_failure_concentrated_in_family")
    template_failures = [row for row in by_template if row["specificity_gap_mean"] < 0]
    if template_failures:
        labels.append("specificity_failure_concentrated_in_template")
    if (
        control_mean > target_mean
        and config.get("control_suite", "matched_non_negation_current")
        == "matched_non_negation_current"
    ):
        labels.append("possible_control_prompt_confound")
    if specificity_gap <= 0 and target_mean > 0:
        labels.append("true_negative_for_current_layer_feature_set")

    summary = {
        "run_dir": str(run_dir),
        "ranking_dir": str(ranking_dir),
        "calibration_dir": str(calibration_dir),
        "claim_status": report.get("claim_status"),
        "calibration_pass_rate": calibration.get("kept_total"),
        "target_absolute_delta_mean": target_mean,
        "matched_control_absolute_delta_mean": control_mean,
        "specificity_gap": specificity_gap,
        "target_control_ratio": ratio,
        "n_behavioral_rows": len(behavioral_rows),
        "n_tasks": len({row.get("task_id") for row in behavioral_rows}),
        "diagnosis_labels": sorted(set(labels)),
        "strongest_family": max(
            by_family,
            key=lambda row: row["specificity_gap_mean"],
            default=None,
        ),
        "weakest_family": min(by_family, key=lambda row: row["specificity_gap_mean"], default=None),
    }

    _write_json(out_dir / "specificity_summary.json", summary)
    _write_csv(
        out_dir / "specificity_by_family.csv",
        by_family,
        list(by_family[0]) if by_family else ["family"],
    )
    _write_csv(
        out_dir / "specificity_by_template.csv",
        by_template,
        list(by_template[0]) if by_template else ["template_id"],
    )
    _write_csv(
        out_dir / "specificity_by_feature_set.csv",
        by_feature_set,
        list(by_feature_set[0]) if by_feature_set else ["feature_set_label"],
    )
    _write_csv(
        out_dir / "specificity_by_feature.csv",
        by_feature,
        list(by_feature[0]) if by_feature else ["feature_id"],
    )
    _write_csv(
        out_dir / "task_level_effects.csv",
        task_effects,
        list(task_effects[0]) if task_effects else ["task_id"],
    )
    _write_csv(
        out_dir / "target_vs_control_outliers.csv",
        outliers,
        list(outliers[0]) if outliers else ["task_id"],
    )
    _write_jsonl(out_dir / "control_dominance_cases.jsonl", control_dominant[:50])
    _write_csv(
        out_dir / "feature_score_vs_effect.csv",
        feature_score_vs_effect,
        list(feature_score_vs_effect[0]) if feature_score_vs_effect else ["feature_id"],
    )

    likely = "target effect is present but matched controls move more"
    if "specificity_failure_concentrated_in_family" in labels and summary["weakest_family"]:
        likely = f"specificity failure is concentrated in {summary['weakest_family']['family']}"
    if "control_dominates_globally" in labels:
        likely = "control movement dominates the top feature-set effect globally"
    diagnosis = f"""# E003 Specificity Failure Diagnosis

Most likely failure mode: {likely}.

This analysis is artifact-only. It does not rerun the model or SAE.

## Summary

- claim status: `{summary['claim_status']}`
- target absolute delta mean: `{target_mean}`
- matched-control absolute delta mean: `{control_mean}`
- specificity gap: `{specificity_gap}`
- target/control ratio: `{ratio}`
- diagnosis labels: `{', '.join(summary['diagnosis_labels'])}`

## Interpretation

The decoded SAE intervention moves logits, but this artifact-backed run does not
show negation-specific movement against the evaluated matched controls. The
labels identify plausible failure modes; they are not causal proof.
"""
    (out_dir / "diagnosis.md").write_text(diagnosis, encoding="utf-8")
    (out_dir / "README.md").write_text(diagnosis, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose E003 specificity failure from artifacts."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ranking-dir", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = diagnose(
        run_dir=Path(args.run_dir),
        ranking_dir=Path(args.ranking_dir),
        calibration_dir=Path(args.calibration_dir),
        out_dir=Path(args.out),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
