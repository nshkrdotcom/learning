from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from self_ground.mechanism_report import REQUIRED_EVIDENCE_ARTIFACTS

INSPECTION_REQUIRED_ARTIFACTS = [
    *REQUIRED_EVIDENCE_ARTIFACTS,
    "baseline_validation.json",
    "mechanism_report.json",
]


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


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_vs_control_ratio(summary_rows: list[dict[str, str]]) -> float | None:
    top_rows = [
        row
        for row in summary_rows
        if row.get("family") == "__all__" and row.get("feature_set_label") == "top"
    ]
    if not top_rows:
        return None
    top_row = top_rows[0]
    top_value = _float_or_none(top_row.get("target_absolute_delta_mean"))
    if top_value is None:
        return None
    controls = [
        row
        for row in summary_rows
        if row.get("family") == "__all__"
        and row.get("feature_set_label") != "top"
        and row.get("operation") == top_row.get("operation")
        and row.get("factor") == top_row.get("factor")
        and row.get("patch_mode") == top_row.get("patch_mode")
        and row.get("control_suite", "matched_non_negation_current")
        == top_row.get("control_suite", "matched_non_negation_current")
    ]
    control_values = [
        value
        for row in controls
        if (value := _float_or_none(row.get("target_absolute_delta_mean"))) is not None
    ]
    if not control_values:
        return None
    control_mean = sum(control_values) / len(control_values)
    if control_mean <= 0:
        return None
    return top_value / control_mean


def _classify_run(config: dict[str, Any], feature_set_rows: list[dict[str, Any]]) -> str:
    per_family = int(config.get("per_family") or 0)
    top_k = int(config.get("top_k_features") or 0)
    device = str(config.get("device") or "")
    labels = {str(row.get("label")) for row in feature_set_rows}
    density_present = any(label.startswith("density_matched_seed_") for label in labels)
    random_present = any(label.startswith("random_seed_") for label in labels)
    bottom_active_present = "bottom_active" in labels
    if (
        device.startswith("cuda")
        and per_family >= 10
        and top_k >= 5
        and density_present
        and random_present
        and bottom_active_present
    ):
        return "serious_gpu_evidence_run"
    if device == "cpu" and per_family >= 3 and top_k >= 5 and density_present:
        return "large_cpu_diagnostic"
    return "diagnostic_or_smoke_run"


def inspect_claim_run(run_dir: Path, *, allow_missing: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir)
    missing = [
        name
        for name in INSPECTION_REQUIRED_ARTIFACTS
        if not (run_dir / name).exists()
    ]
    if missing and not allow_missing:
        return {
            "run_dir": str(run_dir),
            "ok": False,
            "missing_required_artifacts": missing,
            "error": "missing required artifacts",
        }

    config = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    compatibility = (
        _read_json(run_dir / "compatibility.json")
        if (run_dir / "compatibility.json").exists()
        else {}
    )
    validation = (
        _read_json(run_dir / "behavioral_task_validation.json")
        if (run_dir / "behavioral_task_validation.json").exists()
        else {}
    )
    validation_summary = validation.get("summary", validation)
    feature_sets = (
        _read_json(run_dir / "feature_sets.json")
        if (run_dir / "feature_sets.json").exists()
        else {"feature_sets": []}
    )
    control_suite = (
        _read_json(run_dir / "control_suite.json")
        if (run_dir / "control_suite.json").exists()
        else {}
    )
    task_source = (
        _read_json(run_dir / "task_source.json")
        if (run_dir / "task_source.json").exists()
        else {}
    )
    feature_set_rows = list(feature_sets.get("feature_sets", []))
    behavioral_rows = _read_jsonl(run_dir / "behavioral_intervention_results.jsonl")
    skipped = (
        _read_json(run_dir / "skipped_behavioral_rows.json")
        if (run_dir / "skipped_behavioral_rows.json").exists()
        else {}
    )
    report = (
        _read_json(run_dir / "mechanism_report.json")
        if (run_dir / "mechanism_report.json").exists()
        else {}
    )
    summary_rows = _read_csv(run_dir / "behavioral_summary.csv")
    all_rows = [row for row in summary_rows if row.get("family") == "__all__"]
    top_summary = next((row for row in all_rows if row.get("feature_set_label") == "top"), {})
    top_control_suite_rows = [
        row for row in all_rows if row.get("feature_set_label") == "top"
    ]
    per_control_suite = [
        {
            "control_suite": row.get("control_suite", "matched_non_negation_current"),
            "operation": row.get("operation"),
            "target_absolute_delta_mean": _float_or_none(
                row.get("target_absolute_delta_mean")
            ),
            "control_absolute_delta_mean": _float_or_none(
                row.get("control_absolute_delta_mean")
            ),
            "specificity_gap_mean": _float_or_none(row.get("specificity_gap_mean")),
            "collateral_ratio_mean": _float_or_none(row.get("collateral_ratio_mean")),
        }
        for row in top_control_suite_rows
    ]
    multi_control_min_gap = min(
        [
            value
            for row in per_control_suite
            if (value := row["specificity_gap_mean"]) is not None
        ],
        default=None,
    )
    top_features = next(
        (
            list(row.get("feature_ids", []))
            for row in feature_set_rows
            if row.get("label") == "top"
        ),
        [],
    )
    density_sets = [
        row
        for row in feature_set_rows
        if str(row.get("label", "")).startswith("density_matched_seed_")
        or row.get("selection_method") == "activation_density_matched"
    ]
    density_metadata = [row.get("matched_control_metadata", {}) for row in density_sets]
    summary = {
        "run_dir": str(run_dir),
        "ok": not missing,
        "missing_required_artifacts": missing,
        "run_classification": _classify_run(config, feature_set_rows),
        "config": {
            "model_name": config.get("model_name"),
            "hook_point": config.get("hook_point"),
            "sae_release": config.get("sae_release"),
            "sae_id": config.get("sae_id"),
            "engine_backend": config.get("engine_backend"),
            "sae_backend": config.get("sae_backend"),
            "evaluation_adapter": config.get("evaluation_adapter"),
            "baseline_mode": config.get("baseline_mode"),
            "control_suite": config.get("control_suite"),
            "per_family": config.get("per_family"),
            "top_k_features": config.get("top_k_features"),
            "device": config.get("device"),
        },
        "compatibility": {
            "compatible": compatibility.get("compatible"),
            "metadata_compatible": compatibility.get("metadata_compatible"),
            "shape_compatible": compatibility.get("shape_compatible"),
            "reconstruction_compatible": compatibility.get("reconstruction_compatible"),
            "diagnostic_only": compatibility.get("diagnostic_only"),
            "declared_model": compatibility.get("declared_model"),
            "declared_hook_point": compatibility.get("declared_hook_point"),
        },
        "task_validation": {
            "passes_minimum": validation_summary.get("passes_minimum"),
            "total_tasks": validation_summary.get("total_tasks"),
            "valid_tasks": validation_summary.get("valid_tasks"),
            "excluded_tasks": validation_summary.get("excluded_tasks"),
            "valid_by_family": validation_summary.get("valid_by_family"),
        },
        "task_source": {
            "task_source": task_source.get("task_source", config.get("task_source")),
            "task_source_id": task_source.get("task_source_id", config.get("task_source_id")),
            "task_file": task_source.get("task_file", config.get("task_file")),
            "task_bank_calibration_dir": task_source.get(
                "task_bank_calibration_dir",
                config.get("task_bank_calibration_dir"),
            ),
            "calibrated_task_count_by_family": task_source.get(
                "calibrated_task_count_by_family"
            ),
        },
        "feature_sets": {
            "labels": [row.get("label") for row in feature_set_rows],
            "top_feature_ids": top_features,
            "density_matched_present": bool(density_sets),
            "density_matched_count": len(density_sets),
            "density_stats_sources": sorted(
                {
                    str(metadata.get("stats_source"))
                    for metadata in density_metadata
                    if metadata.get("stats_source")
                }
            ),
            "density_relaxed": any(bool(metadata.get("relaxed")) for metadata in density_metadata),
        },
        "control_suite": {
            "mode": control_suite.get("control_suite", config.get("control_suite")),
            "expanded_suites": control_suite.get("expanded_suites"),
        },
        "rows": {
            "behavioral_rows": len(behavioral_rows),
            "skipped_rows": skipped.get("n_skipped_rows"),
            "skipped_reason_counts": skipped.get("reason_counts", {}),
        },
        "claim": {
            "claim_status": report.get("claim_status"),
            "recommended_claim": report.get("recommended_claim"),
            "blocker_reason": report.get("blocker_reason"),
            "limitations": report.get("limitations", []),
        },
        "metrics": {
            "top_target_delta": _float_or_none(top_summary.get("target_absolute_delta_mean")),
            "top_control_delta": _float_or_none(top_summary.get("control_absolute_delta_mean")),
            "specificity_gap": _float_or_none(top_summary.get("specificity_gap_mean")),
            "top_vs_control_ratio": _top_vs_control_ratio(summary_rows),
            "multi_control_min_specificity_gap": multi_control_min_gap,
            "per_control_suite": per_control_suite,
        },
    }
    return summary


def _print_human(summary: dict[str, Any]) -> None:
    print(f"run_dir: {summary['run_dir']}")
    print(f"ok: {summary['ok']}")
    if summary["missing_required_artifacts"]:
        print("missing_required_artifacts: " + ", ".join(summary["missing_required_artifacts"]))
    print(f"run_classification: {summary['run_classification']}")
    print("\nconfig:")
    for key, value in summary["config"].items():
        print(f"  {key}: {value}")
    print("\ncompatibility:")
    for key, value in summary["compatibility"].items():
        print(f"  {key}: {value}")
    print("\ntask_validation:")
    for key, value in summary["task_validation"].items():
        print(f"  {key}: {value}")
    print("\ntask_source:")
    for key, value in summary["task_source"].items():
        print(f"  {key}: {value}")
    print("\nfeature_sets:")
    for key, value in summary["feature_sets"].items():
        print(f"  {key}: {value}")
    print("\ncontrol_suite:")
    for key, value in summary["control_suite"].items():
        print(f"  {key}: {value}")
    print("\nrows:")
    for key, value in summary["rows"].items():
        print(f"  {key}: {value}")
    print("\nclaim:")
    for key, value in summary["claim"].items():
        print(f"  {key}: {value}")
    print("\nmetrics:")
    for key, value in summary["metrics"].items():
        print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a SELF-GROUND claim run from artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    summary = inspect_claim_run(Path(args.run_dir), allow_missing=args.allow_missing)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human(summary)
    return 0 if summary["ok"] or args.allow_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
