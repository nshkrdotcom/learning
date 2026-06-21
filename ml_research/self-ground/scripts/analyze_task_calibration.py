from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_INPUTS = [
    "baseline_task_scores.jsonl",
    "baseline_task_summary.csv",
    "behavioral_tasks.jsonl",
    "behavioral_task_validation.json",
    "excluded_behavioral_tasks.jsonl",
    "behavioral_summary.csv",
    "behavioral_intervention_results.jsonl",
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


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": _mean(ordered),
        "max": ordered[-1],
    }


def _group_key(task_by_id: dict[str, dict[str, Any]], row: dict[str, Any], key: str) -> str:
    task = task_by_id.get(str(row.get("task_id")), {})
    if key == "template":
        metadata = task.get("metadata") or {}
        return str(metadata.get("template_index", metadata.get("template_family", "unknown")))
    if key == "target_token":
        tokens = task.get("target_tokens") or []
        return str(tokens[0] if tokens else "unknown")
    return str(row.get(key) or task.get(key) or "unknown")


def _write_group_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    task_by_id: dict[str, dict[str, Any]],
    group_key: str,
    group_column: str,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(task_by_id, row, group_key)].append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                group_column,
                "n_tasks",
                "intended_direction_pass_count",
                "intended_direction_pass_rate",
                "baseline_margin_mean",
                "baseline_margin_abs_mean",
                "target_effect_abs_mean",
                "control_effect_abs_mean",
                "specificity_gap_mean",
            ],
        )
        writer.writeheader()
        for label in sorted(grouped):
            group_rows = grouped[label]
            margins = [
                value
                for row in group_rows
                if (value := _float(row.get("baseline_prompt_contrast"))) is not None
            ]
            writer.writerow(
                {
                    group_column: label,
                    "n_tasks": len(group_rows),
                    "intended_direction_pass_count": sum(
                        1 for row in group_rows if bool(row.get("intended_direction_pass"))
                    ),
                    "intended_direction_pass_rate": _mean(
                        [1.0 if row.get("intended_direction_pass") else 0.0 for row in group_rows]
                    ),
                    "baseline_margin_mean": _mean(margins),
                    "baseline_margin_abs_mean": _mean([abs(value) for value in margins]),
                    "target_effect_abs_mean": _mean(
                        [
                            value
                            for row in group_rows
                            if (value := _float(row.get("target_effect_abs"))) is not None
                        ]
                    ),
                    "control_effect_abs_mean": _mean(
                        [
                            value
                            for row in group_rows
                            if (value := _float(row.get("control_effect_abs"))) is not None
                        ]
                    ),
                    "specificity_gap_mean": _mean(
                        [
                            value
                            for row in group_rows
                            if (value := _float(row.get("specificity_gap"))) is not None
                        ]
                    ),
                }
            )


def analyze_task_calibration(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_INPUTS if not (run_dir / name).exists()]

    baseline_rows = _read_jsonl(run_dir / "baseline_task_scores.jsonl")
    tasks = _read_jsonl(run_dir / "behavioral_tasks.jsonl")
    validation = (
        _read_json(run_dir / "behavioral_task_validation.json")
        if (run_dir / "behavioral_task_validation.json").exists()
        else {}
    )
    task_by_id = {str(task.get("id")): task for task in tasks}
    interventions = _read_jsonl(run_dir / "behavioral_intervention_results.jsonl")
    top_effect_by_task: dict[str, dict[str, float]] = {}
    for row in interventions:
        if row.get("feature_set_label") != "top":
            continue
        task_id = str(row.get("task_id"))
        top_effect_by_task[task_id] = {
            "target_effect_abs": abs(float(row.get("target_signed_delta", 0.0))),
            "control_effect_abs": abs(float(row.get("control_signed_delta", 0.0))),
            "specificity_gap": float(row.get("specificity_gap", 0.0)),
        }
    enriched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in baseline_rows:
        task_id = str(row.get("task_id"))
        merged = {**row, **top_effect_by_task.get(task_id, {})}
        enriched.append(merged)
        margin = _float(row.get("baseline_prompt_contrast"))
        if not bool(row.get("intended_direction_pass")):
            failures.append(
                {
                    "task_id": task_id,
                    "family": row.get("family"),
                    "reason": "baseline_wrong_direction",
                    "baseline_prompt_contrast": row.get("baseline_prompt_contrast"),
                }
            )
        if margin is not None and abs(margin) < 0.1:
            failures.append(
                {
                    "task_id": task_id,
                    "family": row.get("family"),
                    "reason": "tiny_baseline_margin",
                    "baseline_prompt_contrast": margin,
                    "threshold": 0.1,
                }
            )

    margins = [
        value
        for row in baseline_rows
        if (value := _float(row.get("baseline_prompt_contrast"))) is not None
    ]
    target_scores = [
        value
        for row in baseline_rows
        if (value := _float(row.get("baseline_prompt_target_score"))) is not None
    ]
    control_scores = [
        value
        for row in baseline_rows
        if (value := _float(row.get("baseline_control_target_score"))) is not None
    ]
    pass_count = sum(1 for row in baseline_rows if bool(row.get("intended_direction_pass")))
    candidate_filter = {
        "proposed_filters": [
            {
                "type": "baseline_intended_direction_required",
                "enabled": True,
                "rationale": (
                    "The unpatched model should favor the intended target before "
                    "intervention."
                ),
            },
            {
                "type": "minimum_abs_baseline_margin",
                "enabled": True,
                "value": 0.1,
                "rationale": (
                    "Conservative small margin to remove near-ties without using "
                    "intervention outcomes."
                ),
            },
            {
                "type": "single_token_required",
                "enabled": True,
                "rationale": "Token validation is already enforced before baseline scoring.",
            },
            {
                "type": "family_minimum_count",
                "enabled": True,
                "value": 3,
                "rationale": "Each required family must retain enough calibrated tasks.",
            },
        ],
        "apply_command_flags": (
            "--task-calibration-mode baseline-margin --min-baseline-margin 0.1 "
            "--min-calibrated-tasks-per-family 3 --allow-family-drop false"
        ),
        "not_applied_by_this_analysis": True,
    }
    summary = {
        "run_dir": str(run_dir),
        "missing_required_inputs": missing,
        "total_tasks": len(tasks),
        "baseline_rows": len(baseline_rows),
        "valid_tasks": validation.get("summary", validation).get("valid_tasks"),
        "excluded_tasks": validation.get("summary", validation).get("excluded_tasks"),
        "intended_direction_pass_count": pass_count,
        "intended_direction_pass_rate": pass_count / len(baseline_rows) if baseline_rows else None,
        "baseline_margin_distribution": _quantiles(margins),
        "baseline_target_logit_distribution": _quantiles(target_scores),
        "baseline_control_target_logit_distribution": _quantiles(control_scores),
        "wrong_direction_task_count": sum(
            1 for row in failures if row["reason"] == "baseline_wrong_direction"
        ),
        "tiny_margin_task_count": sum(
            1 for row in failures if row["reason"] == "tiny_baseline_margin"
        ),
        "relation_to_intervention_effect": {
            "top_target_abs_mean_for_passing_baseline": _mean(
                [
                    float(row["target_effect_abs"])
                    for row in enriched
                    if row.get("intended_direction_pass") and "target_effect_abs" in row
                ]
            ),
            "top_target_abs_mean_for_failing_baseline": _mean(
                [
                    float(row["target_effect_abs"])
                    for row in enriched
                    if not row.get("intended_direction_pass") and "target_effect_abs" in row
                ]
            ),
        },
    }

    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "candidate_task_filter.json").write_text(
        json.dumps(candidate_filter, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_group_csv(
        out_dir / "calibration_by_family.csv",
        enriched,
        task_by_id=task_by_id,
        group_key="family",
        group_column="family",
    )
    _write_group_csv(
        out_dir / "calibration_by_template.csv",
        enriched,
        task_by_id=task_by_id,
        group_key="template",
        group_column="template",
    )
    _write_group_csv(
        out_dir / "calibration_by_target_token.csv",
        enriched,
        task_by_id=task_by_id,
        group_key="target_token",
        group_column="target_token",
    )
    with (out_dir / "calibration_task_failures.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = ["task_id", "family", "reason", "baseline_prompt_contrast", "threshold"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failures:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    readme = f"""# E002 Task Calibration Analysis

This artifact-only analysis reads baseline and intervention artifacts from `{run_dir}`.
It does not run the model and does not apply filters.

- intended-direction pass rate: `{summary["intended_direction_pass_rate"]}`
- wrong-direction tasks: `{summary["wrong_direction_task_count"]}`
- tiny-margin tasks at 0.1: `{summary["tiny_margin_task_count"]}`
- candidate filter file: `candidate_task_filter.json`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 3 baseline task calibration artifacts."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = analyze_task_calibration(Path(args.run_dir), Path(args.out))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
