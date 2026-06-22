from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from self_ground.behavioral_tasks import TASK_FAMILY_ORDER, write_behavioral_tasks_jsonl
from self_ground.io import read_json, write_config, write_jsonl
from self_ground.real_behavioral_intervention import _baseline_scores
from self_ground.task_bank import CandidateTaskBank, task_bank_to_behavioral_tasks
from self_ground.task_calibration import (
    TaskCalibrationRule,
    apply_task_calibration,
)
from self_ground.task_validation import validate_behavioral_tasks


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_group_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    group_label: str,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "unknown"))].append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                group_label,
                "n_candidates",
                "kept",
                "intended_direction_pass_rate",
                "margin_abs_mean",
            ],
        )
        writer.writeheader()
        for label in sorted(grouped):
            group_rows = grouped[label]
            margins = [abs(float(row["baseline_prompt_contrast"])) for row in group_rows]
            writer.writerow(
                {
                    group_label: label,
                    "n_candidates": len(group_rows),
                    "kept": sum(1 for row in group_rows if row.get("kept")),
                    "intended_direction_pass_rate": _mean(
                        [1.0 if row.get("intended_direction_pass") else 0.0 for row in group_rows]
                    ),
                    "margin_abs_mean": _mean(margins),
                }
            )


def calibrate_task_bank(
    *,
    task_bank_path: Path,
    out_dir: Path,
    model_name: str,
    device: str,
    min_baseline_margin: float,
    min_per_family: int,
    model_adapter=None,
) -> dict[str, Any]:
    if min_per_family < 1:
        raise ValueError("min_per_family must be >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task_bank": str(task_bank_path),
        "model_name": model_name,
        "device": device,
        "min_baseline_margin": min_baseline_margin,
        "min_per_family": min_per_family,
    }
    write_config(config, out_dir / "config.json")

    bank = CandidateTaskBank.model_validate(read_json(task_bank_path))
    tasks = task_bank_to_behavioral_tasks(bank)
    if model_adapter is None:
        from self_ground.model import TransformerLensModelAdapter

        model_adapter = TransformerLensModelAdapter(model_name=model_name, device=device)

    valid_tasks, validation_results, validation_summary = validate_behavioral_tasks(
        model_adapter=model_adapter,
        tasks=tasks,
        min_valid_tasks_per_family=1,
    )
    invalid_exclusions = [
        {
            "task_id": result.task_id,
            "family": result.family,
            "reason": "tokenization_failed",
            "details": result.excluded_reason,
        }
        for result in validation_results
        if not result.valid
    ]
    baseline_rows = _baseline_scores(
        model_adapter=model_adapter,
        tasks=valid_tasks,
        validations=validation_results,
        reduction="mean",
    )
    task_by_id = {task.id: task for task in valid_tasks}
    rule = TaskCalibrationRule(
        require_baseline_intended_direction=True,
        min_abs_baseline_margin=min_baseline_margin,
        min_tasks_per_family=min_per_family,
        allow_family_drop=False,
        reason=(
            "Task-bank calibration keeps only baseline intended-direction tasks "
            "with sufficient absolute prompt margin before intervention."
        ),
    )
    result = apply_task_calibration(
        tasks=valid_tasks,
        baseline_rows=list(baseline_rows.values()),
        rule=rule,
    )
    kept_ids = set(result.kept_task_ids)
    enriched_baseline: list[dict[str, Any]] = []
    for row in baseline_rows.values():
        task = task_by_id[str(row["task_id"])]
        enriched_baseline.append(
            {
                **row,
                "template_id": task.metadata.get("template_id"),
                "template_family": task.metadata.get("template_family"),
                "target_tokens": task.target_tokens,
                "foil_tokens": task.foil_tokens,
                "kept": task.id in kept_ids,
            }
        )
    kept_tasks = [task for task in valid_tasks if task.id in kept_ids]
    all_excluded = [*invalid_exclusions, *result.excluded]
    kept_counts = Counter(task.family for task in kept_tasks)
    excluded_counts = Counter(str(row["reason"]) for row in all_excluded)
    summary = {
        "model_name": model_name,
        "task_bank": str(task_bank_path),
        "total_candidates": len(tasks),
        "token_valid_candidates": len(valid_tasks),
        "min_baseline_margin": min_baseline_margin,
        "min_per_family": min_per_family,
        "passes_minimum": result.passes_minimum,
        "kept_total": len(kept_tasks),
        "kept_by_family": {
            family: int(kept_counts.get(family, 0)) for family in TASK_FAMILY_ORDER
        },
        "valid_by_family_before": result.valid_by_family_before,
        "excluded_by_reason": {
            key: int(excluded_counts.get(key, 0)) for key in sorted(excluded_counts)
        },
        "missing_required_families": result.missing_required_families,
    }

    write_jsonl(enriched_baseline, out_dir / "candidate_baseline_scores.jsonl")
    write_behavioral_tasks_jsonl(kept_tasks, out_dir / "calibrated_behavioral_tasks.jsonl")
    write_jsonl(all_excluded, out_dir / "calibrated_excluded_behavioral_tasks.jsonl")
    write_config(summary, out_dir / "calibration_summary.json")
    write_config(result.rule.model_dump(mode="json"), out_dir / "task_calibration_rule.json")
    write_config(result.model_dump(mode="json"), out_dir / "task_calibration_result.json")
    _write_group_csv(
        out_dir / "calibration_by_family.csv",
        enriched_baseline,
        group_key="family",
        group_label="family",
    )
    _write_group_csv(
        out_dir / "calibration_by_template.csv",
        enriched_baseline,
        group_key="template_family",
        group_label="template_family",
    )
    if not result.passes_minimum:
        write_config(
            {
                "blocker_type": "task_bank_calibration_failed",
                "reason": "calibrated task bank did not retain enough tasks per required family",
                "summary": summary,
                "no_intervention_rows_written": True,
            },
            out_dir / "blocker.json",
        )
    else:
        stale_blocker = out_dir / "blocker.json"
        if stale_blocker.exists():
            stale_blocker.unlink()
    readme = f"""# Phase 3 Task Bank Calibration

- model: `{model_name}`
- task bank: `{task_bank_path}`
- min baseline margin: `{min_baseline_margin}`
- min per family: `{min_per_family}`
- passes minimum: `{result.passes_minimum}`
- kept by family: `{summary["kept_by_family"]}`
- excluded by reason: `{summary["excluded_by_reason"]}`

This run scores baseline target/foil contrasts only. It does not run decoded
SAE interventions.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline-calibrate a Phase 3 task bank.")
    parser.add_argument("--task-bank", required=True)
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-baseline-margin", type=float, default=0.1)
    parser.add_argument("--min-per-family", type=int, default=10)
    args = parser.parse_args()
    try:
        summary = calibrate_task_bank(
            task_bank_path=Path(args.task_bank),
            out_dir=Path(args.out),
            model_name=args.model,
            device=args.device,
            min_baseline_margin=args.min_baseline_margin,
            min_per_family=args.min_per_family,
        )
    except Exception as exc:
        print(f"task bank calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passes_minimum"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
