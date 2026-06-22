from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from self_ground.behavioral_tasks import (
    TASK_FAMILY_ORDER,
    BehavioralTask,
    read_behavioral_tasks_jsonl,
)
from self_ground.io import read_json, write_config


def count_tasks_by_family(tasks: list[BehavioralTask]) -> dict[str, int]:
    counts = Counter(task.family for task in tasks)
    return {family: int(counts.get(family, 0)) for family in TASK_FAMILY_ORDER}


def validate_min_tasks_by_family(
    *,
    counts: dict[str, int],
    min_per_family: int,
) -> list[str]:
    if min_per_family < 1:
        raise ValueError("min_per_family must be >= 1")
    return [family for family in TASK_FAMILY_ORDER if int(counts.get(family, 0)) < min_per_family]


def load_task_file_with_minimum(
    *,
    task_file: str | Path,
    min_per_family: int,
) -> tuple[list[BehavioralTask], dict[str, int]]:
    tasks = read_behavioral_tasks_jsonl(task_file)
    counts = count_tasks_by_family(tasks)
    missing = validate_min_tasks_by_family(counts=counts, min_per_family=min_per_family)
    if missing:
        raise ValueError(
            "task file underfills required families for requested per_family="
            f"{min_per_family}: {missing}; counts={counts}"
        )
    return tasks, counts


def write_task_source_artifacts(
    *,
    out_dir: str | Path,
    task_source: str,
    task_file: str | Path | None,
    task_source_id: str | None,
    task_bank_calibration_dir: str | Path | None,
    tasks: list[BehavioralTask],
    min_per_family: int,
) -> dict[str, Any]:
    out = Path(out_dir)
    counts = count_tasks_by_family(tasks)
    payload = {
        "task_source": task_source,
        "task_file": str(task_file) if task_file else None,
        "task_source_id": task_source_id,
        "task_bank_calibration_dir": (
            str(task_bank_calibration_dir) if task_bank_calibration_dir else None
        ),
        "n_tasks": len(tasks),
        "min_per_family": min_per_family,
        "calibrated_task_count_by_family": counts,
        "required_families": list(TASK_FAMILY_ORDER),
    }
    write_config(payload, out / "task_source.json")
    if task_bank_calibration_dir is not None:
        source = Path(task_bank_calibration_dir)
        copies = {
            "calibration_summary.json": "source_calibration_summary.json",
            "calibrated_excluded_behavioral_tasks.jsonl": (
                "source_calibrated_excluded_behavioral_tasks.jsonl"
            ),
            "candidate_baseline_scores.jsonl": "source_candidate_baseline_scores.jsonl",
        }
        for source_name, dest_name in copies.items():
            src = source / source_name
            if src.exists():
                shutil.copyfile(src, out / dest_name)
        summary = source / "calibration_summary.json"
        if summary.exists():
            payload["source_calibration_summary"] = read_json(summary)
            write_config(payload, out / "task_source.json")
    return payload
