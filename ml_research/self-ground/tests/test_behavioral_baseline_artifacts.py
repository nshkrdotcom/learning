from __future__ import annotations

import csv
import json

from self_ground.behavioral_tasks import generate_behavioral_tasks
from self_ground.real_behavioral_intervention import (
    write_baseline_task_artifacts,
)
from self_ground.task_validation import TokenValidationResult


def test_baseline_task_artifacts_have_stable_columns(tmp_path) -> None:
    tasks = generate_behavioral_tasks(per_family=1)
    validations = [
        TokenValidationResult(
            task_id=task.id,
            family=task.family,
            valid=True,
            prompt=task.prompt,
            control_prompt=task.control_prompt,
            control_type="matched_non_negation",
            target_token_ids=[1],
            foil_token_ids=[2],
            control_target_token_ids=[2],
            control_foil_token_ids=[1],
        )
        for task in tasks
    ]
    scores = [
        {
            "task_id": task.id,
            "family": task.family,
            "baseline_prompt_target_score": 2.0,
            "baseline_prompt_foil_score": 1.0,
            "baseline_prompt_contrast": 1.0,
            "baseline_control_target_score": 3.0,
            "baseline_control_foil_score": 1.5,
            "baseline_control_contrast": 1.5,
            "intended_direction_pass": True,
        }
        for task in tasks
    ]

    write_baseline_task_artifacts(
        out_dir=tmp_path,
        tasks=tasks,
        validations=validations,
        baseline_rows=scores,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "baseline_task_scores.jsonl").read_text().splitlines()
    ]
    assert rows
    assert "baseline_prompt_contrast" in rows[0]
    with (tmp_path / "baseline_task_summary.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == [
        "family",
        "n_tasks",
        "prompt_contrast_mean",
        "prompt_contrast_abs_mean",
        "control_contrast_mean",
        "control_contrast_abs_mean",
        "intended_direction_pass_rate",
    ]
