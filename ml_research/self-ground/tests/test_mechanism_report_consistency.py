from __future__ import annotations

import csv
import json

from self_ground.io import write_config, write_jsonl
from self_ground.mechanism_report import build_mechanism_evidence_report


def _write_minimal_report_inputs(run_dir) -> None:
    write_config(
        {
            "model_name": "test-local",
            "hook_point": "blocks.2.hook_resid_post",
            "sae_release": "test-release",
            "sae_id": "blocks.2.hook_resid_post",
            "allow_metadata_mismatch": False,
            "baseline_mode": "top-vs-random",
            "random_seeds": [7],
            "operations": ["ablate"],
        },
        run_dir / "config.json",
    )
    write_config(
        {
            "compatible": True,
            "diagnostic_only": False,
            "metadata_compatible": True,
            "shape_compatible": True,
            "reconstruction_compatible": True,
        },
        run_dir / "compatibility.json",
    )
    write_config(
        {
            "summary": {
                "passes_minimum": True,
                "total_tasks": 6,
                "valid_tasks": 6,
                "excluded_tasks": 0,
                "valid_by_family": {"sentiment_negation": 3, "property_negation": 3},
                "excluded_by_family": {"sentiment_negation": 0, "property_negation": 0},
                "min_valid_tasks_per_family": 2,
                "required_families": ["sentiment_negation", "property_negation"],
                "missing_required_families": [],
            }
        },
        run_dir / "behavioral_task_validation.json",
    )
    write_jsonl(
        [
            {
                "id": "sentiment",
                "family": "sentiment_negation",
                "concept": "movie",
                "prompt": "The movie was not good. The movie was",
                "target_tokens": [" bad"],
                "foil_tokens": [" good"],
                "control_prompt": "The movie was good. The movie was",
                "control_type": "matched_non_negation",
                "control_target_tokens": [" good"],
                "control_foil_tokens": [" bad"],
                "expected_baseline_direction": "positive",
                "metadata": {},
            },
            {
                "id": "property",
                "family": "property_negation",
                "concept": "animal",
                "prompt": "The animal is not dangerous. The animal is",
                "target_tokens": [" safe"],
                "foil_tokens": [" dangerous"],
                "control_prompt": "The animal is dangerous. The animal is",
                "control_type": "matched_non_negation",
                "control_target_tokens": [" dangerous"],
                "control_foil_tokens": [" safe"],
                "expected_baseline_direction": "positive",
                "metadata": {},
            },
        ],
        run_dir / "behavioral_tasks.jsonl",
    )
    write_jsonl([], run_dir / "excluded_behavioral_tasks.jsonl")
    write_config(
        {
            "feature_sets": [
                {"label": "top", "feature_ids": ["sae_0", "sae_1"]},
                {"label": "random_seed_7", "feature_ids": ["sae_2", "sae_3"]},
            ]
        },
        run_dir / "feature_sets.json",
    )
    write_jsonl(
        [
            {
                "task_id": "sentiment",
                "family": "sentiment_negation",
                "baseline_prompt_target_score": 1.0,
                "baseline_prompt_foil_score": 0.0,
                "baseline_prompt_contrast": 1.0,
                "baseline_control_target_score": 1.0,
                "baseline_control_foil_score": 0.0,
                "baseline_control_contrast": 1.0,
                "intended_direction_pass": True,
            },
            {
                "task_id": "property",
                "family": "property_negation",
                "baseline_prompt_target_score": 1.0,
                "baseline_prompt_foil_score": 0.0,
                "baseline_prompt_contrast": 1.0,
                "baseline_control_target_score": 1.0,
                "baseline_control_foil_score": 0.0,
                "baseline_control_contrast": 1.0,
                "intended_direction_pass": True,
            },
        ],
        run_dir / "baseline_task_scores.jsonl",
    )
    with (run_dir / "baseline_task_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "n_tasks",
                "prompt_contrast_mean",
                "prompt_contrast_abs_mean",
                "control_contrast_mean",
                "control_contrast_abs_mean",
                "intended_direction_pass_rate",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "family": "sentiment_negation",
                "n_tasks": 3,
                "prompt_contrast_mean": 1.0,
                "prompt_contrast_abs_mean": 1.0,
                "control_contrast_mean": 1.0,
                "control_contrast_abs_mean": 1.0,
                "intended_direction_pass_rate": 1.0,
            }
        )
        writer.writerow(
            {
                "family": "property_negation",
                "n_tasks": 3,
                "prompt_contrast_mean": 1.0,
                "prompt_contrast_abs_mean": 1.0,
                "control_contrast_mean": 1.0,
                "control_contrast_abs_mean": 1.0,
                "intended_direction_pass_rate": 1.0,
            }
        )
    write_config(
        {"finite": True, "n_rows": 2, "n_nonfinite_rows": 0, "nonfinite_fields": []},
        run_dir / "baseline_validation.json",
    )
    fieldnames = [
        "feature_set_label",
        "feature_selection_method",
        "operation",
        "factor",
        "patch_mode",
        "family",
        "n_tasks",
        "target_absolute_delta_mean",
        "control_absolute_delta_mean",
        "specificity_gap_mean",
        "collateral_ratio_mean",
        "n_null_collateral_ratio",
        "baseline_contrast_mean",
        "relative_norm_drift_mean",
        "decoded_delta_norm_mean",
        "norm_drift_warning_rate",
    ]
    rows = [
        {
            "feature_set_label": "top",
            "feature_selection_method": "ranking_abs_score_top_k",
            "operation": "ablate",
            "factor": "",
            "patch_mode": "delta",
            "family": "__all__",
            "n_tasks": 6,
            "target_absolute_delta_mean": 0.25,
            "control_absolute_delta_mean": 0.02,
            "specificity_gap_mean": 0.23,
            "collateral_ratio_mean": 0.08,
            "n_null_collateral_ratio": 0,
            "baseline_contrast_mean": 1.0,
            "relative_norm_drift_mean": 0.1,
            "decoded_delta_norm_mean": 0.2,
            "norm_drift_warning_rate": 0.0,
        },
        {
            "feature_set_label": "random_seed_7",
            "feature_selection_method": "seeded_random_excluding_top_fraction",
            "operation": "ablate",
            "factor": "",
            "patch_mode": "delta",
            "family": "__all__",
            "n_tasks": 6,
            "target_absolute_delta_mean": 0.05,
            "control_absolute_delta_mean": 0.01,
            "specificity_gap_mean": 0.04,
            "collateral_ratio_mean": 0.2,
            "n_null_collateral_ratio": 0,
            "baseline_contrast_mean": 1.0,
            "relative_norm_drift_mean": 0.1,
            "decoded_delta_norm_mean": 0.1,
            "norm_drift_warning_rate": 0.0,
        },
    ]
    with (run_dir / "behavioral_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(
        [{"family": "sentiment_negation", "target_absolute_delta": 0.25}],
        run_dir / "behavioral_intervention_results.jsonl",
    )
    write_config(
        {"n_skipped_rows": 0, "reason_counts": {}, "examples": []},
        run_dir / "skipped_behavioral_rows.json",
    )


def test_report_values_match_behavioral_summary(tmp_path) -> None:
    _write_minimal_report_inputs(tmp_path)

    report = build_mechanism_evidence_report(
        behavioral_run_dir=tmp_path,
        out_json=tmp_path / "mechanism_report.json",
    )
    saved = json.loads((tmp_path / "mechanism_report.json").read_text())

    assert report.feature_sets[0].target_absolute_delta_mean == 0.25
    assert saved["feature_sets"][0]["target_absolute_delta_mean"] == 0.25
