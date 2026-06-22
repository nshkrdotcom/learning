from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_e002_e003.py"
SPEC = importlib.util.spec_from_file_location("compare_e002_e003", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_module)
compare_e002_e003 = compare_module.compare_e002_e003


def _write_run(
    path: Path,
    *,
    status: str = "insufficient_evidence",
    specificity: float = 0.1,
    task_source: str = "generated",
    blocker_reason: str | None = None,
) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_name": "test-local",
                "hook_point": "blocks.2.hook_resid_post",
                "sae_release": "test-release",
                "sae_id": "blocks.2.hook_resid_post",
                "engine_backend": "transformer_lens",
                "sae_backend": "sae_lens",
                "evaluation_adapter": "negation_ravel_adapter",
                "baseline_mode": "top-vs-random-density-and-bottom-active",
                "per_family": 10,
                "top_k_features": 5,
                "device": "cuda",
                "task_source": task_source,
            }
        ),
        encoding="utf-8",
    )
    (path / "task_source.json").write_text(
        json.dumps(
            {
                "task_source": task_source,
                "task_source_id": "unit",
                "calibrated_task_count_by_family": {
                    "sentiment_negation": 10,
                    "property_negation": 10,
                    "state_negation": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "behavioral_task_validation.json").write_text(
        json.dumps(
            {
                "summary": {
                    "passes_minimum": True,
                    "valid_tasks": 30,
                    "excluded_tasks": 0,
                    "valid_by_family": {
                        "sentiment_negation": 10,
                        "property_negation": 10,
                        "state_negation": 10,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (path / "compatibility.json").write_text(
        json.dumps(
            {
                "compatible": True,
                "metadata_compatible": True,
                "shape_compatible": True,
                "reconstruction_compatible": True,
            }
        ),
        encoding="utf-8",
    )
    (path / "behavioral_tasks.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"t{idx}", "family": "sentiment_negation"})
            for idx in range(30)
        ),
        encoding="utf-8",
    )
    (path / "excluded_behavioral_tasks.jsonl").write_text("", encoding="utf-8")
    (path / "mechanism_report.json").write_text(
        json.dumps(
            {
                "claim_status": status,
                "recommended_claim": "unit claim",
                "blocker_reason": blocker_reason,
                "limitations": [],
            }
        ),
        encoding="utf-8",
    )
    (path / "baseline_task_scores.jsonl").write_text(
        "\n".join(
            json.dumps({"task_id": f"t{idx}", "intended_direction_pass": True})
            for idx in range(30)
        ),
        encoding="utf-8",
    )
    with (path / "baseline_task_summary.csv").open("w", newline="", encoding="utf-8") as handle:
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
        for family in ["sentiment_negation", "property_negation", "state_negation"]:
            writer.writerow(
                {
                    "family": family,
                    "n_tasks": 10,
                    "prompt_contrast_mean": 1.0,
                    "prompt_contrast_abs_mean": 1.0,
                    "control_contrast_mean": 1.0,
                    "control_contrast_abs_mean": 1.0,
                    "intended_direction_pass_rate": 1.0,
                }
            )
    (path / "baseline_validation.json").write_text(
        json.dumps({"finite": True, "n_nonfinite_rows": 0, "nonfinite_fields": []}),
        encoding="utf-8",
    )
    with (path / "behavioral_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_set_label",
                "operation",
                "factor",
                "patch_mode",
                "family",
                "target_absolute_delta_mean",
                "control_absolute_delta_mean",
                "specificity_gap_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "feature_set_label": "top",
                "operation": "ablate",
                "factor": "",
                "patch_mode": "delta",
                "family": "__all__",
                "target_absolute_delta_mean": 0.3,
                "control_absolute_delta_mean": 0.1,
                "specificity_gap_mean": specificity,
            }
        )
        writer.writerow(
            {
                "feature_set_label": "density_matched_seed_7",
                "operation": "ablate",
                "factor": "",
                "patch_mode": "delta",
                "family": "__all__",
                "target_absolute_delta_mean": 0.1,
                "control_absolute_delta_mean": 0.1,
                "specificity_gap_mean": 0.0,
            }
        )
    (path / "feature_sets.json").write_text(
        json.dumps(
            {
                "feature_sets": [
                    {"label": "top", "feature_ids": ["sae_1"]},
                    {
                        "label": "density_matched_seed_7",
                        "feature_ids": ["sae_9"],
                        "selection_method": "activation_density_matched",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (path / "skipped_behavioral_rows.json").write_text(
        json.dumps({"n_skipped_rows": 0, "reason_counts": {}}),
        encoding="utf-8",
    )
    (path / "behavioral_intervention_results.jsonl").write_text(
        json.dumps({"family": "sentiment_negation", "target_absolute_delta": 0.3}) + "\n",
        encoding="utf-8",
    )


def test_compare_e002_e003_completed_runs(tmp_path) -> None:
    e002 = tmp_path / "e002"
    e003 = tmp_path / "e003"
    out = tmp_path / "comparison"
    _write_run(e002, specificity=-0.02)
    _write_run(e003, specificity=0.05, task_source="file")

    payload = compare_e002_e003(e002, e003, out)

    assert payload["interpretation_category"] == (
        "calibration_fixed_task_suite_but_feature_specificity_still_failed"
    )
    assert payload["specificity_gap_delta"] > 0
    assert (out / "comparison.csv").exists()
    assert (out / "family_comparison.csv").exists()


def test_compare_e002_e003_blocked_calibration_is_deterministic(tmp_path) -> None:
    e002 = tmp_path / "e002"
    e003 = tmp_path / "e003"
    out = tmp_path / "comparison"
    _write_run(e002, specificity=-0.02)
    _write_run(
        e003,
        status="blocked",
        task_source="file",
        blocker_reason="task bank calibration failed",
    )

    payload = compare_e002_e003(e002, e003, out)

    assert payload["interpretation_category"] == "calibration_failed_to_build_task_suite"
