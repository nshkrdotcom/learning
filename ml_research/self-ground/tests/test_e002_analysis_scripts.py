from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_feature_selection import analyze_feature_selection
from scripts.analyze_task_calibration import analyze_task_calibration
from scripts.compare_e002_variants import compare_variants


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_calibration_fixture(run_dir) -> None:
    run_dir.mkdir(parents=True)
    tasks = [
        {
            "id": "task_pass",
            "family": "sentiment_negation",
            "target_tokens": [" bad"],
            "metadata": {"template_index": 0},
        },
        {
            "id": "task_fail",
            "family": "property_negation",
            "target_tokens": [" safe"],
            "metadata": {"template_index": 1},
        },
    ]
    _write_jsonl(run_dir / "behavioral_tasks.jsonl", tasks)
    _write_jsonl(
        run_dir / "baseline_task_scores.jsonl",
        [
            {
                "task_id": "task_pass",
                "family": "sentiment_negation",
                "baseline_prompt_target_score": 2.0,
                "baseline_prompt_foil_score": 1.0,
                "baseline_prompt_contrast": 1.0,
                "baseline_control_target_score": 2.0,
                "baseline_control_foil_score": 1.0,
                "baseline_control_contrast": 1.0,
                "intended_direction_pass": True,
            },
            {
                "task_id": "task_fail",
                "family": "property_negation",
                "baseline_prompt_target_score": 0.9,
                "baseline_prompt_foil_score": 1.0,
                "baseline_prompt_contrast": -0.1,
                "baseline_control_target_score": 2.0,
                "baseline_control_foil_score": 1.0,
                "baseline_control_contrast": 1.0,
                "intended_direction_pass": False,
            },
        ],
    )
    _write_jsonl(run_dir / "excluded_behavioral_tasks.jsonl", [])
    _write_jsonl(
        run_dir / "behavioral_intervention_results.jsonl",
        [
            {
                "task_id": "task_pass",
                "feature_set_label": "top",
                "target_signed_delta": 0.2,
                "control_signed_delta": 0.1,
                "specificity_gap": 0.1,
            },
            {
                "task_id": "task_fail",
                "feature_set_label": "top",
                "target_signed_delta": 0.0,
                "control_signed_delta": 0.2,
                "specificity_gap": -0.2,
            },
        ],
    )
    (run_dir / "behavioral_task_validation.json").write_text(
        json.dumps({"summary": {"valid_tasks": 2, "excluded_tasks": 0}}),
        encoding="utf-8",
    )
    (run_dir / "mechanism_report.json").write_text(
        json.dumps({"claim_status": "insufficient_evidence"}),
        encoding="utf-8",
    )
    for filename in ["baseline_task_summary.csv", "behavioral_summary.csv"]:
        with (run_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["family", "n_tasks"])
            writer.writeheader()
            writer.writerow({"family": "__all__", "n_tasks": 2})


def test_analyze_task_calibration_writes_candidate_filter(tmp_path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "analysis"
    _write_calibration_fixture(run_dir)

    summary = analyze_task_calibration(run_dir, out_dir)

    assert summary["intended_direction_pass_count"] == 1
    assert (out_dir / "candidate_task_filter.json").exists()
    failures = (out_dir / "calibration_task_failures.csv").read_text(encoding="utf-8")
    assert "baseline_wrong_direction" in failures
    assert (out_dir / "calibration_by_family.csv").exists()


def _write_feature_fixture(ranking_dir, eval_dir) -> None:
    ranking_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    with (ranking_dir / "feature_rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_id",
                "score",
                "abs_score",
                "mean_pos",
                "mean_neg",
                "mean_para",
                "mean_decoy",
            ],
        )
        writer.writeheader()
        for feature_id, score in [("sae_0", 1.0), ("sae_1", 0.9), ("sae_5", 0.95)]:
            writer.writerow(
                {
                    "feature_id": feature_id,
                    "score": score,
                    "abs_score": abs(score),
                    "mean_pos": 1.0,
                    "mean_neg": 0.8,
                    "mean_para": 0.9,
                    "mean_decoy": 0.7,
                }
            )
    (eval_dir / "feature_sets.json").write_text(
        json.dumps(
            {
                "feature_sets": [
                    {
                        "label": "top",
                        "selection_method": "ranking_abs_score_top_k",
                        "feature_ids": ["sae_0", "sae_1"],
                    },
                    {
                        "label": "density_matched_seed_7",
                        "selection_method": "activation_density_matched",
                        "feature_ids": ["sae_5"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    with (eval_dir / "behavioral_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_set_label",
                "operation",
                "factor",
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
                "family": "__all__",
                "target_absolute_delta_mean": "0.1",
                "control_absolute_delta_mean": "0.2",
                "specificity_gap_mean": "-0.1",
            }
        )


def test_analyze_feature_selection_writes_diagnosis(tmp_path) -> None:
    ranking_dir = tmp_path / "ranking"
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "feature_analysis"
    _write_feature_fixture(ranking_dir, eval_dir)

    result = analyze_feature_selection(ranking_dir, eval_dir, out_dir)

    assert "control_effect_dominates" in result["diagnosis_labels"]
    assert (out_dir / "selected_feature_table.csv").exists()
    assert (out_dir / "feature_set_effects.csv").exists()


def test_compare_variants_writes_summary(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_calibration_fixture(run_dir)
    # Complete the minimal inspection artifacts.
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "device": "cuda",
                "per_family": 10,
                "top_k_features": 5,
                "task_calibration_mode": "baseline-intended-direction",
                "feature_selection_mode": "top-positive",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "compatibility.json").write_text(json.dumps({"compatible": True}), encoding="utf-8")
    (run_dir / "feature_sets.json").write_text(
        json.dumps({"feature_sets": [{"label": "top", "feature_ids": ["sae_1"]}]}),
        encoding="utf-8",
    )
    (run_dir / "baseline_validation.json").write_text(
        json.dumps({"finite": True}),
        encoding="utf-8",
    )
    (run_dir / "skipped_behavioral_rows.json").write_text(
        json.dumps({"n_skipped_rows": 0}),
        encoding="utf-8",
    )

    payload = compare_variants([run_dir], tmp_path / "comparison")

    assert payload["runs"][0]["feature_selection_mode"] == "top-positive"
    assert (tmp_path / "comparison" / "comparison.csv").exists()
