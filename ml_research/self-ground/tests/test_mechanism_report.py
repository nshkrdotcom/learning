from __future__ import annotations

import csv

from self_ground.io import write_config, write_jsonl
from self_ground.mechanism_report import build_mechanism_evidence_report


def _write_report_inputs(
    run_dir,
    *,
    compatible: bool = True,
    diagnostic_only: bool = False,
    n_tasks: int = 6,
    top_delta: float = 0.2,
    random_delta: float = 0.05,
    families: list[str] | None = None,
    random_labels: list[str] | None = None,
    operations: list[str] | None = None,
    relative_norm_drift: float = 0.1,
    norm_drift_warning_rate: float = 0.0,
    skipped_rows: int = 0,
    malformed_summary_value: str | None = None,
) -> None:
    families = families or ["sentiment_negation", "property_negation"]
    random_labels = random_labels or ["random_seed_7"]
    operations = operations or ["ablate"]
    write_config(
        {
            "model_name": "test-local",
            "hook_point": "blocks.2.hook_resid_post",
            "sae_release": "test-release",
            "sae_id": "blocks.2.hook_resid_post",
            "allow_metadata_mismatch": diagnostic_only,
            "baseline_mode": "top-vs-random-multiseed",
            "random_seeds": [7, 11, 13],
            "operations": operations,
        },
        run_dir / "config.json",
    )
    write_config(
        {
            "compatible": compatible,
            "metadata_compatible": not diagnostic_only,
            "shape_compatible": True,
            "reconstruction_compatible": True,
            "diagnostic_only": diagnostic_only,
            "reconstruction_mse": 0.1,
            "reconstruction_l2_relative": 0.1,
            "reconstruction_max_abs_error": 0.3,
        },
        run_dir / "compatibility.json",
    )
    write_config(
        {
            "summary": {
                "passes_minimum": True,
                "valid_tasks": n_tasks,
                "valid_by_family": {family: n_tasks // len(families) for family in families},
            }
        },
        run_dir / "behavioral_task_validation.json",
    )
    write_config(
        {
            "feature_sets": [
                {
                    "label": "top",
                    "selection_method": "ranking_abs_score_top_k",
                    "feature_ids": ["sae_0", "sae_1"],
                    "seed": None,
                },
                *[
                    {
                        "label": label,
                        "selection_method": "seeded_random_excluding_top_fraction",
                        "feature_ids": [f"sae_{idx + 3}", f"sae_{idx + 4}"],
                        "seed": int(label.rsplit("_", 1)[-1]),
                    }
                    for idx, label in enumerate(random_labels)
                ],
            ]
        },
        run_dir / "feature_sets.json",
    )
    write_jsonl(
        [
            {
                "family": family,
                "baseline_prompt_contrast": 1.0,
                "baseline_control_contrast": 1.0,
                "intended_direction_pass": True,
            }
            for family in families
        ],
        run_dir / "baseline_task_scores.jsonl",
    )
    fieldnames = [
        "feature_set_label",
        "feature_selection_method",
        "operation",
        "factor",
        "patch_mode",
        "family",
        "n_tasks",
        "target_signed_delta_mean",
        "target_signed_delta_abs_mean",
        "target_absolute_delta_mean",
        "control_signed_delta_mean",
        "control_signed_delta_abs_mean",
        "control_absolute_delta_mean",
        "specificity_gap_mean",
        "collateral_ratio_mean",
        "n_null_collateral_ratio",
        "baseline_contrast_mean",
        "patched_contrast_mean",
        "control_baseline_contrast_mean",
        "control_patched_contrast_mean",
        "target_score_delta_mean",
        "foil_score_delta_mean",
        "relative_norm_drift_mean",
        "decoded_delta_norm_mean",
        "norm_drift_warning_rate",
    ]
    rows = []
    for operation in operations:
        rows.append(
            {
                "feature_set_label": "top",
                "feature_selection_method": "ranking_abs_score_top_k",
                "operation": operation,
                "factor": "" if operation == "ablate" else "2.0",
                "patch_mode": "delta",
                "family": "__all__",
                "n_tasks": n_tasks,
                "target_signed_delta_mean": top_delta,
                "target_signed_delta_abs_mean": abs(top_delta),
                "target_absolute_delta_mean": (
                    malformed_summary_value
                    if malformed_summary_value is not None
                    else abs(top_delta)
                ),
                "control_signed_delta_mean": 0.02,
                "control_signed_delta_abs_mean": 0.02,
                "control_absolute_delta_mean": 0.02,
                "specificity_gap_mean": abs(top_delta) - 0.02,
                "collateral_ratio_mean": 0.1,
                "n_null_collateral_ratio": 0,
                "baseline_contrast_mean": 1.0,
                "patched_contrast_mean": 1.2,
                "control_baseline_contrast_mean": 1.0,
                "control_patched_contrast_mean": 1.02,
                "target_score_delta_mean": 0.1,
                "foil_score_delta_mean": -0.1,
                "relative_norm_drift_mean": relative_norm_drift,
                "decoded_delta_norm_mean": 0.2,
                "norm_drift_warning_rate": norm_drift_warning_rate,
            }
        )
        for label in random_labels:
            rows.append(
                {
                    "feature_set_label": label,
                    "feature_selection_method": "seeded_random_excluding_top_fraction",
                    "operation": operation,
                    "factor": "" if operation == "ablate" else "2.0",
                    "patch_mode": "delta",
                    "family": "__all__",
                    "n_tasks": n_tasks,
                    "target_signed_delta_mean": random_delta,
                    "target_signed_delta_abs_mean": abs(random_delta),
                    "target_absolute_delta_mean": abs(random_delta),
                    "control_signed_delta_mean": 0.01,
                    "control_signed_delta_abs_mean": 0.01,
                    "control_absolute_delta_mean": 0.01,
                    "specificity_gap_mean": abs(random_delta) - 0.01,
                    "collateral_ratio_mean": 0.2,
                    "n_null_collateral_ratio": 0,
                    "baseline_contrast_mean": 1.0,
                    "patched_contrast_mean": 1.05,
                    "control_baseline_contrast_mean": 1.0,
                    "control_patched_contrast_mean": 1.01,
                    "target_score_delta_mean": 0.03,
                    "foil_score_delta_mean": -0.02,
                    "relative_norm_drift_mean": 0.1,
                    "decoded_delta_norm_mean": 0.1,
                    "norm_drift_warning_rate": 0.0,
                }
            )
    with (run_dir / "behavioral_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(
        [{"family": family, "target_absolute_delta": top_delta} for family in families],
        run_dir / "behavioral_intervention_results.jsonl",
    )
    write_config(
        {
            "n_skipped_rows": skipped_rows,
            "reason_counts": {"nonfinite_row_value": skipped_rows} if skipped_rows else {},
            "examples": [],
        },
        run_dir / "skipped_behavioral_rows.json",
    )


def test_blocked_report_does_not_overclaim(tmp_path) -> None:
    _write_report_inputs(tmp_path, compatible=False)

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status == "blocked"
    assert "complete negation mechanism discovery" in report.not_supported_claims[0]


def test_diagnostic_only_cannot_be_candidate(tmp_path) -> None:
    _write_report_inputs(tmp_path, diagnostic_only=True)

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status == "insufficient_evidence"
    assert report.diagnostic_only is True


def test_candidate_report_contains_limitations_and_threshold_checks(tmp_path) -> None:
    _write_report_inputs(tmp_path, n_tasks=8)

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status == "candidate_evidence"
    assert report.feature_sets[0].threshold_checks
    assert report.feature_sets[0].limitations


def test_tiny_run_cannot_be_strong_candidate(tmp_path) -> None:
    _write_report_inputs(tmp_path, n_tasks=6, families=["sentiment_negation", "property_negation"])

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status != "strong_candidate_evidence"


def test_strong_evidence_denied_when_norm_drift_is_high(tmp_path) -> None:
    _write_report_inputs(
        tmp_path,
        n_tasks=30,
        families=["sentiment_negation", "property_negation", "state_negation"],
        operations=["ablate", "amplify"],
        random_labels=["random_seed_7", "random_seed_11", "random_seed_13"],
        relative_norm_drift=0.9,
    )

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status != "strong_candidate_evidence"
    assert any("norm drift" in item.lower() for item in report.limitations)


def test_strong_evidence_denied_when_warning_rate_is_high(tmp_path) -> None:
    _write_report_inputs(
        tmp_path,
        n_tasks=30,
        families=["sentiment_negation", "property_negation", "state_negation"],
        operations=["ablate", "amplify"],
        random_labels=["random_seed_7", "random_seed_11", "random_seed_13"],
        norm_drift_warning_rate=0.8,
    )

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status == "insufficient_evidence"


def test_strong_evidence_uses_actual_random_control_rows_not_config(tmp_path) -> None:
    _write_report_inputs(
        tmp_path,
        n_tasks=30,
        families=["sentiment_negation", "property_negation", "state_negation"],
        operations=["ablate", "amplify"],
        random_labels=["random_seed_7"],
    )

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status != "strong_candidate_evidence"


def test_zero_top_delta_is_insufficient(tmp_path) -> None:
    _write_report_inputs(tmp_path, top_delta=0.0)

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status == "insufficient_evidence"


def test_nonfinite_summary_values_block_report(tmp_path) -> None:
    _write_report_inputs(tmp_path, malformed_summary_value="nan")

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status in {"blocked", "insufficient_evidence"}
    assert report.row_accounting["n_invalid_summary_rows"] > 0


def test_missing_required_artifacts_block_report(tmp_path) -> None:
    write_config({"model_name": "test-local"}, tmp_path / "config.json")

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status == "blocked"
    assert report.qc["required_artifacts_present"] is False


def test_skipped_rows_prevent_strong_evidence(tmp_path) -> None:
    _write_report_inputs(
        tmp_path,
        n_tasks=30,
        families=["sentiment_negation", "property_negation", "state_negation"],
        operations=["ablate", "amplify"],
        random_labels=["random_seed_7", "random_seed_11", "random_seed_13"],
        skipped_rows=1,
    )

    report = build_mechanism_evidence_report(behavioral_run_dir=tmp_path)

    assert report.claim_status == "insufficient_evidence"
    assert report.row_accounting["n_skipped_rows"] == 1


def test_markdown_contains_rerun_commands(tmp_path) -> None:
    _write_report_inputs(tmp_path, n_tasks=8)

    build_mechanism_evidence_report(
        behavioral_run_dir=tmp_path,
        out_md=tmp_path / "mechanism_report.md",
    )

    assert "uv run python scripts/run_phase3_behavioral_evaluation.py" in (
        tmp_path / "mechanism_report.md"
    ).read_text()
